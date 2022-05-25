import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from dee.modules import (
    AttentiveReducer,
    Biaffine,
    DocArgRelInfo,
    EventTableForArgRel,
    MentionTypeEncoder,
    NERModel,
    SentencePosEncoder,
    append_all_spans,
    append_top_span_only,
    bron_kerbosch_pivoting_decode,
    get_doc_arg_rel_info_list,
    get_span_mention_info,
    transformer,
)


class Trans2CompleteGraphModel(nn.Module):
    """LSTM based Multi-Task Learning Document-level Event Extraction Model"""

    def __init__(self, config, event_type_fields_pairs, ner_model=None):
        super(Trans2CompleteGraphModel, self).__init__()
        # Note that for distributed training, you must ensure that
        # for any batch, all parameters need to be used

        self.config = config
        self.event_type_fields_pairs = event_type_fields_pairs

        if ner_model is None:
            self.ner_model = NERModel(config)
        else:
            self.ner_model = ner_model

        # all event tables
        self.event_tables = nn.ModuleList(
            [
                EventTableForArgRel(
                    event_type, field_types, config.hidden_size, min_field_num
                )
                for event_type, field_types, min_field_num in self.event_type_fields_pairs
            ]
        )

        # sentence position indicator
        self.sent_pos_encoder = SentencePosEncoder(
            config.hidden_size, max_sent_num=config.max_sent_num, dropout=config.dropout
        )

        if self.config.use_span_lstm_projection:
            self.start_lstm = nn.LSTM(
                self.config.hidden_size,
                self.config.biaffine_hidden_size // 2,
                num_layers=1,
                bias=True,
                batch_first=True,
                dropout=self.config.dropout,
                bidirectional=True,
            )
            self.end_lstm = nn.LSTM(
                self.config.hidden_size,
                self.config.biaffine_hidden_size // 2,
                num_layers=1,
                bias=True,
                batch_first=True,
                dropout=self.config.dropout,
                bidirectional=True,
            )
        else:
            self.start_mlp = nn.Linear(
                self.config.hidden_size, self.config.biaffine_hidden_size
            )
            self.end_mlp = nn.Linear(
                self.config.hidden_size, self.config.biaffine_hidden_size
            )

        if self.config.use_span_lstm:
            self.span_lstm = nn.LSTM(
                self.config.hidden_size,
                self.config.hidden_size // 2,
                num_layers=self.config.span_lstm_num_layer,
                bias=True,
                batch_first=True,
                dropout=self.config.dropout,
                bidirectional=True,
            )

        if self.config.use_span_att:
            self.span_att = transformer.MultiHeadedAttention(
                self.config.span_att_heads,
                self.config.hidden_size,
                dropout=self.config.dropout,
            )

        # n_out=2: 1: linked 0: non-linked
        self.biaffine = Biaffine(self.config.biaffine_hidden_size, n_out=2)

        if self.config.use_token_role:
            self.ment_type_encoder = MentionTypeEncoder(
                config.hidden_size, config.num_entity_labels, dropout=config.dropout
            )

        # various attentive reducer
        if self.config.seq_reduce_type == "AWA":
            self.doc_token_reducer = AttentiveReducer(
                config.hidden_size, dropout=config.dropout
            )
            self.span_token_reducer = AttentiveReducer(
                config.hidden_size, dropout=config.dropout
            )
            self.span_mention_reducer = AttentiveReducer(
                config.hidden_size, dropout=config.dropout
            )
        else:
            assert self.config.seq_reduce_type in {"MaxPooling", "MeanPooling"}

        if self.config.use_doc_enc:
            # get doc-level context information for every mention and sentence
            self.doc_context_encoder = transformer.make_transformer_encoder(
                config.num_tf_layers,
                config.hidden_size,
                ff_size=config.ff_size,
                dropout=config.dropout,
            )

        self.losses = dict()

    def forward(
        self,
        doc_batch_dict,
        doc_features,
        train_flag=True,
        use_gold_span=False,
        teacher_prob=1,
        event_idx2entity_idx2field_idx=None,
        heuristic_type=None,
    ):
        self.losses = dict()

        # Using scheduled sampling to gradually transit to predicted entity spans
        if train_flag and self.config.use_scheduled_sampling:
            # teacher_prob will gradually decrease outside
            if random.random() < teacher_prob:
                use_gold_span = True
            else:
                use_gold_span = False

        # get doc token-level local context
        (
            doc_token_emb_list,
            doc_token_masks_list,
            doc_token_types_list,
            doc_sent_emb_list,
            doc_sent_loss_list,
        ) = self.get_local_context_info(
            doc_batch_dict,
            train_flag=train_flag,
            use_gold_span=use_gold_span,
        )

        # get doc feature objects
        ex_idx_list = doc_batch_dict["ex_idx"]
        doc_fea_list = [doc_features[ex_idx] for ex_idx in ex_idx_list]

        # get doc span-level info for event extraction
        doc_arg_rel_info_list = get_doc_arg_rel_info_list(
            doc_token_types_list,
            doc_fea_list,
            self.event_type_fields_pairs,
            use_gold_span=use_gold_span,
        )

        if train_flag:
            doc_event_loss_list = []
            for batch_idx, ex_idx in enumerate(ex_idx_list):
                doc_event_loss_list.append(
                    self.get_loss_on_doc(
                        doc_token_emb_list[batch_idx],
                        doc_sent_emb_list[batch_idx],
                        doc_fea_list[batch_idx],
                        doc_arg_rel_info_list[batch_idx],
                    )
                )
            mix_loss = self.get_mix_loss(
                doc_sent_loss_list, doc_event_loss_list, doc_arg_rel_info_list
            )
            self.losses.update({"loss": mix_loss})
            # return mix_loss
            return self.losses
        else:
            # return a list object may not be supported by torch.nn.parallel.DataParallel
            # ensure to run it under the single-gpu mode
            eval_results = []
            for batch_idx, ex_idx in enumerate(ex_idx_list):
                eval_results.append(
                    self.get_eval_on_doc(
                        # self.get_gold_results_on_doc(
                        doc_token_emb_list[batch_idx],
                        doc_sent_emb_list[batch_idx],
                        doc_fea_list[batch_idx],
                        doc_arg_rel_info_list[batch_idx],
                    )
                )
            return eval_results

    def get_local_context_info(
        self, doc_batch_dict, train_flag=False, use_gold_span=False
    ):
        """
        encoder for the raw texts, and get all the token representations and sentence representations
        """
        label_key = "doc_token_labels"
        if train_flag or use_gold_span:
            assert label_key in doc_batch_dict
            need_label_flag = True
        else:
            need_label_flag = False

        if need_label_flag:  # tzhu: `label` means the role label
            doc_token_labels_list = self.adjust_token_label(doc_batch_dict[label_key])
        else:
            doc_token_labels_list = None

        batch_size = len(doc_batch_dict["ex_idx"])
        doc_token_ids_list = doc_batch_dict["doc_token_ids"]
        doc_token_masks_list = doc_batch_dict["doc_token_masks"]
        valid_sent_num_list = doc_batch_dict["valid_sent_num"]

        # transform doc_batch into sent_batch
        ner_batch_idx_start_list = [0]
        ner_token_ids = []
        ner_token_masks = []
        ner_token_labels = [] if need_label_flag else None
        for batch_idx, valid_sent_num in enumerate(valid_sent_num_list):
            idx_start = ner_batch_idx_start_list[-1]
            idx_end = idx_start + valid_sent_num
            ner_batch_idx_start_list.append(idx_end)

            ner_token_ids.append(doc_token_ids_list[batch_idx])
            ner_token_masks.append(doc_token_masks_list[batch_idx])
            if need_label_flag:
                ner_token_labels.append(doc_token_labels_list[batch_idx])

        # [ner_batch_size, norm_sent_len]
        ner_token_ids = torch.cat(ner_token_ids, dim=0)
        ner_token_masks = torch.cat(ner_token_masks, dim=0)
        if need_label_flag:
            ner_token_labels = torch.cat(ner_token_labels, dim=0)

        # get ner output
        ner_token_emb, ner_loss, ner_token_preds = self.ner_model(
            ner_token_ids,
            ner_token_masks,
            label_ids=ner_token_labels,
            train_flag=train_flag,
            decode_flag=not use_gold_span,
        )

        if use_gold_span:  # definitely use gold span info
            ner_token_types = ner_token_labels
        else:
            ner_token_types = ner_token_preds

        # get sentence embedding
        ner_sent_emb = self.get_batch_sent_emb(
            ner_token_emb, ner_token_masks, valid_sent_num_list
        )

        assert sum(valid_sent_num_list) == ner_token_emb.size(0) == ner_sent_emb.size(0)

        # followings are all lists of tensors
        doc_token_emb_list = []
        doc_token_masks_list = []
        doc_token_types_list = []
        doc_sent_emb_list = []
        doc_sent_loss_list = []
        for batch_idx in range(batch_size):
            idx_start = ner_batch_idx_start_list[batch_idx]
            idx_end = ner_batch_idx_start_list[batch_idx + 1]
            doc_token_emb_list.append(ner_token_emb[idx_start:idx_end, :, :])
            doc_token_masks_list.append(ner_token_masks[idx_start:idx_end, :])
            doc_token_types_list.append(ner_token_types[idx_start:idx_end, :])
            doc_sent_emb_list.append(ner_sent_emb[idx_start:idx_end, :])
            if ner_loss is not None:
                # every doc_sent_loss.size is [valid_sent_num]
                doc_sent_loss_list.append(ner_loss[idx_start:idx_end])

        return (
            doc_token_emb_list,
            doc_token_masks_list,
            doc_token_types_list,
            doc_sent_emb_list,
            doc_sent_loss_list,
        )

    def get_doc_span_mention_emb(self, doc_token_emb, doc_arg_rel_info):
        """
        get all the mention representations by aggregating the token representations
        """
        if len(doc_arg_rel_info.mention_drange_list) == 0:
            doc_mention_emb = None
        else:
            mention_emb_list = []
            for sent_idx, char_s, char_e in doc_arg_rel_info.mention_drange_list:
                mention_token_emb = doc_token_emb[
                    sent_idx, char_s:char_e, :
                ]  # [num_mention_tokens, hidden_size]
                if self.config.seq_reduce_type == "AWA":
                    mention_emb = self.span_token_reducer(
                        mention_token_emb
                    )  # [hidden_size]
                elif self.config.seq_reduce_type == "MaxPooling":
                    mention_emb = mention_token_emb.max(dim=0)[0]
                elif self.config.seq_reduce_type == "MeanPooling":
                    mention_emb = mention_token_emb.mean(dim=0)
                else:
                    raise Exception(
                        "Unknown seq_reduce_type {}".format(self.config.seq_reduce_type)
                    )
                mention_emb_list.append(mention_emb)
            doc_mention_emb = torch.stack(mention_emb_list, dim=0)

            # add sentence position embedding
            mention_sent_id_list = [
                drange[0] for drange in doc_arg_rel_info.mention_drange_list
            ]
            doc_mention_emb = self.sent_pos_encoder(
                doc_mention_emb, sent_pos_ids=mention_sent_id_list
            )

            if self.config.use_token_role:
                # get mention type embedding
                doc_mention_emb = self.ment_type_encoder(
                    doc_mention_emb, doc_arg_rel_info.mention_type_list
                )

        return doc_mention_emb

    def get_batch_sent_emb(self, ner_token_emb, ner_token_masks, valid_sent_num_list):
        # From [ner_batch_size, sent_len, hidden_size] to [ner_batch_size, hidden_size]
        if self.config.seq_reduce_type == "AWA":
            total_sent_emb = self.doc_token_reducer(
                ner_token_emb, masks=ner_token_masks
            )
        elif self.config.seq_reduce_type == "MaxPooling":
            total_sent_emb = ner_token_emb.max(dim=1)[0]
        elif self.config.seq_reduce_type == "MeanPooling":
            total_sent_emb = ner_token_emb.mean(dim=1)
        else:
            raise Exception(
                "Unknown seq_reduce_type {}".format(self.config.seq_reduce_type)
            )

        total_sent_pos_ids = []
        for valid_sent_num in valid_sent_num_list:
            total_sent_pos_ids += list(range(valid_sent_num))
        total_sent_emb = self.sent_pos_encoder(
            total_sent_emb, sent_pos_ids=total_sent_pos_ids
        )

        return total_sent_emb

    def get_doc_span_sent_context(
        self, doc_token_emb, doc_sent_emb, doc_fea, doc_arg_rel_info
    ):
        """
        get all the span representations by aggregating mention representations,
        and sentence representations
        """
        doc_mention_emb = self.get_doc_span_mention_emb(doc_token_emb, doc_arg_rel_info)

        # only consider actual sentences
        if doc_sent_emb.size(0) > doc_fea.valid_sent_num:
            doc_sent_emb = doc_sent_emb[: doc_fea.valid_sent_num, :]

        span_context_list = []

        if doc_mention_emb is None:
            if self.config.use_doc_enc:
                doc_sent_context = self.doc_context_encoder(
                    doc_sent_emb.unsqueeze(0), None
                ).squeeze(0)
            else:
                doc_sent_context = doc_sent_emb
        else:
            num_mentions = doc_mention_emb.size(0)

            if self.config.use_doc_enc:
                # Size([1, num_mentions + num_valid_sents, hidden_size])
                total_ment_sent_emb = torch.cat(
                    [doc_mention_emb, doc_sent_emb], dim=0
                ).unsqueeze(0)

                # size = [num_mentions+num_valid_sents, hidden_size]
                # here we do not need mask
                total_ment_sent_context = self.doc_context_encoder(
                    total_ment_sent_emb, None
                ).squeeze(0)

                # collect span context
                for mid_s, mid_e in doc_arg_rel_info.span_mention_range_list:
                    assert mid_e <= num_mentions
                    multi_ment_context = total_ment_sent_context[
                        mid_s:mid_e
                    ]  # [num_mentions, hidden_size]

                    # span_context.size [1, hidden_size]
                    if self.config.seq_reduce_type == "AWA":
                        span_context = self.span_mention_reducer(
                            multi_ment_context, keepdim=True
                        )
                    elif self.config.seq_reduce_type == "MaxPooling":
                        span_context = multi_ment_context.max(dim=0, keepdim=True)[0]
                    elif self.config.seq_reduce_type == "MeanPooling":
                        span_context = multi_ment_context.mean(dim=0, keepdim=True)
                    else:
                        raise Exception(
                            "Unknown seq_reduce_type {}".format(
                                self.config.seq_reduce_type
                            )
                        )

                    span_context_list.append(span_context)

                # collect sent context
                doc_sent_context = total_ment_sent_context[num_mentions:, :]
            else:
                # collect span context
                for mid_s, mid_e in doc_arg_rel_info.span_mention_range_list:
                    assert mid_e <= num_mentions
                    multi_ment_emb = doc_mention_emb[
                        mid_s:mid_e
                    ]  # [num_mentions, hidden_size]

                    # span_context.size is [1, hidden_size]
                    if self.config.seq_reduce_type == "AWA":
                        span_context = self.span_mention_reducer(
                            multi_ment_emb, keepdim=True
                        )
                    elif self.config.seq_reduce_type == "MaxPooling":
                        span_context = multi_ment_emb.max(dim=0, keepdim=True)[0]
                    elif self.config.seq_reduce_type == "MeanPooling":
                        span_context = multi_ment_emb.mean(dim=0, keepdim=True)
                    else:
                        raise Exception(
                            "Unknown seq_reduce_type {}".format(
                                self.config.seq_reduce_type
                            )
                        )
                    span_context_list.append(span_context)

                # collect sent context
                doc_sent_context = doc_sent_emb

        return span_context_list, doc_sent_context

    def get_event_cls_info(self, sent_context_emb, doc_fea, train_flag=True):
        """
        get the event type classification results

        Args:
            train_flag: if True, return the sum of cross entropy loss
                else return the predicted result in List format
        """
        doc_event_logps = []
        for event_idx, event_label in enumerate(doc_fea.event_type_labels):
            event_table = self.event_tables[event_idx]
            cur_event_logp = event_table(
                sent_context_emb=sent_context_emb
            )  # [1, hidden_size]
            doc_event_logps.append(cur_event_logp)
        doc_event_logps = torch.cat(doc_event_logps, dim=0)  # [num_event_types, 2]

        if train_flag:
            device = doc_event_logps.device
            doc_event_labels = torch.tensor(
                doc_fea.event_type_labels,
                device=device,
                dtype=torch.long,
                requires_grad=False,
            )  # [num_event_types]
            doc_event_cls_loss = F.nll_loss(
                doc_event_logps, doc_event_labels, reduction="sum"
            )
            return doc_event_cls_loss
        else:
            doc_event_pred_list = doc_event_logps.argmax(dim=-1).tolist()
            return doc_event_pred_list

    def get_none_span_context(self, init_tensor):
        none_span_context = torch.zeros(
            1,
            self.config.hidden_size,
            device=init_tensor.device,
            dtype=init_tensor.dtype,
            requires_grad=False,
        )
        return none_span_context

    def get_arg_combination_loss(self, biaffine_out, doc_arg_rel_info, event_idx=None):
        if self.config.event_relevant_combination:
            rel_adj_mat = torch.tensor(
                doc_arg_rel_info.event_arg_rel_mats[event_idx].reveal_adj_mat(),
                dtype=torch.long,
                device=biaffine_out.device,
                requires_grad=False,
            )
        else:
            rel_adj_mat = torch.tensor(
                doc_arg_rel_info.whole_arg_rel_mat.reveal_adj_mat(),
                dtype=torch.long,
                device=biaffine_out.device,
                requires_grad=False,
            )
        # [1, 2, num_span, num_span] -> [num_span*num_span, 2]
        result_tensor = biaffine_out.squeeze(0).permute(1, 2, 0).view(-1, 2)
        # [num_span, num_span] -> [num_span*num_span]
        rel_adj_mat = rel_adj_mat.view(-1)
        # ignore_index: diag line elements in target rel_adj_mat, not included in calculation
        if self.config.add_adj_mat_weight_bias:
            combination_loss = F.cross_entropy(
                result_tensor,
                rel_adj_mat,
                ignore_index=-1,
                reduction="sum",
                weight=torch.tensor([1.0, 0.5], device=biaffine_out.device),
            )
        else:
            combination_loss = F.cross_entropy(
                result_tensor, rel_adj_mat, ignore_index=-1, reduction="sum"
            )
        return combination_loss

    def get_arg_role_loss(self, arg_role_logits, role_types):
        role_types_tensor = torch.tensor(
            role_types,
            dtype=torch.long,
            device=arg_role_logits.device,
            requires_grad=False,
        )
        role_loss = F.cross_entropy(arg_role_logits, role_types_tensor, reduction="sum")
        return role_loss

    def get_loss_on_doc(self, doc_token_emb, doc_sent_emb, doc_fea, doc_arg_rel_info):
        span_context_list, doc_sent_context = self.get_doc_span_sent_context(
            doc_token_emb,
            doc_sent_emb,
            doc_fea,
            doc_arg_rel_info,
        )
        if len(span_context_list) == 0:
            raise Exception(
                "Error: doc_fea.ex_idx {} does not have valid span".format(
                    doc_fea.ex_idx
                )
            )
        # 0. get span representations
        batch_span_context = torch.cat(span_context_list, dim=0)
        if self.config.use_span_lstm:
            batch_span_context = batch_span_context.unsqueeze(0)
            batch_span_context, (_, _) = self.span_lstm(batch_span_context)
            batch_span_context = batch_span_context.squeeze(0)

        if self.config.use_span_att:
            batch_span_context = self.span_att(
                batch_span_context, batch_span_context, batch_span_context
            )
            batch_span_context = batch_span_context.squeeze(1)

        # 1. get event type classification loss
        event_cls_loss = self.get_event_cls_info(
            doc_sent_context, doc_fea, train_flag=True
        )

        # 2. for each event type, get argument combination loss
        # argument combination loss, calculated by comparing
        # the biaffine output and the gold event SpanArgRelAdjMat
        arg_combination_loss = []
        arg_role_loss = []
        if not self.config.event_relevant_combination:
            # 2. combination loss via biaffine
            biaffine_out = self.get_adj_mat_logits(batch_span_context)
            assert (
                biaffine_out.shape[-1] == doc_arg_rel_info.whole_arg_rel_mat.len_spans
            )
            comb_loss = self.get_arg_combination_loss(
                biaffine_out, doc_arg_rel_info, event_idx=None
            )
            arg_combination_loss.append(comb_loss)

            for event_idx, event_label in enumerate(doc_fea.event_type_labels):
                event_table = self.event_tables[event_idx]
                # 3. role classification loss
                events = doc_arg_rel_info.pred_event_arg_idxs_objs_list[event_idx]
                # only update role type loss when event_label == 1
                if events is not None:
                    for event_instance in events:
                        span_idxs = []
                        role_types = []
                        span_rep_list_for_event_instance = []
                        for span_idx, role_type in event_instance:
                            span_idxs.append(span_idx)
                            role_types.append(role_type)
                            span_rep_list_for_event_instance.append(
                                span_context_list[span_idx]
                            )
                        span_rep_for_event_instance = torch.cat(
                            span_rep_list_for_event_instance, dim=0
                        )
                        role_cls_logits = event_table(
                            batch_span_emb=span_rep_for_event_instance
                        )
                        # be aware that the gold here are intersection between gold
                        # and predictions if using scheduling sampling, and the gold labels
                        # have been changed during the doc_arg_rel_info generation
                        role_loss = self.get_arg_role_loss(role_cls_logits, role_types)
                        arg_role_loss.append(role_loss)
        else:  # event-relevant combination, attention between event representation and batch_span_context output
            for event_idx, event_label in enumerate(doc_fea.event_type_labels):
                event_table = self.event_tables[event_idx]
                # attention with the event representation
                batch_span_context = batch_span_context.unsqueeze(1)  # [Ns, 1, H]
                batch_span_context, _ = transformer.attention(
                    event_table.event_query, batch_span_context, batch_span_context
                )
                batch_span_context = batch_span_context.squeeze(1)  # [Ns, H]

                # combination loss via biaffine
                biaffine_out = self.get_adj_mat_logits(batch_span_context)
                assert (
                    biaffine_out.shape[-1]
                    == doc_arg_rel_info.event_arg_rel_mats[event_idx].len_spans
                )
                comb_loss = self.get_arg_combination_loss(
                    biaffine_out, doc_arg_rel_info, event_idx=event_idx
                )
                arg_combination_loss.append(comb_loss)

                # 3. role classification loss
                events = doc_arg_rel_info.pred_event_arg_idxs_objs_list[event_idx]
                # only update role type loss when event_label == 1
                if events is not None:
                    for event_instance in events:
                        span_idxs = []
                        role_types = []
                        span_rep_list_for_event_instance = []
                        for span_idx, role_type in event_instance:
                            span_idxs.append(span_idx)
                            role_types.append(role_type)
                            span_rep_list_for_event_instance.append(
                                span_context_list[span_idx]
                            )
                        span_rep_for_event_instance = torch.cat(
                            span_rep_list_for_event_instance, dim=0
                        )
                        role_cls_logits = event_table(
                            batch_span_emb=span_rep_for_event_instance
                        )
                        # be aware that the gold here are intersection between gold
                        # and predictions if using scheduling sampling, and the gold labels
                        # have been changed during the doc_arg_rel_info generation
                        role_loss = self.get_arg_role_loss(role_cls_logits, role_types)
                        arg_role_loss.append(role_loss)

        self.losses.update(
            {
                "event_cls": event_cls_loss,
                "arg_combination_loss": sum(arg_combination_loss),
                "arg_role_loss": sum(arg_role_loss),
            }
        )
        return (
            self.config.event_cls_loss_weight * event_cls_loss
            + self.config.combination_loss_weight * sum(arg_combination_loss)
            + self.config.role_loss_weight * sum(arg_role_loss)
        )

    def get_adj_mat_logits(self, batch_span_context):
        if self.config.use_span_lstm_projection:
            batch_span_context = batch_span_context.unsqueeze(0)
            start_hidden, (_, _) = self.start_lstm(batch_span_context)
            end_hidden, (_, _) = self.end_lstm(batch_span_context)
        else:
            start_hidden = self.start_mlp(batch_span_context)
            end_hidden = self.end_mlp(batch_span_context)
            start_hidden = start_hidden.unsqueeze(0)
            end_hidden = end_hidden.unsqueeze(0)
        biaffine_out = self.biaffine(start_hidden, end_hidden)
        return biaffine_out

    def get_mix_loss(
        self, doc_sent_loss_list, doc_event_loss_list, doc_arg_rel_info_list
    ):
        batch_size = len(doc_arg_rel_info_list)
        loss_batch_avg = 1.0 / batch_size
        lambda_1 = self.config.loss_lambda
        lambda_2 = 1 - lambda_1

        doc_ner_loss_list = []
        for doc_sent_loss, doc_span_info in zip(
            doc_sent_loss_list, doc_arg_rel_info_list
        ):
            # doc_sent_loss: Size([num_valid_sents])
            doc_ner_loss_list.append(doc_sent_loss.sum())

        self.losses.update({"ner_loss": sum(doc_ner_loss_list)})
        return loss_batch_avg * (
            lambda_1 * sum(doc_ner_loss_list) + lambda_2 * sum(doc_event_loss_list)
        )

    def get_eval_on_doc(self, doc_token_emb, doc_sent_emb, doc_fea, doc_arg_rel_info):
        """
        Get the final evaluation results (prediction process).
        To unify the evaluation process, the format of output
        event_arg_idxs_objs will stay the same with EDAG.
        Since the `event_idx2event_decode_paths` is not used
        in evaluation, we'll change it to predicted adj_mat
        and adj_decoding combinations.
        """
        final_pred_adj_mat = []
        event_idx2combinations = []

        span_context_list, doc_sent_context = self.get_doc_span_sent_context(
            doc_token_emb, doc_sent_emb, doc_fea, doc_arg_rel_info
        )
        if len(span_context_list) == 0:
            event_pred_list = []
            event_idx2obj_idx2field_idx2token_tup = []
            for event_idx in range(len(self.event_type_fields_pairs)):
                event_pred_list.append(0)
                event_idx2obj_idx2field_idx2token_tup.append(None)

            return (
                doc_fea.ex_idx,
                event_pred_list,
                event_idx2obj_idx2field_idx2token_tup,
                doc_arg_rel_info,
                final_pred_adj_mat,
                event_idx2combinations,
            )

        # 1. get event type prediction
        event_pred_list = self.get_event_cls_info(
            doc_sent_context, doc_fea, train_flag=False
        )

        # 2. for each event type, get argument relation adjacent matrix
        batch_span_context = torch.cat(span_context_list, dim=0)
        if self.config.use_span_lstm:
            batch_span_context = batch_span_context.unsqueeze(0)
            batch_span_context, (_, _) = self.span_lstm(batch_span_context)
            batch_span_context = batch_span_context.squeeze(0)

        if self.config.use_span_att:
            batch_span_context = self.span_att(
                batch_span_context, batch_span_context, batch_span_context
            )
            batch_span_context = batch_span_context.squeeze(1)

        if not self.config.event_relevant_combination:
            biaffine_out = self.get_adj_mat_logits(batch_span_context)
            biaffine_out = biaffine_out.squeeze(0)
            pred_adj_mat = biaffine_out.max(dim=0)[1]
            pred_adj_mat = self.pred_adj_mat_reorgnise(pred_adj_mat)
            assert (
                pred_adj_mat.shape[-1] == doc_arg_rel_info.whole_arg_rel_mat.len_spans
            )
            # debug mode statement only for time saving
            if self.config.run_mode == "debug":
                pred_adj_mat = pred_adj_mat[:10, :10]

            # # only for 100% filled graph testing
            # pred_adj_mat = torch.ones((batch_span_context.shape[0], batch_span_context.shape[0]))
            # # end of testing

            pred_adj_mat = pred_adj_mat.detach().cpu().tolist()
            final_pred_adj_mat.append(pred_adj_mat)
            raw_combinations = bron_kerbosch_pivoting_decode(pred_adj_mat, 3)

            event_idx2obj_idx2field_idx2token_tup = []
            for event_idx, event_pred in enumerate(event_pred_list):
                if event_pred == 0:
                    event_idx2obj_idx2field_idx2token_tup.append(None)
                    continue
                event_table = self.event_tables[event_idx]
                # TODO(tzhu): m2m support from all the combinations
                # combinations = list(filter(lambda x: len(x) >= event_table.min_field_num, raw_combinations))
                combinations = copy.deepcopy(raw_combinations)
                event_idx2combinations.append(combinations)
                if len(combinations) <= 0:
                    event_idx2obj_idx2field_idx2token_tup.append(None)
                    continue
                obj_idx2field_idx2token_tup = []
                for combination in combinations:
                    span_rep_list_for_event_instance = []
                    for span_idx in combination:
                        span_rep_list_for_event_instance.append(
                            span_context_list[span_idx]
                        )
                    span_rep_for_event_instance = torch.cat(
                        span_rep_list_for_event_instance, dim=0
                    )
                    role_cls_logits = event_table(
                        batch_span_emb=span_rep_for_event_instance
                    )
                    role_preds = role_cls_logits.max(dim=-1)[1].detach().cpu().tolist()
                    event_arg_obj = self.reveal_event_arg_obj(
                        combination, role_preds, event_table.num_fields
                    )
                    field_idx2token_tup = self.convert_span_idx_to_token_tup(
                        event_arg_obj, doc_arg_rel_info
                    )
                    obj_idx2field_idx2token_tup.append(field_idx2token_tup)
                event_idx2obj_idx2field_idx2token_tup.append(
                    obj_idx2field_idx2token_tup
                )
            # the first three terms are for metric calculation, the last two are for case studies
            return (
                doc_fea.ex_idx,
                event_pred_list,
                event_idx2obj_idx2field_idx2token_tup,
                doc_arg_rel_info,
                final_pred_adj_mat,
                event_idx2combinations,
            )
        else:
            event_idx2obj_idx2field_idx2token_tup = []
            for event_idx, event_pred in enumerate(event_pred_list):
                if event_pred == 0:
                    event_idx2obj_idx2field_idx2token_tup.append(None)
                    continue
                event_table = self.event_tables[event_idx]

                # attention with the event representation
                batch_span_context = batch_span_context.unsqueeze(1)  # [Ns, 1, H]
                batch_span_context, _ = transformer.attention(
                    event_table.event_query, batch_span_context, batch_span_context
                )
                batch_span_context = batch_span_context.squeeze(1)  # [Ns, H]

                biaffine_out = self.get_adj_mat_logits(batch_span_context)
                biaffine_out = biaffine_out.squeeze(0)
                pred_adj_mat = biaffine_out.max(dim=0)[1]
                pred_adj_mat = self.pred_adj_mat_reorgnise(pred_adj_mat)

                # debug mode statement only for time saving
                if self.config.run_mode == "debug":
                    pred_adj_mat = pred_adj_mat[:10, :10]

                pred_adj_mat = pred_adj_mat.detach().cpu().tolist()
                final_pred_adj_mat.append(pred_adj_mat)
                raw_combinations = bron_kerbosch_pivoting_decode(pred_adj_mat, 3)

                combinations = list(
                    filter(
                        lambda x: len(x) > event_table.min_field_num, raw_combinations
                    )
                )
                event_idx2combinations.append(combinations)
                if len(combinations) <= 0:
                    event_idx2obj_idx2field_idx2token_tup.append(None)
                    continue
                obj_idx2field_idx2token_tup = []
                for combination in combinations:
                    span_rep_list_for_event_instance = []
                    for span_idx in combination:
                        span_rep_list_for_event_instance.append(
                            span_context_list[span_idx]
                        )
                    span_rep_for_event_instance = torch.cat(
                        span_rep_list_for_event_instance, dim=0
                    )
                    role_cls_logits = event_table(
                        batch_span_emb=span_rep_for_event_instance
                    )
                    role_preds = role_cls_logits.max(dim=-1)[1].detach().cpu().tolist()
                    event_arg_obj = self.reveal_event_arg_obj(
                        combination, role_preds, event_table.num_fields
                    )
                    field_idx2token_tup = self.convert_span_idx_to_token_tup(
                        event_arg_obj, doc_arg_rel_info
                    )
                    obj_idx2field_idx2token_tup.append(field_idx2token_tup)
                event_idx2obj_idx2field_idx2token_tup.append(
                    obj_idx2field_idx2token_tup
                )

            # the first three terms are for metric calculation, the last two are for case studies
            return (
                doc_fea.ex_idx,
                event_pred_list,
                event_idx2obj_idx2field_idx2token_tup,
                doc_arg_rel_info,
                final_pred_adj_mat,
                event_idx2combinations,
            )

    def get_gold_results_on_doc(
        self, doc_token_emb, doc_sent_emb, doc_fea, doc_arg_rel_info
    ):
        """
        Replace `get_eval_on_doc` to get the algorithm's upper bound.
        """
        final_pred_adj_mat = []
        event_idx2combinations = []

        (
            span_mention_range_list,
            mention_drange_list,
            mention_type_list,
        ) = get_span_mention_info(
            doc_fea.span_dranges_list, doc_fea.doc_token_labels.detach().cpu().tolist()
        )

        doc_arg_rel_info = DocArgRelInfo(
            doc_fea.span_token_ids_list,
            [
                x in doc_fea.exist_span_token_tup_set
                for x in doc_fea.span_token_ids_list
            ],
            # will not affect the combination results under gold setting (only affect overall predictions)
            [
                x in doc_fea.exist_span_token_tup_set
                for x in doc_fea.span_token_ids_list
            ],
            doc_fea.span_token_ids_list,
            doc_fea.span_dranges_list,
            span_mention_range_list,
            mention_drange_list,
            mention_type_list,
            doc_fea.span_rel_mats,
            doc_fea.whole_arg_rel_mat,
            doc_fea.event_arg_idxs_objs_list,
            doc_arg_rel_info.missed_span_idx_list,
            doc_arg_rel_info.missed_sent_idx_list,
        )

        span_context_list, doc_sent_context = self.get_doc_span_sent_context(
            doc_token_emb, doc_sent_emb, doc_fea, doc_arg_rel_info
        )
        if len(span_context_list) == 0:
            event_pred_list = []
            event_idx2obj_idx2field_idx2token_tup = []
            for event_idx in range(len(self.event_type_fields_pairs)):
                event_pred_list.append(0)
                event_idx2obj_idx2field_idx2token_tup.append(None)

            return (
                doc_fea.ex_idx,
                event_pred_list,
                event_idx2obj_idx2field_idx2token_tup,
                doc_arg_rel_info,
                final_pred_adj_mat,
                event_idx2combinations,
            )

        # 1. get event type prediction
        event_pred_list = doc_fea.event_type_labels

        # 2. for each event type, get argument relation adjacent matrix
        batch_span_context = torch.cat(span_context_list, dim=0)
        if self.config.use_span_lstm:
            batch_span_context = batch_span_context.unsqueeze(0)
            batch_span_context, (_, _) = self.span_lstm(batch_span_context)
            batch_span_context = batch_span_context.squeeze(0)

        if self.config.use_span_att:
            batch_span_context = self.span_att(
                batch_span_context, batch_span_context, batch_span_context
            )
            batch_span_context = batch_span_context.squeeze(1)

        if not self.config.event_relevant_combination:
            pred_adj_mat = torch.tensor(
                doc_fea.whole_arg_rel_mat.reveal_adj_mat(),
                device=batch_span_context.device,
                requires_grad=False,
            )
            pred_adj_mat = self.pred_adj_mat_reorgnise(pred_adj_mat)
            pred_adj_mat = pred_adj_mat.detach().cpu().tolist()
            final_pred_adj_mat.append(pred_adj_mat)
            raw_combinations = bron_kerbosch_pivoting_decode(pred_adj_mat, 3)
            # raw_combinations = brute_force_adj_decode(pred_adj_mat, 3)
            # pred_combinations = set(raw_combinations)
            # gold_combinations = extract_combinations_from_event_objs(doc_fea.event_arg_idxs_objs_list)
            # gold_combinations = remove_combination_roles(gold_combinations)
            # TP = len(pred_combinations & gold_combinations)
            # FP = len(pred_combinations - gold_combinations)
            # FN = len(gold_combinations - pred_combinations)
            # p, r, f1 = get_prec_recall_f1(TP, FP, FN)
            # if r < 1.0 and len(sorted(gold_combinations, key=lambda x: len(x), reverse=True)[0]) > 3:
            #     breakpoint()
            #     plot_graph_from_adj_mat(pred_adj_mat, title="gold")

            event_idx2obj_idx2field_idx2token_tup = []
            for event_idx, event_pred in enumerate(event_pred_list):
                if event_pred == 0:
                    event_idx2obj_idx2field_idx2token_tup.append(None)
                    continue
                event_table = self.event_tables[event_idx]
                # TODO(tzhu): m2m support from all the combinations
                combinations = list(filter(lambda x: len(x) >= 3, raw_combinations))
                # combinations = list(filter(lambda x: len(x) >= event_table.min_field_num, raw_combinations))
                combinations = raw_combinations
                event_idx2combinations.append(combinations)
                if len(combinations) <= 0:
                    event_idx2obj_idx2field_idx2token_tup.append(None)
                    continue
                obj_idx2field_idx2token_tup = []
                for combination in combinations:
                    span_rep_list_for_event_instance = []
                    for span_idx in combination:
                        span_rep_list_for_event_instance.append(
                            span_context_list[span_idx]
                        )
                    span_rep_for_event_instance = torch.cat(
                        span_rep_list_for_event_instance, dim=0
                    )
                    role_cls_logits = event_table(
                        batch_span_emb=span_rep_for_event_instance
                    )
                    role_preds = role_cls_logits.max(dim=-1)[1].detach().cpu().tolist()
                    event_arg_obj = self.reveal_event_arg_obj(
                        combination, role_preds, event_table.num_fields
                    )
                    field_idx2token_tup = self.convert_span_idx_to_token_tup(
                        event_arg_obj, doc_arg_rel_info
                    )
                    obj_idx2field_idx2token_tup.append(field_idx2token_tup)
                event_idx2obj_idx2field_idx2token_tup.append(
                    obj_idx2field_idx2token_tup
                )
            # the first three terms are for metric calculation, the last two are for case studies
            return (
                doc_fea.ex_idx,
                event_pred_list,
                event_idx2obj_idx2field_idx2token_tup,
                doc_arg_rel_info,
                final_pred_adj_mat,
                event_idx2combinations,
            )
        else:
            event_idx2obj_idx2field_idx2token_tup = []
            for event_idx, event_pred in enumerate(event_pred_list):
                if event_pred == 0:
                    event_idx2obj_idx2field_idx2token_tup.append(None)
                    continue
                event_table = self.event_tables[event_idx]

                # attention with the event representation
                batch_span_context = batch_span_context.unsqueeze(1)  # [Ns, 1, H]
                batch_span_context, _ = transformer.attention(
                    event_table.event_query, batch_span_context, batch_span_context
                )
                batch_span_context = batch_span_context.squeeze(1)  # [Ns, H]

                pred_adj_mat = torch.tensor(
                    doc_fea.whole_arg_rel_mat.reveal_adj_mat(),
                    device=batch_span_context.device,
                )
                pred_adj_mat = self.pred_adj_mat_reorgnise(pred_adj_mat)

                # debug mode statement only for time saving
                if self.config.run_mode == "debug":
                    pred_adj_mat = pred_adj_mat[:10, :10]

                pred_adj_mat = pred_adj_mat.detach().cpu().tolist()
                final_pred_adj_mat.append(pred_adj_mat)
                raw_combinations = bron_kerbosch_pivoting_decode(pred_adj_mat, 3)
                # raw_combinations = brute_force_adj_decode(pred_adj_mat, 3)

                combinations = list(
                    filter(
                        lambda x: len(x) >= event_table.min_field_num, raw_combinations
                    )
                )
                event_idx2combinations.append(combinations)
                if len(combinations) <= 0:
                    event_idx2obj_idx2field_idx2token_tup.append(None)
                    continue
                obj_idx2field_idx2token_tup = []
                for combination in combinations:
                    span_rep_list_for_event_instance = []
                    for span_idx in combination:
                        span_rep_list_for_event_instance.append(
                            span_context_list[span_idx]
                        )
                    span_rep_for_event_instance = torch.cat(
                        span_rep_list_for_event_instance, dim=0
                    )
                    role_cls_logits = event_table(
                        batch_span_emb=span_rep_for_event_instance
                    )
                    role_preds = role_cls_logits.max(dim=-1)[1].detach().cpu().tolist()
                    event_arg_obj = self.reveal_event_arg_obj(
                        combination, role_preds, event_table.num_fields
                    )
                    field_idx2token_tup = self.convert_span_idx_to_token_tup(
                        event_arg_obj, doc_arg_rel_info
                    )
                    obj_idx2field_idx2token_tup.append(field_idx2token_tup)
                event_idx2obj_idx2field_idx2token_tup.append(
                    obj_idx2field_idx2token_tup
                )

            # the first three terms are for metric calculation, the last two are for case studies
            return (
                doc_fea.ex_idx,
                event_pred_list,
                event_idx2obj_idx2field_idx2token_tup,
                doc_arg_rel_info,
                final_pred_adj_mat,
                event_idx2combinations,
            )

    def convert_span_idx_to_token_tup(self, event_arg_obj, doc_arg_rel_info):
        results = []
        for span_idx in event_arg_obj:
            if span_idx is None:
                results.append(None)
            else:
                results.append(doc_arg_rel_info.span_token_tup_list[span_idx])
        return results

    def reveal_event_arg_obj(self, combination, roles, num_fields):
        ret_results = [None] * num_fields
        results = list(zip(combination, roles))
        results.sort(key=lambda x: x[1])
        for span_idx, role in results:
            ret_results[role] = span_idx
        return ret_results

    def pred_adj_mat_reorgnise(self, pred_adj_mat):
        """
        fill the diag to 1 and make sure the adj_mat is symmetric
        """
        adj_mat = torch.bitwise_or(pred_adj_mat, pred_adj_mat.T)
        # eye_tensor = torch.eye(adj_mat.shape[0], dtype=adj_mat.dtype, device=adj_mat.device, requires_grad=False)
        # adj_mat = torch.bitwise_or(adj_mat, eye_tensor)
        if adj_mat.dim() <= 1:
            adj_mat = adj_mat.unsqueeze(0)
        adj_mat.fill_diagonal_(0)
        return adj_mat

    def adjust_token_label(self, doc_token_labels_list):
        # tzhu: whether the BI labels has a role or not
        if self.config.use_token_role:  # do not use detailed token
            return doc_token_labels_list
        else:
            adj_doc_token_labels_list = []
            for doc_token_labels in doc_token_labels_list:
                entity_begin_mask = doc_token_labels % 2 == 1
                entity_inside_mask = (doc_token_labels != 0) & (
                    doc_token_labels % 2 == 0
                )
                adj_doc_token_labels = doc_token_labels.masked_fill(
                    entity_begin_mask, 1
                )
                adj_doc_token_labels = adj_doc_token_labels.masked_fill(
                    entity_inside_mask, 2
                )

                adj_doc_token_labels_list.append(adj_doc_token_labels)
            return adj_doc_token_labels_list

    def heuristic_decode_on_doc(
        self,
        doc_token_emb,
        doc_sent_emb,
        doc_fea,
        doc_span_info,
        event_idx2entity_idx2field_idx,
        heuristic_type="GreedyDec",
    ):
        support_heuristic_types = ["GreedyDec", "ProductDec"]
        if heuristic_type not in support_heuristic_types:
            raise Exception(
                "Unsupported heuristic type {}, pleasure choose from {}".format(
                    heuristic_type, str(support_heuristic_types)
                )
            )

        span_context_list, doc_sent_context = self.get_doc_span_sent_context(
            doc_token_emb, doc_sent_emb, doc_fea, doc_span_info
        )

        span_token_tup_list = doc_span_info.span_token_tup_list
        span_mention_range_list = doc_span_info.span_mention_range_list
        mention_drange_list = doc_span_info.mention_drange_list
        mention_type_list = doc_span_info.mention_type_list
        # heuristic decoding strategies will work on these span candidates
        event_idx2field_idx2span_token_tup2dranges = (
            self.get_event_field_span_candidates(
                span_token_tup_list,
                span_mention_range_list,
                mention_drange_list,
                mention_type_list,
                event_idx2entity_idx2field_idx,
            )
        )

        # if there is no extracted span, just directly return
        if len(span_token_tup_list) == 0:
            event_pred_list = []
            event_idx2obj_idx2field_idx2token_tup = (
                []
            )  # this term will be compared with ground-truth table contents
            for event_idx in range(len(self.event_type_fields_pairs)):
                event_pred_list.append(0)
                event_idx2obj_idx2field_idx2token_tup.append(None)

            return (
                doc_fea.ex_idx,
                event_pred_list,
                event_idx2obj_idx2field_idx2token_tup,
                doc_span_info,
                event_idx2field_idx2span_token_tup2dranges,
            )

        # 1. get event type prediction as model-based approach
        event_pred_list = self.get_event_cls_info(
            doc_sent_context, doc_fea, train_flag=False
        )

        # 2. for each event type, get field prediction
        # From now on, use heuristic inference to get the token for the field
        # the following mappings are all implemented using list index
        event_idx2obj_idx2field_idx2token_tup = []
        for event_idx, event_pred in enumerate(event_pred_list):
            if event_pred == 0:
                event_idx2obj_idx2field_idx2token_tup.append(None)
                continue

            num_fields = self.event_tables[event_idx].num_fields
            field_idx2span_token_tup2dranges = (
                event_idx2field_idx2span_token_tup2dranges[event_idx]
            )

            obj_idx2field_idx2token_tup = [
                []
            ]  # at least one decode path will be appended
            for field_idx in range(num_fields):
                if heuristic_type == support_heuristic_types[0]:
                    obj_idx2field_idx2token_tup = append_top_span_only(
                        obj_idx2field_idx2token_tup,
                        field_idx,
                        field_idx2span_token_tup2dranges,
                    )
                elif heuristic_type == support_heuristic_types[1]:
                    obj_idx2field_idx2token_tup = append_all_spans(
                        obj_idx2field_idx2token_tup,
                        field_idx,
                        field_idx2span_token_tup2dranges,
                    )
                else:
                    raise Exception(
                        "Unsupported heuristic type {}, pleasure choose from {}".format(
                            heuristic_type, str(support_heuristic_types)
                        )
                    )

            event_idx2obj_idx2field_idx2token_tup.append(obj_idx2field_idx2token_tup)

        return (
            doc_fea.ex_idx,
            event_pred_list,
            event_idx2obj_idx2field_idx2token_tup,
            doc_span_info,
            event_idx2field_idx2span_token_tup2dranges,
        )

    def get_event_field_span_candidates(
        self,
        span_token_tup_list,
        span_mention_range_list,
        mention_drange_list,
        mention_type_list,
        event_idx2entity_idx2field_idx,
    ):
        # get mention idx -> span idx
        mention_span_idx_list = []
        for span_idx, (ment_idx_s, ment_idx_e) in enumerate(span_mention_range_list):
            mention_span_idx_list.extend([span_idx] * (ment_idx_e - ment_idx_s))
        assert len(mention_span_idx_list) == len(mention_drange_list)

        event_idx2field_idx2span_token_tup2dranges = {}
        for event_idx, (event_type, field_types, _) in enumerate(
            self.event_type_fields_pairs
        ):
            # get the predefined entity idx to field idx mapping
            gold_entity_idx2field_idx = event_idx2entity_idx2field_idx[event_idx]

            # store field candidates for this doc
            field_idx2span_token_tup2dranges = {}
            for field_idx, _ in enumerate(field_types):
                field_idx2span_token_tup2dranges[field_idx] = {}

            # aggregate field candidates according to mention types
            for ment_idx, (ment_drange, ment_entity_idx) in enumerate(
                zip(mention_drange_list, mention_type_list)
            ):
                if ment_entity_idx not in gold_entity_idx2field_idx:
                    continue
                ment_field_idx = gold_entity_idx2field_idx[ment_entity_idx]
                if ment_field_idx is None:
                    continue

                ment_span_idx = mention_span_idx_list[ment_idx]
                span_token_tup = span_token_tup_list[ment_span_idx]

                # because it is dict, so all modifications to the key will take effect in raw dict
                cur_span_token_tup2dranges = field_idx2span_token_tup2dranges[
                    ment_field_idx
                ]
                if span_token_tup not in cur_span_token_tup2dranges:
                    cur_span_token_tup2dranges[span_token_tup] = []
                cur_span_token_tup2dranges[span_token_tup].append(ment_drange)

            event_idx2field_idx2span_token_tup2dranges[
                event_idx
            ] = field_idx2span_token_tup2dranges

        return event_idx2field_idx2span_token_tup2dranges
