import copy
import math
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from dee.models.lstmmtl2complete_graph import LSTMMTL2CompleteGraphModel
from dee.modules import (
    GAT,
    MLP,
    EventTableForSigmoidMultiArgRel,
    MentionTypeConcatEncoder,
    MentionTypeEncoder,
    SentencePosEncoder,
    directed_trigger_graph_decode,
    directed_trigger_graph_incremental_decode,
    get_doc_arg_rel_info_list,
    mlp,
    normalize_adj,
    transformer,
)
from dee.utils import assign_role_from_gold_to_comb, closest_match


class TriggerAwarePrunedCompleteGraph(LSTMMTL2CompleteGraphModel):
    def __init__(self, config, event_type_fields_pairs, ner_model):
        super().__init__(config, event_type_fields_pairs, ner_model=ner_model)

        if self.config.use_token_role:
            if config.ment_feature_type == "concat":
                self.ment_type_encoder = MentionTypeConcatEncoder(
                    config.ment_type_hidden_size,
                    len(config.ent_type2id),
                    dropout=config.dropout,
                )
                self.hidden_size = config.hidden_size + config.ment_type_hidden_size
            else:
                self.ment_type_encoder = MentionTypeEncoder(
                    config.hidden_size, config.num_entity_labels, dropout=config.dropout
                )
                self.hidden_size = config.hidden_size
        else:
            self.hidden_size = config.hidden_size

        self.start_lstm = (
            self.end_lstm
        ) = self.start_mlp = self.end_mlp = self.biaffine = None

        if self.config.use_span_lstm:
            self.span_lstm = nn.LSTM(
                self.hidden_size,
                self.hidden_size // 2,
                num_layers=self.config.span_lstm_num_layer,
                bias=True,
                batch_first=True,
                dropout=self.config.dropout,
                bidirectional=True,
            )

        if self.config.mlp_before_adj_measure:
            self.q_w = MLP(
                self.hidden_size, self.hidden_size, dropout=self.config.dropout
            )
            self.k_w = MLP(
                self.hidden_size, self.hidden_size, dropout=self.config.dropout
            )
        else:
            self.q_w = nn.Linear(self.hidden_size, self.hidden_size)
            self.k_w = nn.Linear(self.hidden_size, self.hidden_size)

        if self.config.use_mention_lstm:
            self.mention_lstm = nn.LSTM(
                self.hidden_size,
                self.hidden_size // 2,
                num_layers=self.config.num_mention_lstm_layer,
                bias=True,
                batch_first=True,
                dropout=self.config.dropout,
                bidirectional=True,
            )

        # self.span_att = transformer.SelfAttention(
        #     self.hidden_size,
        #     dropout=self.config.dropout
        # )

        self.event_tables = nn.ModuleList(
            [
                EventTableForSigmoidMultiArgRel(
                    event_type,
                    field_types,
                    self.config.hidden_size,
                    self.hidden_size,
                    min_field_num,
                    use_field_cls_mlp=self.config.use_field_cls_mlp,
                    dropout=self.config.dropout,
                )
                for event_type, field_types, _, min_field_num in self.event_type_fields_pairs
            ]
        )

    # def pred_adj_mat_reorgnise(self, pred_adj_mat):
    #     """
    #     fill the diag to 1 and make sure the adj_mat is symmetric
    #     """
    #     adj_mat = pred_adj_mat
    #     if not self.config.directed_trigger_graph:
    #         adj_mat = torch.bitwise_and(adj_mat, adj_mat.T)
    #     adj_mat.fill_diagonal_(0)
    #     return adj_mat

    def get_arg_role_loss(self, arg_role_logits, role_types):
        rt_multihot = torch.zeros_like(arg_role_logits, requires_grad=False)
        for ent_idx, roles in enumerate(role_types):
            if roles is None:
                continue
            for role in roles:
                rt_multihot[ent_idx, role] = 1
        # role_loss = F.binary_cross_entropy(arg_role_logits.reshape(-1), rt_multihot.reshape(-1), reduction='sum')
        role_loss = F.binary_cross_entropy(
            arg_role_logits.reshape(-1), rt_multihot.reshape(-1)
        )
        return role_loss

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
            ent_fix_mode=self.config.ent_fix_mode,
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
                        use_gold_adj_mat=use_gold_span,
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
                    # self.get_gold_results_on_doc(
                    self.get_eval_on_doc(
                        doc_token_emb_list[batch_idx],
                        doc_sent_emb_list[batch_idx],
                        doc_fea_list[batch_idx],
                        doc_arg_rel_info_list[batch_idx],
                    )
                )
            return eval_results

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

            if self.config.use_token_role:
                # get mention type embedding
                if self.config.ment_feature_type == "concat":
                    yy = [
                        self.config.tag_id2tag_name[x]
                        for x in doc_arg_rel_info.mention_type_list
                    ]
                    # there will be 'O' labels for mentions if `OtherType` is not included in the ent list
                    zz = [
                        self.config.ent_type2id[xx[2:] if len(xx) > 2 else xx]
                        for xx in yy
                    ]
                    doc_mention_emb = self.ment_type_encoder(doc_mention_emb, zz)
                else:
                    doc_mention_emb = self.ment_type_encoder(
                        doc_mention_emb, doc_arg_rel_info.mention_type_list
                    )

        return doc_mention_emb

    def get_doc_span_sent_context(
        self, doc_token_emb, doc_sent_emb, doc_fea, doc_arg_rel_info
    ):
        """
        get all the span representations by aggregating mention representations,
        and sentence representations
        """
        doc_mention_emb = self.get_doc_span_mention_emb(doc_token_emb, doc_arg_rel_info)

        if self.config.use_mention_lstm:
            # mention further encoding
            doc_mention_emb = self.mention_lstm(doc_mention_emb.unsqueeze(0))[
                0
            ].squeeze(0)

        # only consider actual sentences
        if doc_sent_emb.size(0) > doc_fea.valid_sent_num:
            doc_sent_emb = doc_sent_emb[: doc_fea.valid_sent_num, :]

        span_context_list = []

        if doc_mention_emb is None:
            doc_sent_context = doc_sent_emb
        else:
            num_mentions = doc_mention_emb.size(0)

            # collect span context
            for mid_s, mid_e in doc_arg_rel_info.span_mention_range_list:
                assert mid_e <= num_mentions
                multi_ment_emb = doc_mention_emb[
                    mid_s:mid_e
                ]  # [num_mentions, hidden_size]

                if self.config.span_mention_sum:
                    span_context = multi_ment_emb.sum(0, keepdim=True)
                else:
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

    def get_arg_combination_loss(
        self, scores, doc_arg_rel_info, event_idx=None, margin=0.1
    ):
        # rel_adj_mat = doc_arg_rel_info.whole_arg_rel_mat.reveal_adj_mat(masked_diagonal=1, tolist=False).to(scores.device).float()
        if self.config.self_loop:
            rel_adj_mat = (
                doc_arg_rel_info.whole_arg_rel_mat.reveal_adj_mat(
                    masked_diagonal=None, tolist=False
                )
                .to(scores.device)
                .float()
            )
        else:
            rel_adj_mat = (
                doc_arg_rel_info.whole_arg_rel_mat.reveal_adj_mat(
                    masked_diagonal=1, tolist=False
                )
                .to(scores.device)
                .float()
            )

        combination_loss = F.binary_cross_entropy_with_logits(scores, rel_adj_mat)
        # combination_loss = F.binary_cross_entropy(torch.clamp(scores, min=1e-6, max=1.0), rel_adj_mat)
        # combination_loss = F.mse_loss(torch.sigmoid(scores), rel_adj_mat, reduction='sum')
        # combination_loss = F.mse_loss(scores, rel_adj_mat)
        # combination_loss = F.mse_loss(torch.sigmoid(scores.view(-1)), rel_adj_mat.view(-1))
        # combination_loss = F.mse_loss(torch.sigmoid(scores), rel_adj_mat)
        # combination_loss = F.mse_loss(torch.sigmoid(torch.clamp(scores, min=-5.0, max=5.0)), rel_adj_mat)
        # combination_loss = F.mse_loss(scores, rel_adj_mat.masked_fill(rel_adj_mat == 0, -1))
        # combination_loss = F.mse_loss(torch.tanh(scores), rel_adj_mat.masked_fill(rel_adj_mat == 0, -1))

        # # Su Jianlin's multilabel CE
        # # reference: https://spaces.ac.cn/archives/7359/comment-page-2
        # scores = scores.view(-1)
        # rel_adj_mat = rel_adj_mat.view(-1)
        # scores = (1 - 2 * rel_adj_mat) * scores
        # pred_neg = scores - rel_adj_mat * 1e12
        # pred_pos = scores - (1 - rel_adj_mat) * 1e12
        # zeros = torch.zeros_like(scores)
        # pred_neg = torch.stack([pred_neg, zeros], dim=-1)
        # pred_pos = torch.stack([pred_pos, zeros], dim=-1)
        # neg_loss = torch.logsumexp(pred_neg, dim=-1)
        # pos_loss = torch.logsumexp(pred_pos, dim=-1)
        # combination_loss = neg_loss + pos_loss
        # combination_loss = combination_loss.mean()

        # contrastive learning
        # cl = F.log_softmax(scores / 0.05, dim=-1)
        # cl = cl * rel_adj_mat.masked_fill(rel_adj_mat == 0, -1)
        # c_score = F.mse_loss(scores, rel_adj_mat.masked_fill(rel_adj_mat == 0, -1), reduction='none')
        # contrastive_loss = -F.log_softmax(c_score / 0.05, dim=-1).mean()
        # return max(-cl.mean(), 0.0)

        # # cos emb loss
        # target = rel_adj_mat.masked_fill(rel_adj_mat == 0, -1)
        # pos = (1.0 - scores).masked_fill(target == -1, 0.0).sum() / torch.sum(target == 1.0)
        # neg_scores = scores.masked_fill(target == 1, margin) - margin
        # neg = neg_scores.masked_fill(neg_scores < 0.0, 0.0).sum() / torch.sum(target == -1.0)
        # combination_loss = pos + neg
        return combination_loss

    def get_adj_mat_logits(self, hidden):
        # dot scaled similarity
        query = self.q_w(hidden)
        key = self.k_w(hidden)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # # cos similarity
        # num_ents = hidden.shape[0]
        # query = self.q_w(hidden).unsqueeze(1).repeat((1, num_ents, 1))
        # key = self.k_w(hidden).unsqueeze(0).repeat((num_ents, 1, 1))
        # scores = F.cosine_similarity(query, key, dim=-1)
        return scores

    def get_loss_on_doc(
        self,
        doc_token_emb,
        doc_sent_emb,
        doc_fea,
        doc_arg_rel_info,
        use_gold_adj_mat=False,
    ):
        if self.config.stop_gradient:
            doc_token_emb = doc_token_emb.detach()
            doc_sent_emb = doc_sent_emb.detach()
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
        lstm_batch_span_context = None
        if self.config.use_span_lstm:
            # there's no padding in spans, no need to pack rnn sequence
            lstm_batch_span_context = batch_span_context.unsqueeze(0)
            lstm_batch_span_context, (_, _) = self.span_lstm(lstm_batch_span_context)
            lstm_batch_span_context = lstm_batch_span_context.squeeze(0)

        if lstm_batch_span_context is not None:
            batch_span_context = lstm_batch_span_context

        # if self.config.use_span_att:
        # batch_span_context = batch_span_context.unsqueeze(0)
        # batch_span_context, p_attn, scores = self.span_att(batch_span_context, return_scores=True)
        # batch_span_context = batch_span_context.squeeze(1)

        scores = self.get_adj_mat_logits(batch_span_context)

        # for each event type, get argument combination loss
        # argument combination loss, calculated by comparing
        # the biaffine output and the gold event SpanArgRelAdjMat
        arg_combination_loss = []
        arg_role_loss = []

        # event-relevant combination, attention between event representation and batch_span_context output
        if self.config.event_relevant_combination:
            raise RuntimeError("event_relevant_combination is not supported yet")

        # combination loss via biaffine
        # biaffine_out = self.get_adj_mat_logits(batch_span_context)
        assert scores.shape[-1] == doc_arg_rel_info.whole_arg_rel_mat.len_spans
        comb_loss = self.get_arg_combination_loss(
            scores, doc_arg_rel_info, event_idx=None
        )
        arg_combination_loss.append(comb_loss)

        if use_gold_adj_mat:
            pred_adj_mat = doc_fea.whole_arg_rel_mat.reveal_adj_mat()
            event_pred_list = doc_fea.event_type_labels
        else:
            pred_adj_mat = (
                torch.sigmoid(scores).ge(self.config.biaffine_hard_threshold).long()
            )
            # pred_adj_mat = self.pred_adj_mat_reorgnise(pred_adj_mat)
            pred_adj_mat = pred_adj_mat.detach().cpu().tolist()
            event_pred_list = self.get_event_cls_info(
                doc_sent_context, doc_fea, train_flag=False
            )

        if self.config.guessing_decode:
            num_triggers = 0
        else:
            num_triggers = self.config.eval_num_triggers

        if self.config.incremental_min_conn > -1:
            combs = directed_trigger_graph_incremental_decode(
                pred_adj_mat, num_triggers, self.config.incremental_min_conn
            )
        else:
            # combs = directed_trigger_graph_decode(pred_adj_mat, num_triggers, self.config.max_clique_decode, self.config.with_left_trigger, self.config.with_all_one_trigger_comb)
            combs = directed_trigger_graph_decode(
                pred_adj_mat,
                num_triggers,
                self_loop=self.config.self_loop,
                max_clique=self.config.max_clique_decode,
                with_left_trigger=self.config.with_left_trigger,
            )

        if self.config.at_least_one_comb:
            if len(combs) < 1:
                combs = [set(range(len(pred_adj_mat)))]

        event_cls_loss = self.get_event_cls_info(
            doc_sent_context, doc_fea, train_flag=True
        )
        for event_idx, event_label in enumerate(event_pred_list):
            if not event_label:
                continue
            events = doc_arg_rel_info.pred_event_arg_idxs_objs_list[event_idx]
            if events is None:
                continue
            gold_combinations = events
            for comb in combs:
                event_table = self.event_tables[event_idx]
                gold_comb, _ = closest_match(comb, gold_combinations)
                instance = assign_role_from_gold_to_comb(comb, gold_comb)
                span_idxs = []
                role_types = []
                span_rep_list_for_event_instance = []
                for span_idx, role_type in instance:
                    span_idxs.append(span_idx)
                    role_types.append(role_type)
                    if self.config.role_by_encoding:
                        span_rep_list_for_event_instance.append(
                            batch_span_context[span_idx]
                        )
                    else:
                        span_rep_list_for_event_instance.append(
                            span_context_list[span_idx].squeeze(0)
                        )
                span_rep_for_event_instance = torch.stack(
                    span_rep_list_for_event_instance, dim=0
                )
                role_cls_logits = event_table(
                    batch_span_emb=span_rep_for_event_instance
                )
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
        lstm_batch_span_context = None
        if self.config.use_span_lstm:
            lstm_batch_span_context = batch_span_context.unsqueeze(0)
            lstm_batch_span_context, (_, _) = self.span_lstm(lstm_batch_span_context)
            lstm_batch_span_context = lstm_batch_span_context.squeeze(0)

        if lstm_batch_span_context is not None:
            batch_span_context = lstm_batch_span_context

        # if self.config.use_span_att:
        # batch_span_context = batch_span_context.unsqueeze(0)
        # batch_span_context, p_attn, scores = self.span_att(batch_span_context, return_scores=True)
        # batch_span_context = batch_span_context.squeeze(1)

        scores = self.get_adj_mat_logits(batch_span_context)

        if (
            self.config.event_relevant_combination
        ):  # event-relevant combination, attention between event representation and batch_span_context output
            raise RuntimeError("event_relevant_combination is not supported yet")

        pred_adj_mat = (
            torch.sigmoid(scores).ge(self.config.biaffine_hard_threshold).long()
        )
        # pred_adj_mat = self.pred_adj_mat_reorgnise(torch.sigmoid(scores).ge(self.config.biaffine_hard_threshold).long())
        assert pred_adj_mat.shape[-1] == doc_arg_rel_info.whole_arg_rel_mat.len_spans
        # debug mode statement only for time saving
        if self.config.run_mode == "debug":
            pred_adj_mat = pred_adj_mat[:10, :10]

        """only for 100% filled graph testing"""
        # pred_adj_mat = torch.ones((batch_span_context.shape[0], batch_span_context.shape[0]))
        """end of testing"""

        pred_adj_mat = pred_adj_mat.detach().cpu().tolist()
        final_pred_adj_mat.append(pred_adj_mat)
        if self.config.guessing_decode:
            num_triggers = 0
        else:
            num_triggers = self.config.eval_num_triggers

        if self.config.incremental_min_conn > -1:
            raw_combinations = directed_trigger_graph_incremental_decode(
                pred_adj_mat, num_triggers, self.config.incremental_min_conn
            )
        else:
            # raw_combinations = directed_trigger_graph_decode(pred_adj_mat, num_triggers, self.config.max_clique_decode, self.config.with_left_trigger, self.config.with_all_one_trigger_comb)
            raw_combinations = directed_trigger_graph_decode(
                pred_adj_mat,
                num_triggers,
                self_loop=self.config.self_loop,
                max_clique=self.config.max_clique_decode,
                with_left_trigger=self.config.with_left_trigger,
            )

        if self.config.at_least_one_comb:
            if len(raw_combinations) < 1:
                raw_combinations = [set(range(len(pred_adj_mat)))]

        event_idx2obj_idx2field_idx2token_tup = []
        for event_idx, event_pred in enumerate(event_pred_list):
            if event_pred == 0:
                event_idx2obj_idx2field_idx2token_tup.append(None)
                continue
            event_table = self.event_tables[event_idx]
            # TODO(tzhu): m2m support from all the combinations
            """combinations filtering based on minimised number of argument"""
            # combinations = list(filter(lambda x: len(x) >= event_table.min_field_num, raw_combinations))
            """end of combination filtering"""
            combinations = copy.deepcopy(raw_combinations)
            event_idx2combinations.append(combinations)
            if len(combinations) <= 0:
                event_idx2obj_idx2field_idx2token_tup.append(None)
                continue
            obj_idx2field_idx2token_tup = []
            for combination in combinations:
                span_rep_list_for_event_instance = []
                for span_idx in combination:
                    if self.config.role_by_encoding:
                        span_rep_list_for_event_instance.append(
                            batch_span_context[span_idx]
                        )
                    else:
                        span_rep_list_for_event_instance.append(
                            span_context_list[span_idx].squeeze(0)
                        )
                span_rep_for_event_instance = torch.stack(
                    span_rep_list_for_event_instance, dim=0
                )
                role_preds = event_table.predict_span_role(span_rep_for_event_instance)
                """roles random generation (only for debugging)"""
                # role_preds = [random.randint(0, event_table.num_fields - 1) for _ in range(len(combination))]
                """end of random roles generation"""
                event_arg_obj = self.reveal_event_arg_obj(
                    combination, role_preds, event_table.num_fields
                )
                field_idx2token_tup = self.convert_span_idx_to_token_tup(
                    event_arg_obj, doc_arg_rel_info
                )
                obj_idx2field_idx2token_tup.append(field_idx2token_tup)
            # obj_idx2field_idx2token_tup = merge_non_conflicting_ins_objs(obj_idx2field_idx2token_tup)
            event_idx2obj_idx2field_idx2token_tup.append(obj_idx2field_idx2token_tup)
        # the first three terms are for metric calculation, the last three are for case studies
        return (
            doc_fea.ex_idx,
            event_pred_list,
            event_idx2obj_idx2field_idx2token_tup,
            doc_arg_rel_info,
            final_pred_adj_mat,
            event_idx2combinations,
        )
