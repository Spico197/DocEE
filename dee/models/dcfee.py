from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dee.modules import AttentiveReducer, NERModel, get_doc_span_info_list


def get_one_key_sent_event(key_sent_idx, num_fields, field_idx2span_token_tup2dranges):
    field_idx2token_tup = []
    for field_idx in range(num_fields):
        token_tup2dranges = field_idx2span_token_tup2dranges[field_idx]

        # find the closest token_tup to the key sentence
        best_token_tup = None
        best_dist = 10000
        for token_tup, dranges in token_tup2dranges.items():
            for sent_idx, _, _ in dranges:
                cur_dist = abs(sent_idx - key_sent_idx)
                if cur_dist < best_dist:
                    best_token_tup = token_tup
                    best_dist = cur_dist

        field_idx2token_tup.append(best_token_tup)
    return field_idx2token_tup


def get_many_key_sent_event(key_sent_idx, num_fields, field_idx2span_token_tup2dranges):
    # get key_field_idx contained in key event sentence
    key_field_idx2token_tup_set = defaultdict(lambda: set())
    for field_idx, token_tup2dranges in field_idx2span_token_tup2dranges.items():
        assert field_idx < num_fields
        for token_tup, dranges in token_tup2dranges.items():
            for sent_idx, _, _ in dranges:
                if sent_idx == key_sent_idx:
                    key_field_idx2token_tup_set[field_idx].add(token_tup)

    field_idx2token_tup_list = []
    while len(key_field_idx2token_tup_set) > 0:
        # get key token tup candidates according to the distance in the sentence
        prev_field_idx = None
        prev_token_cand = None
        key_field_idx2token_cand = {}
        for key_field_idx, token_tup_set in key_field_idx2token_tup_set.items():
            assert len(token_tup_set) > 0

            if prev_token_cand is None:
                best_token_tup = token_tup_set.pop()
            else:
                prev_char_range = field_idx2span_token_tup2dranges[prev_field_idx][
                    prev_token_cand
                ][0][1:]
                best_dist = 10000
                best_token_tup = None
                for token_tup in token_tup_set:
                    cur_char_range = field_idx2span_token_tup2dranges[key_field_idx][
                        token_tup
                    ][0][1:]
                    cur_dist = min(
                        abs(cur_char_range[1] - prev_char_range[0]),
                        abs(cur_char_range[0] - prev_char_range[1]),
                    )
                    if cur_dist < best_dist:
                        best_dist = cur_dist
                        best_token_tup = token_tup
                token_tup_set.remove(best_token_tup)

            key_field_idx2token_cand[key_field_idx] = best_token_tup
            prev_field_idx = key_field_idx
            prev_token_cand = best_token_tup

        field_idx2token_tup = []
        for field_idx in range(num_fields):
            token_tup2dranges = field_idx2span_token_tup2dranges[field_idx]

            if field_idx in key_field_idx2token_tup_set:
                token_tup_set = key_field_idx2token_tup_set[field_idx]
                if len(token_tup_set) == 0:
                    del key_field_idx2token_tup_set[field_idx]
                token_tup = key_field_idx2token_cand[field_idx]
                field_idx2token_tup.append(token_tup)
            else:
                # find the closest token_tup to the key sentence
                best_token_tup = None
                best_dist = 10000
                for token_tup, dranges in token_tup2dranges.items():
                    for sent_idx, _, _ in dranges:
                        cur_dist = abs(sent_idx - key_sent_idx)
                        if cur_dist < best_dist:
                            best_token_tup = token_tup
                            best_dist = cur_dist

                field_idx2token_tup.append(best_token_tup)

        field_idx2token_tup_list.append(field_idx2token_tup)

    return field_idx2token_tup_list


class DCFEEModel(nn.Module):
    """
    This module implements the baseline model described in http://www.aclweb.org/anthology/P18-4009:
        "DCFEE: A Document-level Chinese Financial Event Extraction System
        based on Automatically Labeled Training Data"
    """

    def __init__(self, config, event_type_fields_pairs, ner_model=None):
        super(DCFEEModel, self).__init__()
        # Note that for distributed training, you must ensure that
        # for any batch, all parameters need to be used

        self.config = config
        self.event_type_fields_pairs = event_type_fields_pairs

        if ner_model is None:
            self.ner_model = NERModel(config)
        else:
            self.ner_model = ner_model

        # attentively reduce token embedding into sentence embedding
        self.doc_token_reducer = AttentiveReducer(
            config.hidden_size, dropout=config.dropout
        )
        # map sentence embedding to event prediction logits
        self.event_cls_layers = nn.ModuleList(
            [nn.Linear(config.hidden_size, 2) for _ in self.event_type_fields_pairs]
        )

    def get_batch_sent_emb(self, ner_token_emb, ner_token_masks, valid_sent_num_list):
        # From [ner_batch_size, sent_len, hidden_size] to [ner_batch_size, hidden_size]
        total_sent_emb = self.doc_token_reducer(ner_token_emb, ner_token_masks)
        total_sent_pos_ids = []
        for valid_sent_num in valid_sent_num_list:
            total_sent_pos_ids += list(range(valid_sent_num))

        return total_sent_emb

    def get_loss_on_doc(self, doc_sent_emb, doc_fea):
        doc_sent_label_mat = torch.tensor(
            doc_fea.doc_sent_labels,
            dtype=torch.long,
            device=doc_sent_emb.device,
            requires_grad=False,
        )
        event_cls_loss_list = []
        for event_idx, event_cls in enumerate(self.event_cls_layers):
            doc_sent_logits = event_cls(doc_sent_emb)  # [sent_num, 2]
            doc_sent_labels = doc_sent_label_mat[:, event_idx]  # [sent_num]
            event_cls_loss = F.cross_entropy(
                doc_sent_logits, doc_sent_labels, reduction="sum"
            )
            event_cls_loss_list.append(event_cls_loss)

        final_loss = sum(event_cls_loss_list)
        return final_loss

    def get_mix_loss(self, doc_sent_loss_list, doc_event_loss_list, doc_span_info_list):
        batch_size = len(doc_span_info_list)
        loss_batch_avg = 1.0 / batch_size
        lambda_1 = self.config.loss_lambda
        lambda_2 = 1 - lambda_1

        doc_ner_loss_list = []
        for doc_sent_loss, doc_span_info in zip(doc_sent_loss_list, doc_span_info_list):
            # doc_sent_loss: Size([num_valid_sents])
            sent_loss_scaling = doc_sent_loss.new_full(
                doc_sent_loss.size(), 1, requires_grad=False
            )
            sent_loss_scaling[
                doc_span_info.missed_sent_idx_list
            ] = self.config.loss_gamma
            doc_ner_loss = (doc_sent_loss * sent_loss_scaling).sum()
            doc_ner_loss_list.append(doc_ner_loss)

        return loss_batch_avg * (
            lambda_1 * sum(doc_ner_loss_list) + lambda_2 * sum(doc_event_loss_list)
        )

    def get_local_context_info(
        self, doc_batch_dict, train_flag=False, use_gold_span=False
    ):
        label_key = "doc_token_labels"
        if train_flag or use_gold_span:
            assert label_key in doc_batch_dict
            need_label_flag = True
        else:
            need_label_flag = False

        if need_label_flag:
            doc_token_labels_list = doc_batch_dict[label_key]
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

    def forward(
        self,
        doc_batch_dict,
        doc_features,
        use_gold_span=False,
        train_flag=True,
        heuristic_type="DCFEE-O",
        event_idx2entity_idx2field_idx=None,
        **kwargs
    ):
        # DCFEE does not need scheduled sampling
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
        doc_span_info_list = get_doc_span_info_list(
            doc_token_types_list, doc_fea_list, use_gold_span=use_gold_span
        )

        if train_flag:
            doc_event_loss_list = []
            for batch_idx, ex_idx in enumerate(ex_idx_list):
                doc_event_loss_list.append(
                    self.get_loss_on_doc(
                        doc_sent_emb_list[batch_idx],
                        doc_fea_list[batch_idx],
                    )
                )

            mix_loss = self.get_mix_loss(
                doc_sent_loss_list, doc_event_loss_list, doc_span_info_list
            )

            return mix_loss
        else:
            # return a list object may not be supported by torch.nn.parallel.DataParallel
            # ensure to run it under the single-gpu mode
            eval_results = []

            assert event_idx2entity_idx2field_idx is not None
            for batch_idx, ex_idx in enumerate(ex_idx_list):
                eval_results.append(
                    self.heuristic_decode_on_doc(
                        doc_sent_emb_list[batch_idx],
                        doc_fea_list[batch_idx],
                        doc_span_info_list[batch_idx],
                        event_idx2entity_idx2field_idx,
                        heuristic_type=heuristic_type,
                    )
                )

            return eval_results

    def heuristic_decode_on_doc(
        self,
        doc_sent_emb,
        doc_fea,
        doc_span_info,
        event_idx2entity_idx2field_idx,
        heuristic_type="DCFEE-O",
    ):
        # DCFEE-O: just produce One event per triggered sentence
        # DCFEE-M: produce Multiple potential events per triggered sentence
        support_heuristic_types = ["DCFEE-O", "DCFEE-M"]
        if heuristic_type not in support_heuristic_types:
            raise Exception(
                "Unsupported heuristic type {}, pleasure choose from {}".format(
                    heuristic_type, str(support_heuristic_types)
                )
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

        event_idx2key_sent_idx_list = []
        event_pred_list = []
        event_idx2obj_idx2field_idx2token_tup = []
        for event_idx, event_cls in enumerate(self.event_cls_layers):
            event_type, field_types, _, _ = self.event_type_fields_pairs[event_idx]
            num_fields = len(field_types)
            field_idx2span_token_tup2dranges = (
                event_idx2field_idx2span_token_tup2dranges[event_idx]
            )

            # get key event sentence prediction
            doc_sent_logits = event_cls(doc_sent_emb)  # [sent_num, 2]
            doc_sent_logp = F.log_softmax(doc_sent_logits, dim=-1)  # [sent_num, 2]
            doc_sent_pred_list = doc_sent_logp.argmax(dim=-1).tolist()
            key_sent_idx_list = [
                sent_idx
                for sent_idx, sent_pred in enumerate(doc_sent_pred_list)
                if sent_pred == 1
            ]
            event_idx2key_sent_idx_list.append(key_sent_idx_list)

            if len(key_sent_idx_list) == 0:
                event_pred_list.append(0)
                event_idx2obj_idx2field_idx2token_tup.append(None)
            else:
                obj_idx2field_idx2token_tup = []
                for key_sent_idx in key_sent_idx_list:
                    if heuristic_type == support_heuristic_types[0]:
                        field_idx2token_tup = get_one_key_sent_event(
                            key_sent_idx, num_fields, field_idx2span_token_tup2dranges
                        )
                        obj_idx2field_idx2token_tup.append(field_idx2token_tup)
                    elif heuristic_type == support_heuristic_types[1]:
                        field_idx2token_tup_list = get_many_key_sent_event(
                            key_sent_idx, num_fields, field_idx2span_token_tup2dranges
                        )
                        obj_idx2field_idx2token_tup.extend(field_idx2token_tup_list)
                    else:
                        raise Exception(
                            "Unsupported heuristic type {}, pleasure choose from {}".format(
                                heuristic_type, str(support_heuristic_types)
                            )
                        )
                event_pred_list.append(1)
                event_idx2obj_idx2field_idx2token_tup.append(
                    obj_idx2field_idx2token_tup
                )

        return (
            doc_fea.ex_idx,
            event_pred_list,
            event_idx2obj_idx2field_idx2token_tup,
            doc_span_info,
            event_idx2field_idx2span_token_tup2dranges,
            event_idx2key_sent_idx_list,
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
        for event_idx, (event_type, field_types, _, _) in enumerate(
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
