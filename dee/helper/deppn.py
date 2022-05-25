from collections import Counter

import torch
from tqdm import tqdm

from dee.utils import logger

from .dee import DEEExample
from .ner import NERExample, NERFeatureConverter


class DEPPNFeature(object):
    def __init__(
        self,
        guid,
        ex_idx,
        doc_type,
        doc_token_id_mat,
        doc_token_mask_mat,
        doc_token_label_mat,
        span_token_ids_list,
        span_dranges_list,
        event_type_labels,
        event_arg_idxs_objs_list,
        valid_sent_num=None,
    ):
        self.guid = guid
        self.ex_idx = ex_idx  # example row index, used for backtracking
        self.bak_ex_idx = ex_idx
        self.doc_type = doc_type
        self.valid_sent_num = valid_sent_num

        # directly set tensor for dee feature to save memory
        # self.doc_token_id_mat = doc_token_id_mat
        # self.doc_token_mask_mat = doc_token_mask_mat
        # self.doc_token_label_mat = doc_token_label_mat
        self.doc_token_ids = torch.tensor(doc_token_id_mat, dtype=torch.long)
        self.doc_token_masks = torch.tensor(
            doc_token_mask_mat, dtype=torch.uint8
        )  # uint8 for mask
        self.doc_token_labels = torch.tensor(doc_token_label_mat, dtype=torch.long)

        # sorted by the first drange tuple
        # [(token_id, ...), ...]
        # span_idx -> span_token_id tuple
        self.span_token_ids_list = span_token_ids_list
        # [[(sent_idx, char_s, char_e), ...], ...]
        # span_idx -> [drange tuple, ...]
        self.span_dranges_list = span_dranges_list

        # [event_type_label, ...]
        # length = the total number of events to be considered
        # event_type_label \in {0, 1}, 0: no 1: yes
        self.event_type_labels = event_type_labels
        # event_type is denoted by the index of event_type_labels
        # event_type_idx -> event_obj_idx -> event_arg_idx -> span_idx
        # if no event objects, event_type_idx -> None
        self.event_arg_idxs_objs_list = event_arg_idxs_objs_list

        # event_type_idx -> event_field_idx -> pre_path -> {span_idx, ...}
        # pre_path is tuple of span_idx
        self.event_idx2field_idx2pre_path2cur_span_idx_set = self.build_dag_info(
            self.event_arg_idxs_objs_list
        )

        # event_type_idx -> key_sent_idx_set, used for key-event sentence detection
        (
            self.event_idx2key_sent_idx_set,
            self.doc_sent_labels,
        ) = self.build_key_event_sent_info()

    def generate_dag_info_for(self, pred_span_token_tup_list, return_miss=False):
        """
        :param pred_span_token_tup_list:  entity span token id (pred or gold)
        """
        num_pred_span = len(pred_span_token_tup_list)
        token_tup2pred_span_idx = {
            token_tup: pred_span_idx
            for pred_span_idx, token_tup in enumerate(pred_span_token_tup_list)
        }
        gold_span_idx2pred_span_idx = {}
        missed_span_idx_list = []  # in terms of self
        missed_sent_idx_list = []  # in terms of self
        for gold_span_idx, token_tup in enumerate(self.span_token_ids_list):
            if token_tup in token_tup2pred_span_idx:
                pred_span_idx = token_tup2pred_span_idx[token_tup]
                gold_span_idx2pred_span_idx[gold_span_idx] = pred_span_idx
            else:
                missed_span_idx_list.append(gold_span_idx)
                for gold_drange in self.span_dranges_list[gold_span_idx]:
                    missed_sent_idx_list.append(gold_drange[0])
        pred_event_arg_idxs_objs_lists = []
        pred_event_type_idxs_lists = []
        # one_event_type = False
        for i, (event_arg_idxs_objs, event_type_idxs) in enumerate(
            zip(self.event_arg_idxs_objs_list, self.event_type_labels)
        ):
            if event_arg_idxs_objs is None:
                pred_event_arg_idxs_objs_lists.append(event_arg_idxs_objs)
                # pred_event_type_idxs_lists.append(list((event_type_idxs,)))
                pred_event_type_idxs_lists.append(event_type_idxs)
            else:
                pred_event_arg_idxs_objs_list = []
                pred_event_type_idxs_list = []
                for event_arg_idxs in event_arg_idxs_objs:
                    pred_event_arg_idxs = []
                    for gold_span_idx in event_arg_idxs:
                        if gold_span_idx in gold_span_idx2pred_span_idx:
                            pred_event_arg_idxs.append(
                                gold_span_idx2pred_span_idx[gold_span_idx]
                            )
                        else:
                            pred_event_arg_idxs.append(num_pred_span)
                    pred_event_type_idxs_list.append(i)
                    pred_event_arg_idxs_objs_list.append(tuple(pred_event_arg_idxs))
                pred_event_type_idxs_lists.append(pred_event_type_idxs_list)
                pred_event_arg_idxs_objs_lists.append(pred_event_arg_idxs_objs_list)
        # pred_event_type_idxs_list = pred_event_type_idxs_list[:self.num_generated_triplets]
        # pred_event_arg_idxs_objs_list = pred_event_arg_idxs_objs_list[:self.num_generated_triplets]
        return (
            gold_span_idx2pred_span_idx,
            pred_event_arg_idxs_objs_lists,
            pred_event_type_idxs_lists,
        )

    def get_event_args_objs_list(self):
        event_args_objs_list = []
        for event_arg_idxs_objs in self.event_arg_idxs_objs_list:
            if event_arg_idxs_objs is None:
                event_args_objs_list.append(None)
            else:
                event_args_objs = []
                for event_arg_idxs in event_arg_idxs_objs:
                    event_args = []
                    for arg_idx in event_arg_idxs:
                        if arg_idx is None:
                            token_tup = None
                        else:
                            token_tup = self.span_token_ids_list[arg_idx]
                        event_args.append(token_tup)
                    event_args_objs.append(event_args)
                event_args_objs_list.append(event_args_objs)

        return event_args_objs_list

    def build_key_event_sent_info(self):
        assert len(self.event_type_labels) == len(self.event_arg_idxs_objs_list)
        # event_idx -> key_event_sent_index_set
        event_idx2key_sent_idx_set = [set() for _ in self.event_type_labels]
        for key_sent_idx_set, event_label, event_arg_idxs_objs in zip(
            event_idx2key_sent_idx_set,
            self.event_type_labels,
            self.event_arg_idxs_objs_list,
        ):
            if event_label == 0:
                assert event_arg_idxs_objs is None
            else:
                for event_arg_idxs_obj in event_arg_idxs_objs:
                    sent_idx_cands = []
                    for span_idx in event_arg_idxs_obj:
                        if span_idx is None:
                            continue
                        span_dranges = self.span_dranges_list[span_idx]
                        for sent_idx, _, _ in span_dranges:
                            sent_idx_cands.append(sent_idx)
                    if len(sent_idx_cands) == 0:
                        raise Exception(
                            "Event {} has no valid spans".format(
                                str(event_arg_idxs_obj)
                            )
                        )
                    sent_idx_cnter = Counter(sent_idx_cands)
                    key_sent_idx = sent_idx_cnter.most_common()[0][0]
                    key_sent_idx_set.add(key_sent_idx)

        doc_sent_labels = []  # 1: key event sentence, 0: otherwise
        for sent_idx in range(
            self.valid_sent_num
        ):  # masked sents will be truncated at the model part
            sent_labels = []
            for (
                key_sent_idx_set
            ) in event_idx2key_sent_idx_set:  # this mapping is a list
                if sent_idx in key_sent_idx_set:
                    sent_labels.append(1)
                else:
                    sent_labels.append(0)
            doc_sent_labels.append(sent_labels)

        return event_idx2key_sent_idx_set, doc_sent_labels

    @staticmethod
    def build_dag_info(event_arg_idxs_objs_list):
        # event_idx -> field_idx -> pre_path -> {span_idx, ...}
        # pre_path is tuple of span_idx
        event_idx2field_idx2pre_path2cur_span_idx_set = []
        for event_idx, event_arg_idxs_list in enumerate(event_arg_idxs_objs_list):
            if event_arg_idxs_list is None:
                event_idx2field_idx2pre_path2cur_span_idx_set.append(None)
            else:
                num_fields = len(event_arg_idxs_list[0])
                # field_idx -> pre_path -> {span_idx, ...}
                field_idx2pre_path2cur_span_idx_set = []
                for field_idx in range(num_fields):
                    pre_path2cur_span_idx_set = {}
                    for event_arg_idxs in event_arg_idxs_list:
                        pre_path = event_arg_idxs[:field_idx]
                        span_idx = event_arg_idxs[field_idx]
                        if pre_path not in pre_path2cur_span_idx_set:
                            pre_path2cur_span_idx_set[pre_path] = set()
                        pre_path2cur_span_idx_set[pre_path].add(span_idx)
                    field_idx2pre_path2cur_span_idx_set.append(
                        pre_path2cur_span_idx_set
                    )
                event_idx2field_idx2pre_path2cur_span_idx_set.append(
                    field_idx2pre_path2cur_span_idx_set
                )

        return event_idx2field_idx2pre_path2cur_span_idx_set

    def is_multi_event(self):
        event_cnt = 0
        for event_objs in self.event_arg_idxs_objs_list:
            if event_objs is not None:
                event_cnt += len(event_objs)
                if event_cnt > 1:
                    return True

        return False


class DEPPNFeatureConverter(object):
    def __init__(
        self,
        entity_label_list,
        template,
        max_sent_len,
        max_sent_num,
        tokenizer,
        ner_fea_converter=None,
        include_cls=True,
        include_sep=True,
    ):
        self.entity_label_list = entity_label_list
        self.template = template
        self.event_type_fields_pairs = template.event_type_fields_list
        self.max_sent_len = max_sent_len
        self.max_sent_num = max_sent_num
        self.tokenizer = tokenizer
        self.truncate_doc_count = (
            0  # track how many docs have been truncated due to max_sent_num
        )
        self.truncate_span_count = 0  # track how may spans have been truncated

        # label not in entity_label_list will be default 'O'
        # sent_len > max_sent_len will be truncated, and increase ner_fea_converter.truncate_freq
        if ner_fea_converter is None:
            self.ner_fea_converter = NERFeatureConverter(
                entity_label_list,
                self.max_sent_len,
                tokenizer,
                include_cls=include_cls,
                include_sep=include_sep,
            )
        else:
            self.ner_fea_converter = ner_fea_converter

        self.include_cls = include_cls
        self.include_sep = include_sep

        # prepare entity_label -> entity_index mapping
        self.entity_label2index = {}
        for entity_idx, entity_label in enumerate(self.entity_label_list):
            self.entity_label2index[entity_label] = entity_idx

        # prepare event_type -> event_index and event_index -> event_fields mapping
        self.event_type2index = {}
        self.event_type_list = []
        self.event_fields_list = []
        self.event_type2num = {}
        for event_idx, (event_type, event_fields, *_) in enumerate(
            self.event_type_fields_pairs
        ):
            self.event_type2index[event_type] = event_idx
            self.event_type_list.append(event_type)
            self.event_fields_list.append(event_fields)
            self.event_type2num[event_type] = 0

    def convert_example_to_feature(self, ex_idx, dee_example, log_flag=False):
        annguid = dee_example.guid
        assert isinstance(dee_example, DEEExample)

        # 1. prepare doc token-level feature

        # Size(num_sent_num, num_sent_len)
        doc_token_id_mat = []  # [[token_idx, ...], ...]
        doc_token_mask_mat = []  # [[token_mask, ...], ...]
        doc_token_label_mat = []  # [[token_label_id, ...], ...]

        for sent_idx, sent_text in enumerate(dee_example.sentences):
            if sent_idx >= self.max_sent_num:
                # truncate doc whose number of sentences is longer than self.max_sent_num
                self.truncate_doc_count += 1
                break

            if sent_idx in dee_example.sent_idx2srange_mspan_mtype_tuples:
                srange_mspan_mtype_tuples = (
                    dee_example.sent_idx2srange_mspan_mtype_tuples[sent_idx]
                )
            else:
                srange_mspan_mtype_tuples = []

            ner_example = NERExample(
                "{}-{}".format(annguid, sent_idx),
                sent_text,
                self.tokenizer.dee_tokenize(sent_text),
                srange_mspan_mtype_tuples,
            )
            # sentence truncated count will be recorded incrementally
            ner_feature = self.ner_fea_converter.convert_example_to_feature(
                ner_example, log_flag=log_flag
            )

            doc_token_id_mat.append(ner_feature.input_ids)
            doc_token_mask_mat.append(ner_feature.input_masks)
            doc_token_label_mat.append(ner_feature.label_ids)

        assert (
            len(doc_token_id_mat)
            == len(doc_token_mask_mat)
            == len(doc_token_label_mat)
            <= self.max_sent_num
        )
        valid_sent_num = len(doc_token_id_mat)

        # 2. prepare span feature
        # spans are sorted by the first drange
        span_token_ids_list = []
        span_dranges_list = []
        mspan2span_idx = {}
        for mspan in dee_example.ann_valid_mspans:
            if mspan in mspan2span_idx:
                continue

            raw_dranges = dee_example.ann_mspan2dranges[mspan]
            char_base_s = 1 if self.include_cls else 0
            char_max_end = (
                self.max_sent_len - 1 if self.include_sep else self.max_sent_len
            )
            span_dranges = []
            for sent_idx, char_s, char_e in raw_dranges:
                if (
                    char_base_s + char_e <= char_max_end
                    and sent_idx < self.max_sent_num
                ):
                    span_dranges.append(
                        (sent_idx, char_base_s + char_s, char_base_s + char_e)
                    )
                else:
                    self.truncate_span_count += 1
            if len(span_dranges) == 0:
                # span does not have any valid location in truncated sequences
                continue

            span_tokens = self.tokenizer.dee_tokenize(mspan)
            span_token_ids = tuple(self.tokenizer.convert_tokens_to_ids(span_tokens))

            mspan2span_idx[mspan] = len(span_token_ids_list)
            span_token_ids_list.append(span_token_ids)
            span_dranges_list.append(span_dranges)
        assert len(span_token_ids_list) == len(span_dranges_list) == len(mspan2span_idx)

        if len(span_token_ids_list) == 0 and not dee_example.only_inference:
            logger.warning("Neglect example {}".format(ex_idx))
            return None

        # 3. prepare doc-level event feature
        # event_type_labels: event_type_index -> event_type_exist_sign (1: exist, 0: no)
        # event_arg_idxs_objs_list: event_type_index -> event_obj_index -> event_arg_index -> arg_span_token_ids

        event_type_labels = []  # event_type_idx -> event_type_exist_sign (1 or 0)
        event_arg_idxs_objs_list = (
            []
        )  # event_type_idx -> event_obj_idx -> event_arg_idx -> span_idx
        for event_idx, event_type in enumerate(self.event_type_list):
            event_fields = self.event_fields_list[event_idx]

            if event_type not in dee_example.event_type2event_objs:
                event_type_labels.append(0)
                event_arg_idxs_objs_list.append(None)
            else:
                event_objs = dee_example.event_type2event_objs[event_type]

                event_arg_idxs_objs = []
                for event_obj in event_objs:
                    assert isinstance(event_obj, self.template.BaseEvent)

                    event_arg_idxs = []
                    any_valid_flag = False
                    for field in event_fields:
                        arg_span = event_obj.field2content[field]

                        if arg_span is None or arg_span not in mspan2span_idx:
                            # arg_span can be none or valid span is truncated
                            arg_span_idx = None
                        else:
                            # when constructing data files,
                            # must ensure event arg span is covered by the total span collections
                            arg_span_idx = mspan2span_idx[arg_span]
                            any_valid_flag = True

                        event_arg_idxs.append(arg_span_idx)

                    if any_valid_flag:
                        event_arg_idxs_objs.append(tuple(event_arg_idxs))

                if event_arg_idxs_objs:
                    event_type_labels.append(1)
                    event_arg_idxs_objs_list.append(event_arg_idxs_objs)
                    self.event_type2num[event_type] += 1
                else:
                    event_type_labels.append(0)
                    event_arg_idxs_objs_list.append(None)

        doc_type = {
            "o2o": 0,
            "o2m": 1,
            "m2m": 2,
            "unk": 3,
        }[dee_example.doc_type]
        dee_feature = DEPPNFeature(
            annguid,
            ex_idx,
            doc_type,
            doc_token_id_mat,
            doc_token_mask_mat,
            doc_token_label_mat,
            span_token_ids_list,
            span_dranges_list,
            event_type_labels,
            event_arg_idxs_objs_list,
            valid_sent_num=valid_sent_num,
        )

        return dee_feature

    def __call__(self, dee_examples, log_example_num=0):
        """Convert examples to features suitable for document-level event extraction"""
        assert len(dee_examples) > 0
        dee_features = []
        self.truncate_doc_count = 0
        self.truncate_span_count = 0
        self.ner_fea_converter.truncate_count = 0

        remove_ex_cnt = 0
        for ex_idx, dee_example in enumerate(tqdm(dee_examples, ncols=80, ascii=True)):
            if ex_idx < log_example_num:
                dee_feature = self.convert_example_to_feature(
                    ex_idx - remove_ex_cnt, dee_example, log_flag=True
                )
            else:
                dee_feature = self.convert_example_to_feature(
                    ex_idx - remove_ex_cnt, dee_example, log_flag=False
                )

            if dee_feature is None:
                remove_ex_cnt += 1
                continue

            dee_features.append(dee_feature)

        logger.info(
            "{} documents, ignore {} examples, truncate {} docs, {} sents, {} spans".format(
                len(dee_examples),
                remove_ex_cnt,
                self.truncate_doc_count,
                self.ner_fea_converter.truncate_count,
                self.truncate_span_count,
            )
        )
        print(self.event_type2num)

        return dee_features


def convert_deppn_features_to_dataset(dee_features):
    # just view a list of doc_fea as the dataset, that only requires __len__, __getitem__
    assert len(dee_features) > 0 and isinstance(dee_features[0], DEPPNFeature)

    return dee_features
