import copy
import re
from collections import Counter, defaultdict

import torch
from tqdm import tqdm

from dee.utils import default_load_json, logger, regex_extractor

from .ner import NERExample, NERFeatureConverter


class DEEExample(object):
    def __init__(
        self,
        annguid,
        detail_align_dict,
        template,
        tokenizer,
        only_inference=False,
        inlcude_complementary_ents=False,
    ):
        self.guid = annguid
        # [sent_text, ...]
        self.sentences = detail_align_dict["sentences"]
        self.doc_type = detail_align_dict.get("doc_type", "unk")
        self.num_sentences = len(self.sentences)
        if inlcude_complementary_ents:
            self.complementary_field2ents = regex_extractor.extract_doc(
                detail_align_dict["sentences"]
            )
        else:
            self.complementary_field2ents = {}

        if only_inference:
            # set empty entity/event information
            self.only_inference = True
            self.ann_valid_mspans = []
            self.ann_mspan2dranges = {}
            self.ann_mspan2guess_field = {}
            self.recguid_eventname_eventdict_list = []
            self.num_events = 0
            self.sent_idx2srange_mspan_mtype_tuples = {}
            self.event_type2event_objs = {}
        else:
            # set event information accordingly
            self.only_inference = False

            if inlcude_complementary_ents:
                # build index
                comp_ents_sent_index = defaultdict(list)
                comp_ents_start_index = defaultdict(list)
                comp_ents_end_index = defaultdict(list)
                for raw_field, ents in self.complementary_field2ents.items():
                    # field = 'Other' + field.title()
                    field = "OtherType"
                    for ent, pos_span in ents:
                        pos_span = list(pos_span)
                        if ent not in detail_align_dict["ann_valid_mspans"]:
                            comp_ents_sent_index[pos_span[0]].append(
                                [ent, raw_field, pos_span]
                            )
                            comp_ents_start_index[(pos_span[0], pos_span[1])].append(
                                [ent, raw_field, pos_span]
                            )
                            comp_ents_end_index[(pos_span[0], pos_span[2])].append(
                                [ent, raw_field, pos_span]
                            )

                # remove overlaped Date
                mspan2fields = copy.deepcopy(detail_align_dict["ann_mspan2guess_field"])
                mspan2dranges = copy.deepcopy(detail_align_dict["ann_mspan2dranges"])
                for ent, field in mspan2fields.items():
                    for drange in mspan2dranges[ent]:
                        for s_ent in comp_ents_sent_index.get(drange[0], []):
                            s_ent, raw_field, pos_span = s_ent
                            if (
                                drange[1] <= pos_span[1] < drange[2]
                                or drange[1] < pos_span[2] <= drange[2]
                            ):
                                if [s_ent, pos_span] in self.complementary_field2ents[
                                    raw_field
                                ]:
                                    self.complementary_field2ents[raw_field].remove(
                                        [s_ent, pos_span]
                                    )

                for raw_field, ents in self.complementary_field2ents.items():
                    field = "OtherType"
                    for ent, pos_span in ents:
                        pos_span = list(pos_span)
                        if ent not in detail_align_dict["ann_valid_mspans"]:
                            detail_align_dict["ann_valid_mspans"].append(ent)
                            detail_align_dict["ann_mspan2guess_field"][ent] = field
                            detail_align_dict["ann_mspan2dranges"][ent] = [pos_span]
                        elif (
                            list(pos_span)
                            not in detail_align_dict["ann_mspan2dranges"][ent]
                        ):
                            detail_align_dict["ann_mspan2dranges"][ent].append(pos_span)

                # OtherType wrong ratio annotation correction
                mspan2fields = copy.deepcopy(detail_align_dict["ann_mspan2guess_field"])
                mspan2dranges = copy.deepcopy(detail_align_dict["ann_mspan2dranges"])
                for ent, field in mspan2fields.items():
                    if field == "OtherType" and "%" in ent:
                        for drange in mspan2dranges[ent]:
                            if self.sentences[drange[0]][drange[1] - 1] in "0123456789":
                                # not-complete ratio, drop
                                detail_align_dict["ann_valid_mspans"].remove(ent)
                                detail_align_dict["ann_mspan2guess_field"].pop(ent)
                                detail_align_dict["ann_mspan2dranges"].pop(ent)
                                break

            # [span_text, ...]
            self.ann_valid_mspans = detail_align_dict["ann_valid_mspans"]
            # span_text -> [drange_tuple, ...]
            self.ann_mspan2dranges = detail_align_dict["ann_mspan2dranges"]
            # span_text -> guessed_field_name
            self.ann_mspan2guess_field = detail_align_dict["ann_mspan2guess_field"]
            # [(recguid, event_name, event_dict), ...]
            self.recguid_eventname_eventdict_list = detail_align_dict[
                "recguid_eventname_eventdict_list"
            ]
            self.num_events = len(self.recguid_eventname_eventdict_list)

            # for create ner examples
            # sentence_index -> [(sent_match_range, match_span, match_type), ...]
            self.sent_idx2srange_mspan_mtype_tuples = {}
            for sent_idx in range(self.num_sentences):
                self.sent_idx2srange_mspan_mtype_tuples[sent_idx] = []

            for mspan in self.ann_valid_mspans:
                for drange in self.ann_mspan2dranges[mspan]:
                    sent_idx, char_s, char_e = drange
                    sent_mrange = (char_s, char_e)

                    sent_text = self.sentences[sent_idx]
                    sent_text = tokenizer.dee_tokenize(sent_text)
                    if sent_text[char_s:char_e] != tokenizer.dee_tokenize(mspan):
                        raise ValueError(
                            "GUID: {} span range is not correct, span={}, range={}, sent={}".format(
                                annguid, mspan, str(sent_mrange), sent_text
                            )
                        )

                    guess_field = self.ann_mspan2guess_field[mspan]

                    self.sent_idx2srange_mspan_mtype_tuples[sent_idx].append(
                        (sent_mrange, mspan, guess_field)
                    )

            # for create event objects
            # the length of event_objs should >= 1
            self.event_type2event_objs = {}
            for (
                mrecguid,
                event_name,
                event_dict,
            ) in self.recguid_eventname_eventdict_list:
                event_class = template.event_type2event_class[event_name]
                event_obj = event_class()
                # assert isinstance(event_obj, BaseEvent)
                event_obj.update_by_dict(event_dict, recguid=mrecguid)

                if event_obj.name in self.event_type2event_objs:
                    self.event_type2event_objs[event_obj.name].append(event_obj)
                else:
                    self.event_type2event_objs[event_name] = [event_obj]

    def __repr__(self):
        dee_str = "DEEExample (\n"
        dee_str += "  guid: {},\n".format(repr(self.guid))

        if not self.only_inference:
            dee_str += "  span info: (\n"
            for span_idx, span in enumerate(self.ann_valid_mspans):
                gfield = self.ann_mspan2guess_field[span]
                dranges = self.ann_mspan2dranges[span]
                dee_str += "    {:2} {:20} {:30} {}\n".format(
                    span_idx, span, gfield, str(dranges)
                )
            dee_str += "  ),\n"

            dee_str += "  event info: (\n"
            event_str_list = repr(self.event_type2event_objs).split("\n")
            for event_str in event_str_list:
                dee_str += "    {}\n".format(event_str)
            dee_str += "  ),\n"

        dee_str += "  sentences: (\n"
        for sent_idx, sent in enumerate(self.sentences):
            dee_str += "    {:2} {}\n".format(sent_idx, sent)
        dee_str += "  ),\n"

        dee_str += ")\n"

        return dee_str

    @staticmethod
    def get_event_type_fields_pairs(template):
        return list(template.event_type_fields_list)

    @staticmethod
    def get_entity_label_list(template):
        visit_set = set()
        entity_label_list = [NERExample.basic_entity_label]

        for field in template.common_fields:
            if field not in visit_set:
                visit_set.add(field)
                entity_label_list.extend(["B-" + field, "I-" + field])

        for event_name, fields, _, _ in template.event_type_fields_list:
            for field in fields:
                if field not in visit_set:
                    visit_set.add(field)
                    entity_label_list.extend(["B-" + field, "I-" + field])

        return entity_label_list


class DEEExampleLoader(object):
    def __init__(
        self,
        template,
        tokenizer,
        rearrange_sent_flag,
        max_sent_len,
        drop_irr_ents_flag=False,
        include_complementary_ents=False,
        filtered_data_types=["o2o", "o2m", "m2m"],
    ):
        self.template = template
        self.tokenizer = tokenizer
        self.rearrange_sent_flag = rearrange_sent_flag
        self.max_sent_len = max_sent_len
        self.drop_irr_ents_flag = drop_irr_ents_flag
        self.include_complementary_ents_flag = include_complementary_ents
        self.filtered_data_types = filtered_data_types

    def rearrange_sent_info(self, detail_align_info):
        if "ann_valid_dranges" not in detail_align_info:
            detail_align_info["ann_valid_dranges"] = []
        if "ann_mspan2dranges" not in detail_align_info:
            detail_align_info["ann_mspan2dranges"] = {}

        detail_align_info = dict(detail_align_info)
        split_rgx = re.compile("[，：:；;）)]")

        raw_sents = detail_align_info["sentences"]
        doc_text = "".join(raw_sents)
        raw_dranges = detail_align_info["ann_valid_dranges"]
        raw_sid2span_char_set = defaultdict(lambda: set())
        for raw_sid, char_s, char_e in raw_dranges:
            span_char_set = raw_sid2span_char_set[raw_sid]
            span_char_set.update(range(char_s, char_e))

        # try to split long sentences into short ones by comma, colon, semi-colon, bracket
        short_sents = []
        for raw_sid, sent in enumerate(raw_sents):
            span_char_set = raw_sid2span_char_set[raw_sid]
            if len(sent) > self.max_sent_len:
                cur_char_s = 0
                for mobj in split_rgx.finditer(sent):
                    m_char_s, m_char_e = mobj.span()
                    if m_char_s in span_char_set:
                        continue
                    short_sents.append(sent[cur_char_s:m_char_e])
                    cur_char_s = m_char_e
                short_sents.append(sent[cur_char_s:])
            else:
                short_sents.append(sent)

        # merge adjacent short sentences to compact ones that match max_sent_len
        comp_sents = [""]
        for sent in short_sents:
            prev_sent = comp_sents[-1]
            if len(prev_sent + sent) <= self.max_sent_len:
                comp_sents[-1] = prev_sent + sent
            else:
                comp_sents.append(sent)

        # get global sentence character base indexes
        raw_char_bases = [0]
        for sent in raw_sents:
            raw_char_bases.append(raw_char_bases[-1] + len(sent))
        comp_char_bases = [0]
        for sent in comp_sents:
            comp_char_bases.append(comp_char_bases[-1] + len(sent))

        assert raw_char_bases[-1] == comp_char_bases[-1] == len(doc_text)

        # calculate compact doc ranges
        raw_dranges.sort()
        raw_drange2comp_drange = {}
        prev_comp_sid = 0
        for raw_drange in raw_dranges:
            raw_drange = tuple(
                raw_drange
            )  # important when json dump change tuple to list
            raw_sid, raw_char_s, raw_char_e = raw_drange
            raw_char_base = raw_char_bases[raw_sid]
            doc_char_s = raw_char_base + raw_char_s
            doc_char_e = raw_char_base + raw_char_e
            assert doc_char_s >= comp_char_bases[prev_comp_sid]

            cur_comp_sid = prev_comp_sid
            for cur_comp_sid in range(prev_comp_sid, len(comp_sents)):
                if doc_char_e <= comp_char_bases[cur_comp_sid + 1]:
                    prev_comp_sid = cur_comp_sid
                    break
            comp_char_base = comp_char_bases[cur_comp_sid]
            assert (
                comp_char_base
                <= doc_char_s
                < doc_char_e
                <= comp_char_bases[cur_comp_sid + 1]
            )
            comp_char_s = doc_char_s - comp_char_base
            comp_char_e = doc_char_e - comp_char_base
            comp_drange = (cur_comp_sid, comp_char_s, comp_char_e)

            raw_drange2comp_drange[raw_drange] = comp_drange
            assert (
                raw_sents[raw_drange[0]][raw_drange[1] : raw_drange[2]]
                == comp_sents[comp_drange[0]][comp_drange[1] : comp_drange[2]]
            )

        # update detailed align info with rearranged sentences
        detail_align_info["sentences"] = comp_sents
        detail_align_info["ann_valid_dranges"] = [
            raw_drange2comp_drange[tuple(raw_drange)]
            for raw_drange in detail_align_info["ann_valid_dranges"]
        ]
        ann_mspan2comp_dranges = {}
        for ann_mspan, mspan_raw_dranges in detail_align_info[
            "ann_mspan2dranges"
        ].items():
            comp_dranges = [
                raw_drange2comp_drange[tuple(raw_drange)]
                for raw_drange in mspan_raw_dranges
            ]
            ann_mspan2comp_dranges[ann_mspan] = comp_dranges
        detail_align_info["ann_mspan2dranges"] = ann_mspan2comp_dranges

        return detail_align_info

    def drop_irr_ents(self, detail_align_info):
        ann_valid_mspans = []
        ann_valid_dranges = []
        ann_mspan2dranges = {}
        ann_mspan2guess_field = {}

        real_valid_spans = set()
        for _, _, role2span in detail_align_info["recguid_eventname_eventdict_list"]:
            spans = set(role2span.values())
            real_valid_spans.update(spans)
        if None in real_valid_spans:
            real_valid_spans.remove(None)
        for span in real_valid_spans:
            ann_valid_mspans.append(span)
            ann_valid_dranges.extend(detail_align_info["ann_mspan2dranges"][span])
            ann_mspan2dranges[span] = detail_align_info["ann_mspan2dranges"][span]
            ann_mspan2guess_field[span] = detail_align_info["ann_mspan2guess_field"][
                span
            ]

        detail_align_info["ann_valid_mspans"] = ann_valid_mspans
        detail_align_info["ann_valid_dranges"] = ann_valid_dranges
        detail_align_info["ann_mspan2dranges"] = ann_mspan2dranges
        detail_align_info["ann_mspan2guess_field"] = ann_mspan2guess_field
        return detail_align_info

    def convert_dict_to_example(self, annguid, detail_align_info, only_inference=False):
        if self.drop_irr_ents_flag:
            detail_align_info = self.drop_irr_ents(detail_align_info)
        if self.rearrange_sent_flag:
            detail_align_info = self.rearrange_sent_info(detail_align_info)
        dee_example = DEEExample(
            annguid,
            detail_align_info,
            self.template,
            self.tokenizer,
            only_inference=only_inference,
            inlcude_complementary_ents=self.include_complementary_ents_flag,
        )

        return dee_example

    def __call__(self, dataset_json_path, only_inference=False):
        total_dee_examples = []
        annguid_aligninfo_list = default_load_json(dataset_json_path)
        for annguid, detail_align_info in annguid_aligninfo_list:
            if detail_align_info["doc_type"] not in self.filtered_data_types:
                continue
            dee_example = self.convert_dict_to_example(
                annguid, detail_align_info, only_inference=only_inference
            )
            total_dee_examples.append(dee_example)

        return total_dee_examples


class DEEFeature(object):
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
        # self.doc_type = torch.tensor(doc_type, dtype=torch.uint8)
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
        token_tup2pred_span_idx = {
            token_tup: pred_span_idx
            for pred_span_idx, token_tup in enumerate(pred_span_token_tup_list)
        }
        gold_span_idx2pred_span_idx = {}
        # pred_span_idx2gold_span_idx = {}
        missed_span_idx_list = []  # in terms of self
        missed_sent_idx_list = []  # in terms of self
        for gold_span_idx, token_tup in enumerate(self.span_token_ids_list):
            # tzhu: token_tup: token ids for each span
            if token_tup in token_tup2pred_span_idx:
                pred_span_idx = token_tup2pred_span_idx[token_tup]
                gold_span_idx2pred_span_idx[gold_span_idx] = pred_span_idx
                # pred_span_idx2gold_span_idx[pred_span_idx] = gold_span_idx
            else:  # tzhu: not predicted
                missed_span_idx_list.append(gold_span_idx)
                for gold_drange in self.span_dranges_list[gold_span_idx]:
                    missed_sent_idx_list.append(gold_drange[0])
        missed_sent_idx_list = list(set(missed_sent_idx_list))

        pred_event_arg_idxs_objs_list = []
        for event_arg_idxs_objs in self.event_arg_idxs_objs_list:
            if event_arg_idxs_objs is None:
                pred_event_arg_idxs_objs_list.append(None)
            else:
                pred_event_arg_idxs_objs = []
                for event_arg_idxs in event_arg_idxs_objs:
                    pred_event_arg_idxs = []
                    for gold_span_idx in event_arg_idxs:
                        if gold_span_idx in gold_span_idx2pred_span_idx:
                            pred_event_arg_idxs.append(
                                gold_span_idx2pred_span_idx[gold_span_idx]
                            )
                        else:
                            pred_event_arg_idxs.append(None)

                    pred_event_arg_idxs_objs.append(tuple(pred_event_arg_idxs))
                pred_event_arg_idxs_objs_list.append(pred_event_arg_idxs_objs)

        # event_idx -> field_idx -> pre_path -> cur_span_idx_set
        pred_dag_info = self.build_dag_info(pred_event_arg_idxs_objs_list)

        if return_miss:
            return pred_dag_info, missed_span_idx_list, missed_sent_idx_list
        else:
            return pred_dag_info

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
                            # tzhu: to find the sentence with the biggest amount of spans
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
            doc_sent_labels.append(
                sent_labels
            )  # tzhu: the shape is: #sent x #event_type

        return event_idx2key_sent_idx_set, doc_sent_labels

    @staticmethod
    def build_dag_info(event_arg_idxs_objs_list):
        # event_idx -> field_idx -> pre_path -> {span_idx, ...}
        # pre_path is tuple of span_idx. tzhu: tuple is hashable in python
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
                        # tzhu: make new branches for different role paths, that's why we need ``set`` structure
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


class DEEFeatureConverter(object):
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
        for event_idx, (event_type, event_fields, _, _) in enumerate(
            self.event_type_fields_pairs
        ):
            self.event_type2index[event_type] = event_idx
            self.event_type_list.append(event_type)
            self.event_fields_list.append(event_fields)

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
                else:
                    event_type_labels.append(0)
                    event_arg_idxs_objs_list.append(None)

        doc_type = {
            "o2o": 0,
            "o2m": 1,
            "m2m": 2,
            "unk": 3,
        }[dee_example.doc_type]
        dee_feature = DEEFeature(
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

        return dee_features


def convert_dee_features_to_dataset(dee_features):
    # just view a list of doc_fea as the dataset, that only requires __len__, __getitem__
    assert len(dee_features) > 0 and isinstance(dee_features[0], DEEFeature)

    return dee_features
