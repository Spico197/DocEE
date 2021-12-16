import torch
from torch.utils.data import TensorDataset

from dee.utils import default_load_json, logger


class NERExample(object):
    basic_entity_label = "O"

    def __init__(self, guid, text, tokenized_text, entity_range_span_types):
        self.guid = guid
        self.text = text
        self.tokenized_text = tokenized_text
        self.num_chars = len(tokenized_text)
        entity_range_span_types = list(set(entity_range_span_types))
        self.entity_range_span_types = sorted(
            entity_range_span_types, key=lambda x: x[0]
        )

    def get_char_entity_labels(self):
        char_entity_labels = []
        char_idx = 0
        ent_idx = 0
        while ent_idx < len(self.entity_range_span_types):
            (ent_cid_s, ent_cid_e), ent_span, ent_type = self.entity_range_span_types[
                ent_idx
            ]
            assert ent_cid_s < ent_cid_e <= self.num_chars

            if ent_cid_s > char_idx:
                char_entity_labels.append(NERExample.basic_entity_label)
                char_idx += 1
            elif ent_cid_s == char_idx:
                # tmp_ent_labels = [ent_type] * (ent_cid_e - ent_cid_s)
                tmp_ent_labels = ["B-" + ent_type] + ["I-" + ent_type] * (
                    ent_cid_e - ent_cid_s - 1
                )
                char_entity_labels.extend(tmp_ent_labels)
                char_idx = ent_cid_e
                ent_idx += 1
            else:
                # logger.error('Example GUID {}'.format(self.guid))
                # logger.error('NER conflicts at char_idx {}, ent_cid_s {}'.format(char_idx, ent_cid_s))
                # logger.error(self.text[max(0, char_idx - 20):min(len(self.text), char_idx + 20)])
                # logger.error(self.entity_range_span_types[ent_idx - 1:ent_idx + 1])
                ent_idx += 1
                # breakpoint()
                # raise RuntimeError('Unexpected logic error')

        char_entity_labels.extend(
            [NERExample.basic_entity_label] * (self.num_chars - char_idx)
        )
        assert len(char_entity_labels) == self.num_chars

        return char_entity_labels

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

    def __repr__(self):
        ex_str = "NERExample(guid={}, text={}, entity_info={}".format(
            self.guid, self.tokenized_text, str(self.entity_range_span_types)
        )
        return ex_str


class NERExampleLoader(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, dataset_json_path):
        return self.load_ner_dataset(dataset_json_path)

    def load_ner_dataset(self, dataset_json_path):
        total_ner_examples = []
        annguid2detail_align_info = default_load_json(dataset_json_path)
        for annguid, detail_align_info in annguid2detail_align_info.items():
            sents = detail_align_info["sentences"]
            ann_valid_mspans = detail_align_info["ann_valid_mspans"]
            ann_valid_dranges = detail_align_info["ann_valid_dranges"]
            ann_mspan2guess_field = detail_align_info["ann_mspan2guess_field"]
            assert len(ann_valid_dranges) == len(ann_valid_mspans)

            sent_idx2mrange_mspan_mfield_tuples = {}
            for drange, mspan in zip(ann_valid_dranges, ann_valid_mspans):
                sent_idx, char_s, char_e = drange
                sent_mrange = (char_s, char_e)

                sent_text = sents[sent_idx]
                sent_text = self.tokenizer.dee_tokenize(sent_text)
                assert sent_text[char_s:char_e] == self.tokenizer.dee_tokenize(mspan)

                guess_field = ann_mspan2guess_field[mspan]

                if sent_idx not in sent_idx2mrange_mspan_mfield_tuples:
                    sent_idx2mrange_mspan_mfield_tuples[sent_idx] = []
                sent_idx2mrange_mspan_mfield_tuples[sent_idx].append(
                    (sent_mrange, mspan, guess_field)
                )

            for sent_idx in range(len(sents)):
                sent_text = sents[sent_idx]
                if sent_idx in sent_idx2mrange_mspan_mfield_tuples:
                    mrange_mspan_mfield_tuples = sent_idx2mrange_mspan_mfield_tuples[
                        sent_idx
                    ]
                else:
                    mrange_mspan_mfield_tuples = []

                total_ner_examples.append(
                    NERExample(
                        "{}-{}".format(annguid, sent_idx),
                        sent_text,
                        self.tokenizer.dee_tokenize(sent_text),
                        mrange_mspan_mfield_tuples,
                    )
                )

        return total_ner_examples


class NERFeature(object):
    def __init__(self, input_ids, input_masks, segment_ids, label_ids, seq_len=None):
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.seq_len = seq_len

    def __repr__(self):
        fea_strs = [
            "NERFeature(real_seq_len={}".format(self.seq_len),
        ]
        info_template = "  {:5} {:9} {:5} {:7} {:7}"
        fea_strs.append(
            info_template.format("index", "input_ids", "masks", "seg_ids", "lbl_ids")
        )
        max_print_len = 10
        idx = 0
        for tid, mask, segid, lid in zip(
            self.input_ids, self.input_masks, self.segment_ids, self.label_ids
        ):
            fea_strs.append(info_template.format(idx, tid, mask, segid, lid))
            idx += 1
            if idx >= max_print_len:
                break
        fea_strs.append(info_template.format("...", "...", "...", "...", "..."))
        fea_strs.append(")")

        fea_str = "\n".join(fea_strs)
        return fea_str


class NERFeatureConverter(object):
    def __init__(
        self,
        entity_label_list,
        max_seq_len,
        tokenizer,
        include_cls=True,
        include_sep=True,
    ):
        self.entity_label_list = entity_label_list
        self.max_seq_len = max_seq_len  # used to normalize sequence length
        self.tokenizer = tokenizer
        self.entity_label2index = {  # for entity label to label index mapping
            entity_label: idx for idx, entity_label in enumerate(self.entity_label_list)
        }

        self.include_cls = include_cls
        self.include_sep = include_sep

        # used to track how many examples have been truncated
        self.truncate_count = 0
        # used to track the maximum length of input sentences
        self.data_max_seq_len = -1

    def convert_example_to_feature(self, ner_example, log_flag=False):
        ex_tokens = self.tokenizer.dee_tokenize(ner_example.text)
        ex_entity_labels = ner_example.get_char_entity_labels()

        assert len(ex_tokens) == len(ex_entity_labels)

        # get valid token sequence length
        valid_token_len = self.max_seq_len
        if self.include_cls:
            valid_token_len -= 1
        if self.include_sep:
            valid_token_len -= 1

        # truncate according to max_seq_len and record some statistics
        self.data_max_seq_len = max(self.data_max_seq_len, len(ex_tokens))
        if len(ex_tokens) > valid_token_len:
            ex_tokens = ex_tokens[:valid_token_len]
            ex_entity_labels = ex_entity_labels[:valid_token_len]

            self.truncate_count += 1

        basic_label_index = self.entity_label2index[NERExample.basic_entity_label]

        # add bert-specific token
        if self.include_cls:
            fea_tokens = ["[CLS]"]
            fea_token_labels = [NERExample.basic_entity_label]
            fea_label_ids = [basic_label_index]
        else:
            fea_tokens = []
            fea_token_labels = []
            fea_label_ids = []

        for token, ent_label in zip(ex_tokens, ex_entity_labels):
            fea_tokens.append(token)
            fea_token_labels.append(ent_label)

            if ent_label in self.entity_label2index:
                fea_label_ids.append(self.entity_label2index[ent_label])
            else:
                fea_label_ids.append(basic_label_index)

        if self.include_sep:
            fea_tokens.append("[SEP]")
            fea_token_labels.append(NERExample.basic_entity_label)
            fea_label_ids.append(basic_label_index)

        assert (
            len(fea_tokens)
            == len(fea_token_labels)
            == len(fea_label_ids)
            <= self.max_seq_len
        )

        fea_input_ids = self.tokenizer.convert_tokens_to_ids(fea_tokens)
        fea_seq_len = len(fea_input_ids)
        fea_segment_ids = [0] * fea_seq_len
        fea_masks = [1] * fea_seq_len

        # feature is padded to max_seq_len, but fea_seq_len is the real length
        while len(fea_input_ids) < self.max_seq_len:
            fea_input_ids.append(0)
            fea_label_ids.append(0)
            fea_masks.append(0)
            fea_segment_ids.append(0)

        assert (
            len(fea_input_ids)
            == len(fea_label_ids)
            == len(fea_masks)
            == len(fea_segment_ids)
            == self.max_seq_len
        )

        if log_flag:
            logger.info("*** Example ***")
            logger.info("guid: %s" % ner_example.guid)
            info_template = "{:8} {:4} {:2} {:2} {:2} {}"
            logger.info(
                info_template.format(
                    "TokenId", "Token", "Mask", "SegId", "LabelId", "Label"
                )
            )
            for tid, token, mask, segid, lid, label in zip(
                fea_input_ids,
                fea_tokens,
                fea_masks,
                fea_segment_ids,
                fea_label_ids,
                fea_token_labels,
            ):
                logger.info(info_template.format(tid, token, mask, segid, lid, label))
            if len(fea_input_ids) > len(fea_tokens):
                sid = len(fea_tokens)
                logger.info(
                    info_template.format(
                        fea_input_ids[sid],
                        "[PAD]",
                        fea_masks[sid],
                        fea_segment_ids[sid],
                        fea_label_ids[sid],
                        "O",
                    )
                    + " x {}".format(len(fea_input_ids) - len(fea_tokens))
                )

        return NERFeature(
            fea_input_ids,
            fea_masks,
            fea_segment_ids,
            fea_label_ids,
            seq_len=fea_seq_len,
        )

    def __call__(self, ner_examples, log_example_num=0):
        """Convert examples to features suitable for ner models"""
        self.truncate_count = 0
        self.data_max_seq_len = -1
        ner_features = []

        for ex_index, ner_example in enumerate(ner_examples):
            if ex_index < log_example_num:
                ner_feature = self.convert_example_to_feature(
                    ner_example, log_flag=True
                )
            else:
                ner_feature = self.convert_example_to_feature(
                    ner_example, log_flag=False
                )

            ner_features.append(ner_feature)

        logger.info(
            "{} examples in total, {} truncated example, max_sent_len={}".format(
                len(ner_examples), self.truncate_count, self.data_max_seq_len
            )
        )

        return ner_features


def convert_ner_features_to_dataset(ner_features):
    all_input_ids = torch.tensor([f.input_ids for f in ner_features], dtype=torch.long)
    # very important to use the mask type of uint8 to support advanced indexing
    all_input_masks = torch.tensor(
        [f.input_masks for f in ner_features], dtype=torch.uint8
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in ner_features], dtype=torch.long
    )
    all_label_ids = torch.tensor([f.label_ids for f in ner_features], dtype=torch.long)
    all_seq_len = torch.tensor([f.seq_len for f in ner_features], dtype=torch.long)
    ner_tensor_dataset = TensorDataset(
        all_input_ids, all_input_masks, all_segment_ids, all_label_ids, all_seq_len
    )

    return ner_tensor_dataset
