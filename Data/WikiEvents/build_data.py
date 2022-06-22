import json
from collections import defaultdict

from dee.utils import default_dump_json, default_load_json


def load_jsonlines(filepath):
    data = []
    with open(filepath, "rt", encoding="utf-8") as fin:
        for line in fin:
            d = json.loads(line.strip())
            data.append(d)
    return data


def load_jsonlines_iterator(filepath):
    with open(filepath, "rt", encoding="utf-8") as fin:
        for line in fin:
            d = json.loads(line.strip())
            yield d


def extract_event_template(input_filepaths: list, output_filepath: str):
    tot_data = []
    for input_filepath in input_filepaths:
        data = load_jsonlines(input_filepath)
        tot_data += data

    event_type_to_roles = defaultdict(set)
    for d in tot_data:
        for event_ins in d["event_mentions"]:
            event_type = event_ins["event_type"]
            for arg in event_ins["arguments"]:
                event_type_to_roles[event_type].add(arg["role"])

    for event_type in event_type_to_roles:
        event_type_to_roles[event_type] = list(event_type_to_roles[event_type])

    default_dump_json(dict(event_type_to_roles), output_filepath)


def get_doc_type(recguid_eventname_eventdict_list):
    doc_type = "unk"
    num_ins = len(recguid_eventname_eventdict_list)
    if num_ins == 1:
        doc_type = "o2o"
    else:
        event_types = {x[1] for x in recguid_eventname_eventdict_list}
        if len(event_types) == num_ins:
            doc_type = "o2m"
        else:
            doc_type = "m2m"
    return doc_type


def get_string_from_absolute_index(
    sentence_tokens, sent_idx, start, end, sent_lens=None
):
    if sent_lens is None:
        sent_lens = list(map(len, sentence_tokens))
    cumsum = sum(sent_lens[:sent_idx])
    start = start - cumsum
    end = end - cumsum
    return " ".join(sentence_tokens[sent_idx][start:end]), [sent_idx, start, end]


def build_file(
    event_template_filepath, input_filepath, output_filepath, with_trigger=True
):
    """Convert file into ChFinAnn format.
    Sentences, spans are `List[str]` instead of `str`.
    """
    event_type_to_roles = default_load_json(event_template_filepath)

    new_data = []
    for ins in load_jsonlines_iterator(input_filepath):
        doc_id = ins["doc_id"]
        detail = {}

        # sentences
        sentences = []
        sentence_tokens = []
        for ins_sent in ins["sentences"]:
            # ins_sent[0] is a list of [token, start, end]
            sentences.append(" ".join([token[0] for token in ins_sent[0]]))
            sentence_tokens.append([token[0] for token in ins_sent[0]])
        detail["sentences"] = sentences

        # ann_valid_mspans, ann_valid_dranges and ann_mspan2guess_field
        sent_lens = list(map(len, sentence_tokens))
        span2dranges = defaultdict(list)
        ann_mspan2guess_field = {}
        arg_mention_to_space_split_string = {}
        for entity_mention in ins["entity_mentions"]:
            sent_idx = entity_mention["sent_idx"]
            cumsum_lens = sum(sent_lens[:sent_idx])
            start_idx = entity_mention["start"] - cumsum_lens
            end_idx = entity_mention["end"] - cumsum_lens
            drange = [sent_idx, start_idx, end_idx]
            ent_string = " ".join(sentence_tokens[sent_idx][start_idx:end_idx])
            arg_mention_to_space_split_string[entity_mention["text"]] = ent_string
            ent_type = entity_mention["entity_type"]
            ann_mspan2guess_field[ent_string] = ent_type
            span2dranges[ent_string].append(drange)

        # event instances
        recguid_eventname_eventdict_list = []
        for event_idx, event_ins in enumerate(ins["event_mentions"]):
            event_type = event_ins["event_type"]
            template_roles = event_type_to_roles[event_type]
            role_to_entity = {role: None for role in template_roles}
            if with_trigger:
                trigger_text = event_ins["trigger"]["text"]
                trigger_text, trigger_pos = get_string_from_absolute_index(
                    sentence_tokens,
                    event_ins["trigger"]["sent_idx"],
                    event_ins["trigger"]["start"],
                    event_ins["trigger"]["end"],
                )
                assert trigger_text.replace(" ", "") == event_ins["trigger"][
                    "text"
                ].replace(" ", "")
                if trigger_text not in span2dranges:
                    span2dranges[trigger_text].append(trigger_pos)
                role_to_entity["Trigger"] = trigger_text
                if trigger_text not in ann_mspan2guess_field:
                    ann_mspan2guess_field[trigger_text] = "Trigger"
            for arg in event_ins["arguments"]:
                arg_text = arg["text"]
                arg_text = arg_mention_to_space_split_string[arg_text]
                role_to_entity[arg["role"]] = arg["text"]
            recguid_eventname_eventdict_list.append(
                [event_idx, event_type, role_to_entity]
            )

        ann_valid_mspans, ann_valid_dranges = [], []
        for span, dranges in span2dranges.items():
            ann_valid_mspans.append(span)
            ann_valid_dranges.extend(dranges)

        detail["ann_valid_mspans"] = ann_valid_mspans
        detail["ann_valid_dranges"] = ann_valid_dranges
        detail["ann_mspan2dranges"] = span2dranges
        detail["ann_mspan2guess_field"] = ann_mspan2guess_field
        detail["recguid_eventname_eventdict_list"] = recguid_eventname_eventdict_list
        detail["doc_type"] = get_doc_type(recguid_eventname_eventdict_list)

        new_data.append([doc_id, detail])

    default_dump_json(new_data, output_filepath, indent=None)


if __name__ == "__main__":
    extract_event_template(
        [
            "Data/WikiEvents/train.jsonl",
            "Data/WikiEvents/dev.jsonl",
            "Data/WikiEvents/test.jsonl",
        ],
        "Data/WikiEvents/event_template.json",
    )
    for dataname in ["train", "dev", "test"]:
        build_file(
            "Data/WikiEvents/event_template.json",
            f"Data/WikiEvents/{dataname}.jsonl",
            f"Data/WikiEvents/{dataname}.post.wTgg.json",
            with_trigger=True,
        )
        build_file(
            "Data/WikiEvents/event_template.json",
            f"Data/WikiEvents/{dataname}.jsonl",
            f"Data/WikiEvents/{dataname}.post.woTgg.json",
            with_trigger=False,
        )
