import json
import re
from collections import Counter, defaultdict
from statistics import median

from dee.event_types import get_event_template


def load_line_json_iterator(filepath):
    with open(filepath, "rt", encoding="utf-8") as fin:
        for line in fin:
            d = json.loads(line.strip())
            yield d


def load_json(filepath):
    with open(filepath, "rt", encoding="utf-8") as fin:
        return json.load(fin)


def sent_seg(
    text,
    special_seg_indicators=None,
    lang="zh",
    punctuations=None,
    quotation_seg_mode=True,
) -> list:
    """
    cut texts into sentences (in chinese language).
    Args:
        text <str>: texts ready to be cut
        special_seg_indicators <list>: some special segment indicators and
            their replacement ( [indicator, replacement] ), in baike data,
            this argument could be `[('###', '\n'), ('%%%', ' '), ('%%', ' ')]`
        lang <str>: languages that your corpus is, support `zh` for Chinese
            and `en` for English now.
        punctuations <set>: you can split the texts by specified punctuations.
            texts will not be splited by `;`, so you can specify them by your own.
        quotation_seg_mode <bool>: if True, the quotations will be regarded as a
            part of the former sentence.
            e.g. `我说：“翠花，上酸菜。”，她说：“欸，好嘞。”`
            the text will be splited into
            ['我说：“翠花，上酸菜。”，', '她说：“欸，好嘞。”'], other than
            ['我说：“翠花，上酸菜。', '”，她说：“欸，好嘞。”']
    Rrturns:
        <list>: a list of strings, which are splited sentences.
    """
    # if texts are not in string format, raise an error
    if not isinstance(text, str):
        raise ValueError

    # if the text is empty, return a list with an empty string
    if len(text) == 0:
        return []

    text_return = text

    # segment on specified indicators
    # special indicators standard, like [('###', '\n'), ('%%%', '\t'), ('\s', '')]
    if special_seg_indicators:
        for indicator in special_seg_indicators:
            text_return = re.sub(indicator[0], indicator[1], text_return)

    if lang == "zh":
        punkt = {"。", "？", "！", "…"}
    elif lang == "en":
        punkt = {".", "?", "!"}
    if punctuations:
        punkt = punkt | punctuations

    if quotation_seg_mode:
        text_return = re.sub(
            "([%s]+[’”`'\"]*)" % ("".join(punkt)), "\\1\n", text_return
        )
    else:
        text_return = re.sub("([{}])".format("".join(punkt)), "\\1\n", text_return)

    # drop sentences with no length
    return [
        s.strip()
        for s in filter(
            lambda x: len(x.strip()) == 1
            and x.strip() not in punkt
            or len(x.strip()) > 0,
            text_return.split("\n"),
        )
    ]


def stat_sent_len(filepath):
    num_sents = []
    sent_len = []
    for d in load_line_json_iterator(filepath):
        sents = sent_seg(d["text"])
        num_sents.append(len(sents))
        lens = [len(sent) for sent in sents]
        sent_len.extend(lens)
        # if min(lens) < 5:
        #     print("================= raw text =================")
        #     print(d["text"])
        #     print("================= processed text =================")
        #     print("\n".join(filter(lambda x: len(x) < 5, sents)))
        #     breakpoint()
    sent_len_counter = Counter(sent_len)
    print(
        (
            f"num_sents: min: {min(num_sents)}, median: {median(num_sents)}, max: {max(num_sents)}\n"
            f"sent_len: min: {min(sent_len)}, median: {median(sent_len)}, max: {max(sent_len)}"
            f"{sent_len_counter.most_common()}"
        )
    )


def get_span_drange(sents, span):
    drange = []
    common_span = (
        span.replace("*", "\*")
        .replace("?", "\?")
        .replace("+", "\+")
        .replace("[", "\[")
        .replace("]", "\]")
        .replace("(", "\(")
        .replace(")", "\)")
        .replace(".", "\.")
        .replace("-", "\-")
    )  # noqa: W605
    for sent_idx, sent in enumerate(sents):
        if len(sent) < len(common_span):
            continue
        for ocurr in re.finditer(common_span, sent):
            span_pos = ocurr.span()
            if (
                (
                    "0" <= span[0] <= "9"
                    and "0" <= sents[sent_idx][span_pos[0] - 1] <= "9"
                    and span_pos[0] - 1 > -1
                )
                or (
                    "0" <= span[0] <= "9"
                    and "0" <= sents[sent_idx][span_pos[0] - 2]
                    and sents[sent_idx][span_pos[0] - 1] == "."
                    and span_pos[0] - 2 > -1
                )
                or (
                    "0" <= span[-1] <= "9"
                    and span_pos[1] < len(sents[sent_idx])
                    and "0" <= sents[sent_idx][span_pos[1]] <= "9"
                )
                or (
                    "0" <= span[-1] <= "9"
                    and span_pos[1] + 1 < len(sents[sent_idx])
                    and sents[sent_idx][span_pos[1]] == "."
                    and "0" <= sents[sent_idx][span_pos[1] + 1] <= "9"
                )
            ):
                continue
            drange.append([sent_idx, *span_pos])
    return drange


def reorganise_sents(sents, max_seq_len, concat=False, final_cut=False, concat_str=" "):
    new_sents = []
    group = ""
    for sent in sents:
        if len(sent) + len(group) < max_seq_len:
            if concat:
                if len(group) > 1 and "\u4e00" <= group[-1] <= "\u9fa5":
                    group += concat_str + sent
                else:
                    group += sent
            else:
                new_sents.append(sent)
        else:
            if len(group) > 0:
                new_sents.append(group)
                group = ""
            if len(sent) > max_seq_len:
                if final_cut:
                    group = sent[:max_seq_len]
                else:
                    sent_splits = sent_seg(sent, punctuations={"，", "、"})
                    reorg_sent_splits = reorganise_sents(
                        sent_splits, max_seq_len, concat=True, final_cut=True
                    )
                    new_sents.extend(reorg_sent_splits)
            else:
                group = sent
    if len(group) > 0:
        new_sents.append(group)
    return [s.strip() for s in filter(lambda x: len(x) > 0, new_sents)]


def build(
    event_type2event_class,
    filepath,
    dump_filepath,
    max_seq_len=128,
    inference=False,
    add_trigger=False,
):
    not_valid = 0
    data = []
    for d in load_line_json_iterator(filepath):
        sents = sent_seg(d["text"], punctuations={"；"})
        sents = reorganise_sents(sents, max_seq_len, concat=True)
        # sents = d['map_sentences']
        # sentence length filtering
        sents = list(filter(lambda x: len(x) >= 5, sents))
        sents.insert(0, d["title"])
        # sents.insert(0, d['map_title'])
        ann_valid_mspans = []
        ann_valid_dranges = []
        ann_mspan2dranges = defaultdict(list)
        ann_mspan2guess_field = {}
        recguid_eventname_eventdict_list = []

        event_types = []
        if not inference:
            if "event_list" not in d or len(d["event_list"]) == 0:
                not_valid += 1
                continue

            for event_idx, ins in enumerate(d["event_list"]):
                event_types.append(ins["event_type"])

                roles = event_type2event_class[ins["event_type"]].FIELDS
                role2arg = {x: None for x in roles}
                # take trigger into consideration
                trigger = ins["trigger"]
                trigger_ocurr = get_span_drange(sents, trigger)

                if len(trigger_ocurr) <= 0:
                    continue
                if add_trigger:
                    role2arg["Trigger"] = trigger
                    ann_mspan2guess_field[trigger] = "Trigger"
                    ann_valid_mspans.append(trigger)
                    ann_mspan2dranges[trigger] = trigger_ocurr
                for arg_pair in ins["arguments"]:
                    ocurr = get_span_drange(sents, arg_pair["argument"])
                    if len(ocurr) <= 0:
                        continue
                    role2arg[arg_pair["role"]] = arg_pair["argument"]
                    ann_valid_mspans.append(arg_pair["argument"])
                    ann_mspan2guess_field[arg_pair["argument"]] = arg_pair["role"]
                    ann_mspan2dranges[arg_pair["argument"]] = ocurr
                ann_valid_dranges = list(ann_mspan2dranges.values())
                recguid_eventname_eventdict_list.append(
                    [event_idx, ins["event_type"], role2arg]
                )

        doc_type = "unk"
        if len(event_types) > 0:
            et_counter = Counter(event_types).most_common()
            if len(et_counter) == 1 and et_counter[0][1] == 1:
                doc_type = "o2o"
            elif len(et_counter) == 1 and et_counter[0][1] > 1:
                doc_type = "o2m"
            elif len(et_counter) > 1:
                doc_type = "m2m"

        data.append(
            [
                d["id"],
                {
                    "doc_type": doc_type,
                    "sentences": sents,
                    "ann_valid_mspans": ann_valid_mspans,
                    "ann_valid_dranges": ann_valid_dranges,
                    "ann_mspan2dranges": dict(ann_mspan2dranges),
                    "ann_mspan2guess_field": ann_mspan2guess_field,
                    "recguid_eventname_eventdict_list": recguid_eventname_eventdict_list,
                },
            ]
        )
    print("not valid:", not_valid)
    with open(dump_filepath, "wt", encoding="utf-8") as fout:
        json.dump(data, fout, ensure_ascii=False)


def build_m2m(
    event_type2event_class,
    filepath,
    dump_filepath,
    max_seq_len=128,
    inference=False,
    add_trigger=False,
):
    not_valid = 0
    data = []
    for d in load_line_json_iterator(filepath):
        sents = sent_seg(d["text"], punctuations={"；"})
        sents = reorganise_sents(sents, max_seq_len, concat=True)
        # sents = d['map_sentences']
        # sentence length filtering
        sents = list(filter(lambda x: len(x) >= 5, sents))
        sents.insert(0, d["title"])
        # sents.insert(0, d['map_title'])
        ann_valid_mspans = []
        ann_valid_dranges = []
        ann_mspan2dranges = defaultdict(list)
        ann_mspan2guess_field = {}
        recguid_eventname_eventdict_list = []

        event_types = []
        if not inference:
            if "event_list" not in d or len(d["event_list"]) == 0:
                not_valid += 1
                continue

            for event_idx, ins in enumerate(d["event_list"]):
                event_types.append(ins["event_type"])

                roles = event_type2event_class[ins["event_type"]].FIELDS
                role2arg = {x: [] for x in roles}
                # take trigger into consideration
                trigger = ins["trigger"]
                trigger_ocurr = get_span_drange(sents, trigger)

                if len(trigger_ocurr) <= 0:
                    continue
                if add_trigger:
                    role2arg["Trigger"].append(trigger)
                    ann_mspan2guess_field[trigger] = "Trigger"
                    ann_valid_mspans.append(trigger)
                    ann_mspan2dranges[trigger] = trigger_ocurr

                for arg_pair in ins["arguments"]:
                    ocurr = get_span_drange(sents, arg_pair["argument"])
                    if len(ocurr) <= 0:
                        continue
                    role2arg[arg_pair["role"]].append(arg_pair["argument"])
                    ann_valid_mspans.append(arg_pair["argument"])
                    ann_mspan2guess_field[arg_pair["argument"]] = arg_pair["role"]
                    ann_mspan2dranges[arg_pair["argument"]] = ocurr
                ann_valid_dranges = list(ann_mspan2dranges.values())
                new_role2arg = {x: None for x in roles}
                for role, args in role2arg.items():
                    if len(args) <= 0:
                        new_role2arg[role] = None
                    else:
                        new_role2arg[role] = args

                recguid_eventname_eventdict_list.append(
                    [event_idx, ins["event_type"], new_role2arg]
                )

        doc_type = "unk"
        if len(event_types) > 0:
            et_counter = Counter(event_types).most_common()
            if len(et_counter) == 1 and et_counter[0][1] == 1:
                doc_type = "o2o"
            elif len(et_counter) == 1 and et_counter[0][1] > 1:
                doc_type = "o2m"
            elif len(et_counter) > 1:
                doc_type = "m2m"

        data.append(
            [
                d["id"],
                {
                    "doc_type": doc_type,
                    "sentences": sents,
                    "ann_valid_mspans": ann_valid_mspans,
                    "ann_valid_dranges": ann_valid_dranges,
                    "ann_mspan2dranges": dict(ann_mspan2dranges),
                    "ann_mspan2guess_field": ann_mspan2guess_field,
                    "recguid_eventname_eventdict_list": recguid_eventname_eventdict_list,
                },
            ]
        )
    print("not valid:", not_valid)
    with open(dump_filepath, "wt", encoding="utf-8") as fout:
        json.dump(data, fout, ensure_ascii=False)


def stat_roles(filepath):
    type2roles = defaultdict(set)
    for d in load_line_json_iterator(filepath):
        if "event_list" not in d:
            continue
        for event_idx, ins in enumerate(d["event_list"]):
            for arg_pair in ins["arguments"]:
                type2roles[ins["event_type"]].add(arg_pair["role"])

    for event_type in type2roles:
        print(event_type, len(type2roles[event_type]), list(type2roles[event_type]))


def merge_pred_ents_to_inference(pred_filepath, inference_filepath, dump_filepath):
    inference_data = load_json(inference_filepath)
    pred_data = {}
    pred_sents = {}
    pred_titles = {}
    for pred in load_line_json_iterator(pred_filepath):
        pred_data[pred["id"]] = pred["entity_pred"]
        pred_sents[pred["id"]] = pred["map_sentences"]
        pred_titles[pred["id"]] = pred["map_title"]
    for d in inference_data:
        guid = d[0]
        d[1]["sentences"] = pred_sents[guid]
        d[1]["sentences"].insert(0, pred_titles[guid])
        epd = pred_data[guid]
        ann_valid_mspans = []
        ann_valid_dranges = []
        ann_mspan2guess_field = {}
        ann_mspan2dranges = defaultdict(list)
        for ent in epd:
            if "trigger" in ent[1].lower():
                # ent_type = 'Trigger'
                continue
            else:
                ent_type = ent[1].split("-")[-1]
            ann_mspan2guess_field[ent[0]] = ent_type
            ann_mspan2dranges[ent[0]].append([ent[2] + 1, ent[3], ent[4] + 1])
        # for ent, ent_type in ent_pairs:
        #     drange = get_span_drange(d[1]['sentences'], ent)
        #     if len(drange) == 0:
        #         continue
        #     ann_mspan2guess_field[ent] = ent_type
        #     ann_mspan2dranges[ent] = drange
        ann_mspan2dranges = dict(ann_mspan2dranges)
        ann_valid_mspans = list(ann_mspan2dranges.keys())
        ann_valid_dranges = list(ann_mspan2dranges.values())
        d[1]["ann_valid_mspans"] = ann_valid_mspans
        d[1]["ann_valid_dranges"] = ann_valid_dranges
        d[1]["ann_mspan2guess_field"] = ann_mspan2guess_field
        d[1]["ann_mspan2dranges"] = ann_mspan2dranges

    with open(dump_filepath, "wt", encoding="utf-8") as fout:
        json.dump(inference_data, fout, ensure_ascii=False)

    print(json.dumps(inference_data[:2], ensure_ascii=False, indent=2))


def merge_pred_ents_with_pred_format_to_inference(
    pred_filepath, inference_filepath, dump_filepath
):
    inference_data = load_json(inference_filepath)
    pred_data = {}
    for pred in load_line_json_iterator(pred_filepath):
        pred_data[pred["id"]] = pred["new_comments"]
    for d in inference_data:
        guid = d[0]
        d[1]["sentences"] = pred_data[guid]["sentences"]
        ann_valid_mspans = []
        ann_valid_dranges = []
        ann_mspan2guess_field = {}
        ann_mspan2dranges = defaultdict(list)
        for ent in pred_data[guid]["mspans"]:
            if "trigger" in ent["mtype"].lower():
                # ent_type = 'Trigger'
                continue
            else:
                ent_type = ent["mtype"].split("-")[-1]
            ann_mspan2guess_field[ent["msapn"]] = ent_type
            ann_mspan2dranges[ent["msapn"]].append(ent["drange"])
        ann_mspan2dranges = dict(ann_mspan2dranges)
        ann_valid_mspans = list(ann_mspan2dranges.keys())
        ann_valid_dranges = list(ann_mspan2dranges.values())
        d[1]["ann_valid_mspans"] = ann_valid_mspans
        d[1]["ann_valid_dranges"] = ann_valid_dranges
        d[1]["ann_mspan2guess_field"] = ann_mspan2guess_field
        d[1]["ann_mspan2dranges"] = ann_mspan2dranges

    with open(dump_filepath, "wt", encoding="utf-8") as fout:
        json.dump(inference_data, fout, ensure_ascii=False)

    print(json.dumps(inference_data[:2], ensure_ascii=False, indent=2))


def multi_role_stat(filepath):
    num_ins = 0
    num_multi_role_doc = 0
    type2num_multi_role = defaultdict(lambda: 0)
    type2role2num_multi_role = defaultdict(lambda: defaultdict(list))

    for d in load_line_json_iterator(filepath):
        if "event_list" not in d:
            continue
        for ins in d["event_list"]:
            num_ins += 1
            roles = [x["role"] for x in ins["arguments"]]
            role, role_cnt = Counter(roles).most_common(1)[0]
            if role_cnt > 1:
                # if ins['event_type'] == '高管变动' and role == '高管职位':
                #     breakpoint()
                num_multi_role_doc += 1
                type2num_multi_role[ins["event_type"]] += 1
                type2role2num_multi_role[ins["event_type"]][role].append(role_cnt)

    print("num_ins", num_ins)
    print("num_multi_role_doc", num_multi_role_doc)
    print("type2num_multi_role", type2num_multi_role)
    for event_type in type2role2num_multi_role:
        for role in type2role2num_multi_role[event_type]:
            # type2role2num_multi_role[event_type][role] = Counter(type2role2num_multi_role[event_type][role]).most_common()
            type2role2num_multi_role[event_type][role] = sum(
                type2role2num_multi_role[event_type][role]
            )
    print("type2role2num_multi_role", type2role2num_multi_role)


def stat_shared_triggers(filepath):
    # train: 3400 / 9498
    num_records = 0
    num_share_trigger_records = 0
    with open(filepath, "rt", encoding="utf-8") as fin:
        for line in fin:
            trigger2event = defaultdict(list)
            data = json.loads(line)
            for ins in data.get("event_list", []):
                num_records += 1
                trigger2event[ins["trigger"]].append(ins)
            for trigger, inses in trigger2event.items():
                if len(inses) > 1:
                    num_share_trigger_records += len(inses)
    print(
        f"num_records: {num_records}, num_share_trigger_records: {num_share_trigger_records}"
    )


if __name__ == "__main__":
    # stat_sent_len('train.json')
    # stat_roles('train.json')
    # stat_shared_triggers('train.json')

    template = get_event_template("dueefin_wo_tgg")

    build(
        template.event_type2event_class,
        "duee_fin_train.json",
        "dueefin_train_wo_tgg.json",
    )
    build(
        template.event_type2event_class, "duee_fin_dev.json", "dueefin_dev_wo_tgg.json"
    )
    build(
        template.event_type2event_class,
        "duee_fin_sample.json",
        "dueefin_sample_wo_tgg.json",
    )
    build(
        template.event_type2event_class,
        "duee_fin_test2.json",
        "dueefin_submit_wo_tgg.json",
        inference=True,
    )

    template = get_event_template("dueefin_w_tgg")

    build(
        template.event_type2event_class,
        "duee_fin_train.json",
        "dueefin_train_w_tgg.json",
        add_trigger=True,
    )
    build(
        template.event_type2event_class,
        "duee_fin_dev.json",
        "dueefin_dev_w_tgg.json",
        add_trigger=True,
    )
    build(
        template.event_type2event_class,
        "duee_fin_sample.json",
        "dueefin_sample_w_tgg.json",
        add_trigger=True,
    )
    build(
        template.event_type2event_class,
        "duee_fin_test2.json",
        "dueefin_submit_w_tgg.json",
        inference=True,
        add_trigger=True,
    )
