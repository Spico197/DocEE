import json
from collections import Counter, defaultdict
from itertools import combinations
from typing import DefaultDict, List, Optional

from dee.event_types import get_event_template

template = get_event_template("dueefin_wo_tgg")
event_type_fields_list = template.event_type_fields_list
event_type2event_class = template.event_type2event_class


def load_line_json(filepath):
    data = []
    with open(filepath, "rt", encoding="utf-8") as fin:
        for line in fin:
            data.append(json.loads(line))
    return data


def create_doc_id2IPO_index(data):
    id2data = defaultdict(list)
    for d in data:
        id2data[d["id"]].append(d["pred"]["label"])
    id2result = {}
    for doc_id, labels in id2data.items():
        id2result[doc_id] = Counter(labels).most_common(1)[0][0]
    return id2result


def create_field2class_index(et_fields_list):
    field2class = defaultdict(set)
    for ec in et_fields_list:
        type_name = ec[0]
        fields = ec[1]
        for f in fields:
            field2class[f].add(type_name)
    return field2class


def reveal_instances_from_guessing(
    pred_types: set,
    mspans: List[dict],
    field2class: DefaultDict[str, set],
    guess_strategy: Optional[str] = "&",
):
    event_list = []
    if len(pred_types) <= 0:
        # needs type guessing
        guessed_types = set()
        for pair in mspans:
            if "Trigger" not in pair["mtype"]:
                mspan = pair["mspan"]
                mtype = pair["mtype"]
                g_types = field2class[mtype]
                if len(guessed_types) <= 0:
                    guessed_types.update(g_types)
                else:
                    if guess_strategy == "&":
                        guessed_types.intersection_update(g_types)
                    elif guess_strategy == "|":
                        guessed_types.update(g_types)
                    else:
                        raise ValueError(
                            f"guess_strategy: {guess_strategy} not supported"
                        )
        pred_types = guessed_types
    # need finding
    for pred_type in pred_types:
        args = []
        for pair in mspans:
            if "Trigger" not in pair["mtype"]:
                mspan = pair["mspan"]
                mtype = pair["mtype"]
                if mtype in event_type2event_class[pred_type].FIELDS:
                    args.append({"role": mtype, "argument": mspan})
        if len(args) > 0:
            event_list.append({"event_type": pred_type, "arguments": args})
    return event_list


if __name__ == "__main__":
    add_guessing_instance = False
    guess_strategy = "&"
    add_additional_arg = False
    add_IPO_stage = False
    # 合并可合并事件
    merge_part_of_event = False

    if add_IPO_stage:
        # IPO_stage_result_filepath = "InferenceResults/test1-IPO_pred-large.json"
        IPO_stage_result_filepath = "InferenceResults/test2_IPO_pred-large.json"
        doc_id2IPO_stage_result = create_doc_id2IPO_index(
            load_line_json(IPO_stage_result_filepath)
        )

    remove_comments = True
    remove_triggers = True

    """test2"""
    to_remove_filepath = "dueefin_PTPCG_woTgg_midImpt.json"
    save_filepath = "ppsed_dueefin_PTPCG_woTgg_midImpt.json"

    to_remove = load_line_json(to_remove_filepath)
    field2class = create_field2class_index(event_type_fields_list)

    mergable = 0
    with open(save_filepath, "wt", encoding="utf-8") as fout:
        for d in to_remove:
            event_list = []
            if len(d["event_list"]) <= 0:
                if add_guessing_instance:
                    pred_types = set(d["comments"]["pred_types"])
                    event_list = reveal_instances_from_guessing(
                        pred_types,
                        d["comments"]["mspans"],
                        field2class,
                        guess_strategy=guess_strategy,
                    )
                else:
                    pass
            else:
                for ins in d["event_list"]:
                    arguments = []
                    for arg_pair in ins["arguments"]:
                        if arg_pair["role"] == "Trigger" and remove_triggers:
                            continue
                        arguments.append(arg_pair)
                    if add_additional_arg:
                        exist_roles = {x["role"] for x in arguments}
                        for pair in d["comments"]["mspans"]:
                            tmp_arg = {"role": pair["mtype"], "argument": pair["mspan"]}
                            if (
                                pair["mtype"] not in exist_roles
                                and tmp_arg not in arguments
                                and pair["mtype"]
                                in event_type2event_class[ins["event_type"]].FIELDS
                            ):
                                exist_roles.add(pair["mtype"])
                                arguments.append(tmp_arg)
                    ins["arguments"] = arguments
                    if (
                        add_IPO_stage
                        and ins["event_type"] == "公司上市"
                        and all(x["role"] != "环节" for x in ins["arguments"])
                        and d["id"] in doc_id2IPO_stage_result
                    ):
                        ins["arguments"].append(
                            {"role": "环节", "argument": doc_id2IPO_stage_result[d["id"]]}
                        )
                    event_list.append(ins)

            if merge_part_of_event:
                event_list_merge_flag = [True for _ in range(len(event_list))]
                for ins1, ins2 in combinations(enumerate(event_list), 2):
                    if ins1[1]["event_type"] == ins2[1]["event_type"]:
                        ins1_args = {
                            (arg["role"], arg["argument"])
                            for arg in ins1[1]["arguments"]
                        }
                        ins2_args = {
                            (arg["role"], arg["argument"])
                            for arg in ins2[1]["arguments"]
                        }
                        if ins1_args == ins2_args or ins2_args.issubset(ins1_args):
                            event_list_merge_flag[ins2[0]] = False
                        elif ins1_args.issubset(ins2_args):
                            event_list_merge_flag[ins1[0]] = False
                new_event_list = []
                for flag, events in zip(event_list_merge_flag, event_list):
                    if flag:
                        new_event_list.append(events)
                    else:
                        mergable += 1
            else:
                new_event_list = event_list
            d["event_list"] = new_event_list
            if remove_comments and "comments" in d:
                del d["comments"]
            fout.write(f"{json.dumps(d, ensure_ascii=False)}\n")
    print("可合并的事件实例数：", mergable)
