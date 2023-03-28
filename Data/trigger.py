import json
import random
import sys
from collections import defaultdict
from itertools import combinations


def load_json(filepath):
    data = []
    with open(filepath, "rt", encoding="utf-8") as fin:
        data = json.load(fin)
    return data


def ent_in_other_instances(entity, all_entities, current_event_idx):
    in_flag = False
    for event_idx, ents in enumerate(all_entities):
        if event_idx == current_event_idx:
            continue
        if entity in ents:
            in_flag = True
            break
    return in_flag


def check_trigger(data, num_trigger_group=1, verbose=True):
    r"""
    check if there're some entities able to be triggers

    Args:
        data: data
        num_trigger_group: number of the trigger words in each event instance
            if > 1, it means not all entities can be `None` or be shared
    """
    # number of instances of each event
    type2num = defaultdict(lambda: 0)
    # for each type and each role, check `None` number
    not_existence = defaultdict(lambda: defaultdict(lambda: 0))
    # for each type and each role, check distinguishability
    not_distinguishability = defaultdict(lambda: defaultdict(lambda: 0))

    for _, d in data:
        instances = d["recguid_eventname_eventdict_list"]
        all_ents = [set(role2ent.values()) for _, _, role2ent in instances]
        for event_idx, (_, event_type, role2ent) in enumerate(instances):
            type2num[event_type] += 1
            for role_group in combinations(role2ent.keys(), num_trigger_group):
                role_group = tuple(sorted(role_group))
                if all(role2ent[role] is None for role in role_group):
                    not_existence[event_type][role_group] += 1
                else:
                    not_existence[event_type][role_group] += 0
                if all(
                    ent_in_other_instances(role2ent[role], all_ents, event_idx)
                    for role in role_group
                ):
                    not_distinguishability[event_type][role_group] += 1
                else:
                    not_distinguishability[event_type][role_group] += 0

    results = defaultdict(
        lambda: defaultdict(
            lambda: {"existence": 0.0, "distinguishability": 0.0, "overall": 0.0}
        )
    )
    for event_type in type2num:
        num_instance = type2num[event_type]
        for role in not_existence[event_type]:
            results[event_type][role]["existence"] = (
                num_instance - not_existence[event_type][role]
            ) / num_instance
            results[event_type][role]["distinguishability"] = (
                num_instance - not_distinguishability[event_type][role]
            ) / num_instance
            results[event_type][role]["overall"] = (
                results[event_type][role]["existence"]
                * results[event_type][role]["distinguishability"]
            )

    final_results = defaultdict(list)
    for event_type in results:
        tmp_results = []
        for role_group in results[event_type]:
            tmp_results.append(
                [
                    list(role_group),
                    results[event_type][role_group]["overall"],
                    results[event_type][role_group]["existence"],
                    results[event_type][role_group]["distinguishability"],
                ]
            )
        tmp_results.sort(key=lambda x: x[1], reverse=True)
        if verbose:
            print(f"{event_type}:\t{type2num[event_type]}")
            print("Overall\t\tExistence\tDistinguishability\tRoleGroup")
        for result in tmp_results:
            if verbose:
                print(
                    f"{100 * result[1]:3.3f}\t\t{100 * result[2]:3.3f}\t\t{100 * result[3]:3.3f}\t\t{result[0]}"
                )
            final_results[event_type].append(
                {
                    "overall": result[1],
                    "existence": result[2],
                    "distinguishability": result[3],
                    "role_group": result[0],
                }
            )
        if verbose:
            print()

    return final_results


def auto_select(
    data, strategy="high", max_trigger_num=9, verbose=True, with_trigger=False
):
    assert strategy in ["high", "mid", "low", "random"]
    trigger_group = defaultdict(dict)
    trigger_group_importance = defaultdict(dict)
    for trigger_num in range(1, max_trigger_num + 1):
        results = check_trigger(data, num_trigger_group=trigger_num, verbose=False)

        for event_type in results:
            if trigger_num == 1:
                trigger_all = [x["role_group"][0] for x in results[event_type]]
                trigger_group[event_type]["all"] = trigger_all

            if len(results[event_type]) < 1:
                continue

            if with_trigger:
                results[event_type] = list(
                    filter(lambda x: "Trigger" in x["role_group"], results[event_type])
                )

            selected_index = 0
            if strategy == "high":
                selected_index = 0
            elif strategy == "mid":
                selected_index = len(results[event_type]) // 2
            elif strategy == "low":
                selected_index = -1
            elif strategy == "random":
                selected_index = random.choice(list(range(len(results[event_type]))))
            trigger_group[event_type][trigger_num] = results[event_type][
                selected_index
            ]["role_group"]
            trigger_group_importance[event_type][trigger_num] = results[event_type][
                selected_index
            ]["overall"]

    if verbose:
        for event_type in trigger_group:
            print(f"{event_type} = {{")
            for trigger_num in range(1, len(trigger_group[event_type])):
                print(
                    f"\t{trigger_num}: {trigger_group[event_type][trigger_num]},  # importance: {trigger_group_importance[event_type][trigger_num]}"
                )
            print(f"}}\nTRIGGERS['all'] = {trigger_group[event_type]['all']}")
            print()

    return trigger_group


if __name__ == "__main__":
    num_trigger_group = 9
    if len(sys.argv) > 1:
        num_trigger_group = int(sys.argv[1])

    # data = load_json("DuEEData/dueefin_dev_w_tgg.json")
    # auto_select(data, strategy='high', max_trigger_num=num_trigger_group, verbose=True, with_trigger=True)
    # data = load_json("DuEEData/dueefin_dev_wo_tgg.json")
    # data = load_json("RAMS/typed_train_tgFalse_lv1.json")
    # data = load_json("typed_test.json")
    tot_data = []
    for dname in ["train", "dev", "test"]:
        # tot_data += load_json(f"Data/WikiEvents/{dname}.post.wTgg.json")
        tot_data += load_json(f"Data/CCKS2020/{dname}.post.json")
    # check_trigger(data, num_trigger_group=num_trigger_group)

    auto_select(
        tot_data,
        strategy="high",
        max_trigger_num=num_trigger_group,
        verbose=True,
        with_trigger=True,
    )
