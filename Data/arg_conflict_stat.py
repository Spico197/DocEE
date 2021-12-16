import json
from collections import defaultdict


def load_json(filepath):
    with open(filepath, "rt", encoding="utf-8") as fin:
        return json.load(fin)


def stat_seperated_type(data):
    """trying to figure out the conflicts"""
    stat_results = defaultdict(lambda: defaultdict(lambda: 0))  # type -> role -> number
    total_records = defaultdict(
        lambda: defaultdict(lambda: 0)
    )  # type -> role -> number
    all_total_records = defaultdict(
        lambda: defaultdict(lambda: 0)
    )  # type -> role -> number
    for d in data:
        tmp_results = defaultdict(lambda: defaultdict(lambda: list()))
        for event in d[1]["recguid_eventname_eventdict_list"]:
            _, event_type, role2arg = event
            for role, arg in role2arg.items():
                all_total_records[event_type][role] += 1
                if arg is not None:
                    tmp_results[event_type][role].append(arg)
                    total_records[event_type][role] += 1
        for event_type, role2args in tmp_results.items():
            for role, args in role2args.items():
                if len(args) != 0:
                    if len(set(args)) / len(args) == 1.0:
                        stat_results[event_type][role] += 1
    for event_type, role2count in stat_results.items():
        print("\n", event_type)
        for role, count in role2count.items():
            print(
                "\t",
                role,
                count,
                total_records[event_type][role],
                all_total_records[event_type][role],
            )


def stat_mixed_type(data):
    """trying to figure out the conflicts"""
    stat_results = defaultdict(lambda: defaultdict(lambda: 0))  # type -> role -> number
    total_records = defaultdict(
        lambda: defaultdict(lambda: 0)
    )  # type -> role -> number
    all_total_records = defaultdict(
        lambda: defaultdict(lambda: 0)
    )  # type -> role -> number
    for d in data:
        tmp_results = defaultdict(lambda: defaultdict(lambda: list()))
        for event in d[1]["recguid_eventname_eventdict_list"]:
            _, event_type, role2args = event
            for role, arg in role2args.items():
                all_total_records[event_type][role] += 1
                if arg is not None:
                    tmp_results[event_type][role].append(arg)
                    total_records[event_type][role] += 1
        for event_type, role2args in tmp_results.items():
            for role, args in role2args.items():
                if len(args) != 0:
                    if len(set(args)) / len(args) == 1.0:
                        stat_results[event_type][role] += 1
    for event_type, role2count in stat_results.items():
        print("\n", event_type)
        for role, count in role2count.items():
            print(
                "\t",
                role,
                count,
                total_records[event_type][role],
                all_total_records[event_type][role],
            )


if __name__ == "__main__":
    for data_name in ["train", "dev", "test"]:
        data = load_json(f"{data_name}.json")
        print(f"\n============= {data_name} =============")
        stat_seperated_type(data)
