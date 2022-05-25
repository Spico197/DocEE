import json
import os
import statistics
import sys
from collections import defaultdict

from dee.helper import (
    aggregate_task_eval_info,
    print_single_vs_multi_performance,
    print_total_eval_info,
)
from dee.tasks.dee_task import DEETask, DEETaskSetting

MEASURE_TYPES = [
    "classification",
    "entity",
    "combination",
    "rawCombination",
    "overall",
    "instance",
]


def load_evaluate_results(task_name, model, data_type, dataset, span_type, epoch):
    with open(
        f"Exps/{task_name}/Output/dee_eval.{dataset}.{span_type}.{model}.{epoch}.json",
        "rt",
        encoding="utf-8",
    ) as fin:
        results = json.load(fin)
    return results[data_type]


def print_specified_epoch(
    task_name, model, epoch, dataset="test", span_type="pred_span"
):
    print_data = []
    print(
        f"task_name={task_name}, model={model}, dataset={dataset}, span_type={span_type}, epoch={epoch}"
    )
    result = load_evaluate_results(
        task_name, model, "overall", dataset, span_type, epoch
    )
    header = "Data\t{}".format("\t".join(list(map(lambda x: x.title(), result.keys()))))
    print(header)
    for data_type in ["o2o", "o2m", "m2m", "overall"]:
        result = load_evaluate_results(
            task_name, model, data_type, dataset, span_type, epoch
        )
        tmp_print = [data_type]
        for measure_type in MEASURE_TYPES:
            if measure_type in result:
                tmp_print.append(result[measure_type]["MicroF1"])
        print_data.append(tmp_print)
    for ds in print_data:
        for d in ds:
            if isinstance(d, float):
                print("{:.3f}".format(d * 100), end="\t")
            else:
                print("{}".format(d), end="\t")
        print()


def print_detailed_specified_epoch(
    task_name, model, epoch, dataset="test", span_type="pred_span"
):
    print_data = []
    results = {
        "ModelType": model,
        "o2o": {
            "classification": {"precision": None, "recall": None, "f1": None},
            "entity": {"precision": None, "recall": None, "f1": None},
            "combination": {"precision": None, "recall": None, "f1": None},
            "rawCombination": {"precision": None, "recall": None, "f1": None},
            "overall": {"precision": None, "recall": None, "f1": None},
            "instance": {"precision": None, "recall": None, "f1": None},
        },
        "o2m": {
            "classification": {"precision": None, "recall": None, "f1": None},
            "entity": {"precision": None, "recall": None, "f1": None},
            "combination": {"precision": None, "recall": None, "f1": None},
            "rawCombination": {"precision": None, "recall": None, "f1": None},
            "overall": {"precision": None, "recall": None, "f1": None},
            "instance": {"precision": None, "recall": None, "f1": None},
        },
        "m2m": {
            "classification": {"precision": None, "recall": None, "f1": None},
            "entity": {"precision": None, "recall": None, "f1": None},
            "combination": {"precision": None, "recall": None, "f1": None},
            "rawCombination": {"precision": None, "recall": None, "f1": None},
            "overall": {"precision": None, "recall": None, "f1": None},
            "instance": {"precision": None, "recall": None, "f1": None},
        },
        "overall": {
            "classification": {"precision": None, "recall": None, "f1": None},
            "entity": {"precision": None, "recall": None, "f1": None},
            "combination": {"precision": None, "recall": None, "f1": None},
            "rawCombination": {"precision": None, "recall": None, "f1": None},
            "overall": {"precision": None, "recall": None, "f1": None},
            "instance": {"precision": None, "recall": None, "f1": None},
        },
    }
    print(
        f"task_name={task_name}, model={model}, dataset={dataset}, span_type={span_type}, epoch={epoch}"
    )
    result4header = load_evaluate_results(
        task_name, model, "overall", dataset, span_type, epoch
    )
    headers = []
    for measure_type in MEASURE_TYPES:
        if measure_type in result4header.keys():
            headers.append(measure_type)
    header = "Data\t{}".format(
        "\t".join(list(map(lambda x: "{:20}".format(x.title()), headers)))
    )
    print(header)
    print("    \t{}".format("Prec\tRecall\tF1\t" * len(headers)))
    for data_type in ["o2o", "o2m", "m2m", "overall"]:
        result = load_evaluate_results(
            task_name, model, data_type, dataset, span_type, epoch
        )
        tmp_print = [data_type]
        for measure_type in MEASURE_TYPES:
            if measure_type in result:
                tmp_print.extend(
                    [
                        result[measure_type]["MicroPrecision"],
                        result[measure_type]["MicroRecall"],
                        result[measure_type]["MicroF1"],
                    ]
                )
                results[data_type][measure_type]["precision"] = "{:.3f}".format(
                    result[measure_type]["MicroPrecision"] * 100
                )
                results[data_type][measure_type]["recall"] = "{:.3f}".format(
                    result[measure_type]["MicroRecall"] * 100
                )
                results[data_type][measure_type]["f1"] = "{:.3f}".format(
                    result[measure_type]["MicroF1"] * 100
                )
        print_data.append(tmp_print)

    for ds in print_data:
        for d in ds:
            if isinstance(d, float):
                print("{:.3f}".format(d * 100), end="\t")
            else:
                print("{}".format(d), end="\t")
        print()

    return results


def print_detailed_json(model, data):
    print_data = []
    results = {
        "ModelType": model,
        "o2o": {
            "classification": {"precision": None, "recall": None, "f1": None},
            "entity": {"precision": None, "recall": None, "f1": None},
            "combination": {"precision": None, "recall": None, "f1": None},
            "rawCombination": {"precision": None, "recall": None, "f1": None},
            "overall": {"precision": None, "recall": None, "f1": None},
            "instance": {"precision": None, "recall": None, "f1": None},
        },
        "o2m": {
            "classification": {"precision": None, "recall": None, "f1": None},
            "entity": {"precision": None, "recall": None, "f1": None},
            "combination": {"precision": None, "recall": None, "f1": None},
            "rawCombination": {"precision": None, "recall": None, "f1": None},
            "overall": {"precision": None, "recall": None, "f1": None},
            "instance": {"precision": None, "recall": None, "f1": None},
        },
        "m2m": {
            "classification": {"precision": None, "recall": None, "f1": None},
            "entity": {"precision": None, "recall": None, "f1": None},
            "combination": {"precision": None, "recall": None, "f1": None},
            "rawCombination": {"precision": None, "recall": None, "f1": None},
            "overall": {"precision": None, "recall": None, "f1": None},
            "instance": {"precision": None, "recall": None, "f1": None},
        },
        "overall": {
            "classification": {"precision": None, "recall": None, "f1": None},
            "entity": {"precision": None, "recall": None, "f1": None},
            "combination": {"precision": None, "recall": None, "f1": None},
            "rawCombination": {"precision": None, "recall": None, "f1": None},
            "overall": {"precision": None, "recall": None, "f1": None},
            "instance": {"precision": None, "recall": None, "f1": None},
        },
    }

    headers = []
    for measure_type in MEASURE_TYPES:
        if measure_type in data["overall"].keys():
            headers.append(measure_type)
    header = "Data\t{}".format(
        "\t".join(list(map(lambda x: "{:20}".format(x.title()), headers)))
    )
    print(header)
    print("    \t{}".format("Prec\tRecall\tF1\t" * len(headers)))
    for data_type in ["o2o", "o2m", "m2m", "overall"]:
        result = data[data_type]
        tmp_print = [data_type]
        for measure_type in MEASURE_TYPES:
            if measure_type in result:
                tmp_print.extend(
                    [
                        result[measure_type]["MicroPrecision"],
                        result[measure_type]["MicroRecall"],
                        result[measure_type]["MicroF1"],
                    ]
                )
                results[data_type][measure_type]["precision"] = "{:.3f}".format(
                    result[measure_type]["MicroPrecision"] * 100
                )
                results[data_type][measure_type]["recall"] = "{:.3f}".format(
                    result[measure_type]["MicroRecall"] * 100
                )
                results[data_type][measure_type]["f1"] = "{:.3f}".format(
                    result[measure_type]["MicroF1"] * 100
                )
        print_data.append(tmp_print)

    for ds in print_data:
        for d in ds:
            if isinstance(d, float):
                print("{:.3f}".format(d * 100), end="\t")
            else:
                print("{}".format(d), end="\t")
        print()

    return results


def print_score_on_each_epoch(
    task_name,
    model,
    max_epoch,
    span_type="pred_span",
    data_type="overall",
    measure_type="overall",
    measure_key="MicroF1",
):
    print(
        f"task_name={task_name}, model={model}, max_epoch={max_epoch}, data_type={data_type}, span_type={span_type}"
    )
    print(f"Epoch\t{measure_key}")
    print("\tdev\ttest")
    for epoch in range(1, max_epoch + 1):
        dev_result = load_evaluate_results(
            task_name, model, data_type, "dev", span_type, epoch
        )
        test_result = load_evaluate_results(
            task_name, model, data_type, "test", span_type, epoch
        )
        print(
            "{}\t{:.3f}\t{:.3f}".format(
                epoch,
                dev_result[measure_type][measure_key] * 100,
                test_result[measure_type][measure_key] * 100,
            )
        )


def get_macro_scores(event_results):
    macros = {"precision": [], "recall": [], "f1": []}
    for main, roles in event_results:
        macros["precision"].append(main["MicroPrecision"])
        macros["recall"].append(main["MicroRecall"])
        macros["f1"].append(main["MicroF1"])

    for name, values in macros.items():
        macros[name] = statistics.mean(values)

    return macros


def get_macro_overall(
    task_name,
    model,
    max_epoch,
    span_type="pred_span",
    data_type="overall",
    verbose=False,
):
    results = {"dev": [], "test": []}

    print(
        f"task_name={task_name}, model={model}, max_epoch={max_epoch}, data_type={data_type}, span_type={span_type}"
    )
    if verbose:
        print("Epoch\tDev  \tTest")

    best_epoch = -1
    best_f1 = -1
    for epoch in range(1, max_epoch + 1):
        dev_result = load_evaluate_results(
            task_name, model, data_type, "dev", span_type, epoch
        )
        dev_macro = get_macro_scores(dev_result[data_type]["Events"])
        if dev_macro["f1"] > best_f1:
            best_epoch = epoch
            best_f1 = dev_macro["f1"]
        results["dev"].append(dev_macro)
        test_result = load_evaluate_results(
            task_name, model, data_type, "test", span_type, epoch
        )
        test_macro = get_macro_scores(test_result[data_type]["Events"])
        results["test"].append(test_macro)

        if verbose:
            print(
                "{}\t{:.3f}\t{:.3f}".format(
                    epoch, dev_macro["f1"] * 100, test_macro["f1"] * 100
                )
            )

    print(
        f"best epoch on macro scores: {best_epoch}, DEV scores: {results['dev'][best_epoch - 1]}, TEST scores: {results['test'][best_epoch - 1]}"
    )


def print_best_epoch_result(
    task_name,
    model,
    max_epoch,
    dataset="test",
    span_type="pred_span",
    data_type="overall",
    measure_type="overall",
    measure_key="MicroF1",
):
    print("WARNING: deprecated, please be aware of what you are doing!")
    print(
        f"task_name={task_name}, model={model}, max_epoch={max_epoch}, dataset={dataset}, data_type={data_type}, span_type={span_type}"
    )
    all_results = []
    range_start = 2 if "soft_th_cg" in task_name else 1
    for epoch in range(range_start, max_epoch + 1):
        result = load_evaluate_results(
            task_name, model, data_type, dataset, span_type, epoch
        )
        all_results.append((epoch, result[measure_type][measure_key]))
    all_results.sort(key=lambda x: x[1])
    best_epoch, best_result = all_results[-1]
    print("best_epoch={}, best_result={:.3f}".format(best_epoch, best_result * 100))


def get_best_dev(
    task_name,
    model,
    max_epoch,
    span_type="pred_span",
    data_type="overall",
    measure_type="overall",
    measure_key="MicroF1",
):
    all_results = []
    range_start = 2 if "soft_th_cg" in task_name else 1
    for epoch in range(range_start, max_epoch + 1):
        result = load_evaluate_results(
            task_name, model, data_type, "dev", span_type, epoch
        )
        all_results.append((epoch, result[measure_type][measure_key]))
    all_results.sort(key=lambda x: x[1])
    best_epoch, best_result = all_results[-1]
    return best_epoch, best_result


def print_best_test_via_dev(
    task_name,
    model,
    max_epoch,
    span_type="pred_span",
    data_type="overall",
    measure_type="overall",
    measure_key="MicroF1",
):
    print(
        f"task_name={task_name}, model={model}, max_epoch={max_epoch}, data_type={data_type}, span_type={span_type}"
    )
    best_epoch, best_dev_result = get_best_dev(
        task_name,
        model,
        max_epoch,
        span_type=span_type,
        data_type=data_type,
        measure_type=measure_type,
        measure_key=measure_key,
    )
    test_result = load_evaluate_results(
        task_name, model, data_type, "test", span_type, best_epoch
    )
    print(
        "dev best_epoch={}, best_dev_result={:.3f}, best_test_result={:.3f}".format(
            best_epoch,
            best_dev_result * 100,
            test_result[measure_type][measure_key] * 100,
        )
    )
    return best_epoch


def get_msg_result(
    task_name,
    model,
    max_epoch,
    span_type="pred_span",
    data_type="overall",
    measure_type="overall",
    measure_key="MicroF1",
):
    all_results = []
    range_start = 2 if "soft_th_cg" in task_name else 1
    for epoch in range(range_start, max_epoch + 1):
        result = load_evaluate_results(
            task_name, model, data_type, "dev", span_type, epoch
        )
        all_results.append((epoch, result[measure_type][measure_key]))
    all_results.sort(key=lambda x: x[1])
    best_epoch, best_result = all_results[-1]
    test_result = load_evaluate_results(
        task_name, model, data_type, "test", span_type, best_epoch
    )
    msg = []
    for title in MEASURE_TYPES:
        if title in test_result:
            msg.append(
                "{}: {:.3f}".format(
                    title.capitalize(), test_result[title][measure_key] * 100
                )
            )
    return ", ".join(msg)


def print_tp_fp_fn(
    task_name,
    model,
    epoch,
    dataset="test",
    measure_type="overall",
    span_type="pred_span",
):
    print_data = []
    print(
        f"task_name={task_name}, model={model}, dataset={dataset}, span_type={span_type}, epoch={epoch}"
    )
    print("    \tTP\tFP\tFN")
    for data_type in ["o2o", "o2m", "m2m", "overall"]:
        result = load_evaluate_results(
            task_name, model, data_type, dataset, span_type, epoch
        )
        print_data.append(
            [
                data_type,
                result[measure_type]["TP"],
                result[measure_type]["FP"],
                result[measure_type]["FN"],
            ]
        )
    for ds in print_data:
        for d in ds:
            if isinstance(d, float):
                print("{:d}".format(d), end="\t")
            else:
                print("{}".format(d), end="\t")
        print()


def print_paper_result(task_name, result_type="total"):
    """
    Get the results reported in the Doc2EDAG paper.

    Args:
        task_name: task name
        result_type: `total` or `s&m`
    """
    log_path = os.path.join("Logs", f"{task_name}.log")
    info = []
    sm_pos = -1
    info_start_flag = False
    with open(log_path, "rt", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if "task_name=" in line:
                break

            if info_start_flag and "INFO - dee.utils" not in line:
                if "=============== Single vs. Multi" in line:
                    sm_pos = len(info)
                info.append(line)

            if "--------------- doc_type:" in line:
                info.append(line)
                info_start_flag = True

    print(f"task_name = {task_name}")
    print(info[0])
    if result_type == "total":
        print("\n".join(info[1:sm_pos]))
    elif result_type == "s&m":
        print("\n".join(info[sm_pos:]))
    return info


if __name__ == "__main__":
    total_epoch = 100
    if len(sys.argv) > 1:
        total_epoch = int(sys.argv[1])

    i = 1
    type2results = defaultdict(list)
    records = [
        # ("debug", "Doc2EDAG"),
        # ("doc2edag_2cards", "GreedyDec"),
        # ("doc2edag_2cards", "Doc2EDAG"),
        # ("lstmmtl", "LSTMMTL"),
        # ('lstmmtl2cg-tuned', 'LSTMMTL2CompleteGraphModel'),    # 2021年 03月 17日 星期三 12:20:37 CST
        # ("trans2cg-biaffine_512_dim-sche_samp-lr1e-4-bs64_32", "Trans2CompleteGraphModel"),
        # ("trans2cg-biaffine_512_dim-sche_samp-lr1e-3-bs64_32", "Trans2CompleteGraphModel"),
        # ("lstmmtl2cg", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-tuned-bk_upper", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-tuned-bf_upper", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-tuned-o2o", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-tuned-o2o-drop_irr_ents", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-tuned-o2m", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-tuned-o2m-drop_irr_ents", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-tuned-o2m-hs512", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-tuned-o2m-hs512-bhs256", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-tuned-o2m-hs256", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-tuned-o2m-hs256-hbs256", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-sche_samp", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2hard_th_cg-sche_samp", "LSTMMTL2HardThresholdCompleteGraph"),
        # ("lstmmtl2bhard_th_cg_0.7-sche_samp", "LSTMMTL2BHardThresholdCompleteGraph"),
        # ("lstmmtl2bhard_th_cg_0.5-sche_samp", "LSTMMTL2BHardThresholdCompleteGraph"),
        # ("lstmmtl2bhard_th_cg_0.3-sche_samp", "LSTMMTL2BHardThresholdCompleteGraph"),
        # ("lstmmtl2soft_th_cg-sche_samp", "LSTMMTL2SoftThresholdCompleteGraph"),
        # ("lstmmtl2soft_th_cg-use_span_lstm_2_layers-use_span_att_1_head-sche_samp_20_20-lr1e-3", "LSTMMTL2SoftThresholdCompleteGraph"),
        # ("lstmmtl2cg-drop_irr_ents", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm-shed_samp", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm_2_layers-shed_samp-lr1e-3", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm-use_span_att-shed_samp", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm_2_layers-use_span_att-shed_samp-lr1e-3", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm-use_span_att_1_head-shed_samp", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm-use_span_att-shed_samp_20_20", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm-use_span_att-shed_samp-lr1e-3", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm-use_span_att-shed_samp-bs64-lr1e-4", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm-use_span_att-biaffine_512_dim-shed_samp", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm_2_layers-use_span_att-shed_samp", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm-use_span_att-shed_samp", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-lstmner_2_layers-use_span_lstm-use_span_att-shed_samp", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-lstmner_2_layers-use_span_lstm_2_layers-use_span_att-shed_samp", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_att", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm_projection", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm-use_span_att", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_att-use_span_lstm_projection", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_att-use_span_lstm_projection-sche_samp", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-drop_irr_ents-use_span_lstm-use_span_att-use_span_lstm_projection", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-drop_irr_ents-use_span_lstm-use_span_att", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-drop_irr_ents-use_span_lstm-use_span_lstm_projection", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-drop_irr_ents-use_span_att-use_span_lstm_projection", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm-use_span_att", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm-use_span_att-use_span_lstm_projection", "LSTMMTL2CompleteGraph"),
        # ("bf_upper_bound", "LSTMMTL2CompleteGraph"),
        # ("bk_upper_bound", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm_2_layers-use_span_att_1_head-biaffine_512_dim-sche_samp_20_20-lr5e-4", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm_2_layers-use_span_att_1_head-biaffine_512_dim-sche_samp_20_20-lr7e-4", "LSTMMTL2CompleteGraph"),
        # NOTE: baseline
        # ("lstmmtl2cg-use_span_lstm_2_layers-use_span_att_1_head-biaffine_512_dim-shed_samp_20_20-lr1e-3", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm_2_layers-use_span_att_1_head-biaffine_512_dim-sche_samp_20_20-lr2e-3", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm_2_layers-use_span_att_1_head-biaffine_512_dim-sche_samp_20_20-lr3e-3", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm_2_layers-use_span_att_1_head-biaffine_512_dim-sche_samp_20_20-lr5e-3", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm_2_layers-use_span_att_1_head-biaffine_512_dim_xavier_normal-sche_samp_20_20-lr1e-3", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2soft_th_cg-use_span_lstm_2_layers-use_span_att_1_head-biaffine_512_dim-sche_samp_20_20-lr1e-3", "LSTMMTL2SoftThresholdCompleteGraph"),
        # ("lstmmtl2soft_th_cg-use_span_lstm_2_layers-use_span_att_1_head-biaffine_512_dim-sche_samp_20_10-lr1e-3", "LSTMMTL2SoftThresholdCompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm_2_layers-use_span_att_1_head-biaffine_512_dim-shed_samp_20_10-lr1e-3", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm_2_layers-use_span_att_1_head-biaffine_512_dim-shed_samp_20_20-lr1e-3", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-tuned-lstm_2_layers", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm_2_layers-use_span_att_1_head-biaffine_512_dim-no_role_cls-sche_samp_20_20-lr1e-3", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-all_filled_rel_mat_eval-use_span_lstm_2_layers-use_span_att_1_head-biaffine_512_dim-sche_samp_20_20-lr1e-3", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm_2_layers-use_span_att_1_head-biaffine_512_dim-shed_samp_10_20-lr1e-3", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm_2_layers-use_span_att_1_head-biaffine_512_dim-shed_samp_20_20-bs64-lr1e-3", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-lstmner_2_layers-use_span_lstm_2_layers-use_span_att_1_head-biaffine_512_dim-shed_samp_20_20-lr1e-3-full", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2comb_extra_gold_cg-use_span_lstm_2_layers-use_span_att_1_head-biaffine_512_dim-sche_samp_20_20-lr1e-3", "LSTMMTL2CombExtraGoldCG"),
        # ("lstmmtl2comb_extra_neg_cg-use_span_lstm_2_layers-use_span_att_1_head-biaffine_512_dim-sche_samp_20_20-lr1e-3", "LSTMMTL2CombExtraNegCG"),
        # ("lstmmtl2comb_extra_gold_neg_cg-use_span_lstm_2_layers-use_span_att_1_head-biaffine_512_dim-sche_samp_20_20-lr1e-3", "LSTMMTL2CombExtraGoldNegCG"),
        # ("lstmmtl2comb_sim_cg-use_span_lstm_2_layers-use_span_att_1_head-biaffine_512_dim-sche_samp_20_20-lr1e-3", "LSTMMTL2CombSimCG"),
        # ("lstmmtl2cg-all_filled_rel_mat_eval-use_span_lstm_2_layers-use_span_att_1_head-biaffine_512_dim-sche_samp_20_20-lr1e-3", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2hard_th_mse_cg-tuned", "LSTMMTL2HardThresholdMSECompleteGraph"),
        # ("lstmmtl2hard_th_mse_span_cls_cg-tuned", "LSTMMTL2HardThresholdMSESpanClsCG"),
        # ("lstmmtl2span_cls_cg-tuned", "LSTMMTL2SpanClsCG"),
        # ("lstmmtl2span_cls_cg-tuned-gold_100epochs", "LSTMMTL2SpanClsCG"),
        # ("lstmmtl2single_mlp_biaffine_cg-tuned", "LSTMMTL2SingleMLPBiaffineCG"),
        # ('lstmmtl2symmetric_cg-tuned', 'LSTMMTL2SymmetricCG'),
        # ('lstmmtl2symmetric_weight_component_cg-tuned', 'LSTMMTL2SymmetricWeightComponentCG'),
        # ('lstmmtl2dot_att_cg-tuned', 'LSTMMTL2DotAttentionCG'),
        # ('lstmmtl2dot_att_cg-biaffine_hidden_size_768-tuned', 'LSTMMTL2DotAttentionCG'),  # 2021年 03月 01日 星期一 20:37:42 CST
        # ('lstmmtl2triangle_cg-tuned', 'LSTMMTL2TriangleCG'),  # 2021年 03月 01日 星期一 10:34:49 CST
        # ('lstmmtl2span_cls_plus_cg-tuned', 'LSTMMTL2SpanClsPlusCG'),  # 2021年 03月 01日 星期一 15:43:02 CST
        # ('lstmmtl2mh_dot_att-tuned', 'LSTMMTL2MultiHeadDotAttentionCG'),    # 2021年 03月 02日 星期二 12:23:27 CST
        # ('lstmmtl2mh_dot_att-span_level_att-tuned', 'LSTMMTL2MultiHeadDotAttentionCG'),    # 2021年 03月 02日 星期二 12:30:14 CST
        # ('lstmmtl2comb_rand_samp_cg-tuned', 'LSTMMTL2CombRandSamplingCG'),    # 2021年 03月 02日 星期二 13:30:30 CST
        # ('lstmmtl2comb_all_samp_cg-samp_min_num_span_2-tuned', 'LSTMMTL2CombAllSamplingCG'),    # 2021年 03月 02日 星期二 20:41:06 CST
        # ('lstmmtl2comb_all_samp_cg-samp_min_num_span_5-tuned', 'LSTMMTL2CombAllSamplingCG'),    # 2021年 03月 02日 星期二 20:42:15 CST
        # ('lstmmtl2cg-no_span_att-tuned', 'LSTMMTL2CompleteGraphModel'),    # 2021年 03月 03日 星期三 10:25:29 CST
        # ('lstmmtl2cg-ent_level_span_att-tuned', 'LSTMMTL2CompleteGraphModel'),    # 2021年 03月 03日 星期三 10:33:07 CST
        # ('lstmmtl2sent_context_cg-tuned', 'LSTMMTL2SentContextCG'),    # 2021年 03月 04日 星期四 13:24:28 CST
        # ('lstmmtl2sent_context_cg-norm_plus-tuned', 'LSTMMTL2SentContextCG'),    # 2021年 03月 04日 星期四 13:24:28 CST
        # ('lstmmtl2span_cls_context_cg-tuned', 'LSTMMTL2SpanClsContextCG'),    # 2021年 03月 04日 星期四 14:21:05 CST
        # ('lstmmtl2span_cls_context_cg-no_gold-tuned', 'LSTMMTL2SpanClsContextCG'),    # 2021年 03月 04日 星期四 14:21:05 CST
        # ('lstmmtl2cg-comb_loss_weight.5-tuned', 'LSTMMTL2CompleteGraphModel'),    # 2021年 03月 05日 星期五 22:07:58 CST
        # ('lstmmtl2cg-comb_loss_weight2.0-tuned', 'LSTMMTL2CompleteGraphModel'),    # 2021年 03月 06日 星期六 12:19:52 CST
        # ('lstmmtl2cg-proj_dropout-tuned', 'LSTMMTL2VICompleteGraphModel'),    # 2021年 03月 06日 星期六 13:35:50 CST
        # ('lstmmtl2vi_cg-tuned', 'LSTMMTL2VICompleteGraphModel'),    # 2021年 03月 06日 星期六 15:44:52 CST
        # ('lstmmtl2symm_bia_out_cg-tuned', 'LSTMMTL2SymmBiaOutCG'),    # 2021年 03月 06日 星期六 16:22:56 CST
        # ('lstmmtl2batch_span_rep_cg-tuned', 'LSTMMTL2BatchSpanRepCG'),    # 2021年 03月 06日 星期六 19:27:09 CST
        # ('lstmmtl2o2o_simple_2_steps-tuned', 'LSTMMTL2O2OSimple2Steps'),    # 2021年 03月 06日 星期六 20:51:15 CST
        # ('lstmmtl2cg-grad_clip-tuned', 'LSTMMTL2CompleteGraphModel'),    # 2021年 03月 07日 星期日 19:25:29 CST
        # ('lstmmtl2cg-lr_scheduler-tuned', 'LSTMMTL2CompleteGraphModel'),    # 2021年 03月 07日 星期日 16:39:46 CST
        # ('lstmmtl2cg-lr_scheduler-tuned', 'LSTMMTL2CompleteGraphModel'),    # 2021年 03月 07日 星期日 16:39:46 CST
        # ('lstmmtl2cg-lr_scheduler-bs16_1-tuned', 'LSTMMTL2CompleteGraphModel'),    # 2021年 03月 07日 星期日 16:40:19 CST
        # ('lstmatt2cg-grad_clip-tuned', 'LSTMAtt2CG'),    # 2021年 03月 11日 星期四 10:00:46 CST
        # ("lstmmtl2cg-tuned-o2m-drop_irr_ents", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-tuned-o2m", "LSTMMTL2CompleteGraph"),
        # ('lstmmtl2cg-other_type-tuned', 'LSTMMTL2CompleteGraphModel'),    # 2021年 03月 15日 星期一 13:39:47 CST
        # ('lstmmtl2cg-drop_irr_ents-tuned', 'LSTMMTL2CompleteGraphModel'),    # 2021年 03月 15日 星期一 19:43:23 CST
        # ('lstmmtl2cg-biaffine_ner-tuned', 'LSTMMTL2CompleteGraphModel'),    # 2021年 03月 15日 星期一 14:54:28 CST
        # ('lstmmtl2cg-span_sum_mention-tuned', 'LSTMMTL2CompleteGraphModel'),    # 2021年 03月 15日 星期一 15:20:56 CST
        # ('lstmmtl2cg-ent_fix_f-tuned', 'LSTMMTL2CompleteGraphModel'),    # 2021年 03月 15日 星期一 18:18:25 CST
        # ('lstmmtl2cg-ent_fix_m-tuned', 'LSTMMTL2CompleteGraphModel'),    # 2021年 03月 15日 星期一 19:44:38 CST
        # ('lstmmtl2cg-small_param-tuned', 'LSTMMTL2CompleteGraphModel'),    # 2021年 03月 17日 星期三 10:13:56 CST
        # ('lstmmtl2sigmoid_multi_role_cg-tuned', 'LSTMMTL2SigmoidMultiRoleCG'),    # 2021年 03月 16日 星期二 14:57:15 CST
        # ('lstmmtl2sigmoid_multi_role_split_self_att_cg-tuned', 'LSTMMTL2SigmoidMultiRoleSplitSelfAttCG'),    # 2021年 03月 16日 星期二 15:06:46 CST
        # ('lstmmtl2sigmoid_multi_role_comb_match_cg-tuned', 'LSTMMTL2SigmoidMultiRoleCombMatchCG'),    # 2021年 03月 17日 星期三 11:19:32 CST
        # ('lstmmtl2dot_attended_sigmoid_multi_role_comb_match_cg-tuned', 'LSTMMTL2DotAttendedSigmoidMultiRoleCombMatchCG'),    # 2021年 03月 17日 星期三 19:33:17 CST
        # ('lstmmtl2dot_attended_sigmoid_multi_role_comb_match_cg-0_more_weight-tuned', 'LSTMMTL2DotAttendedSigmoidMultiRoleCombMatchCG'),    # 2021年 03月 17日 星期三 19:33:49 CST
        # ("lstmmtl2cg-use_span_lstm-use_span_att-shed_samp-bs64-lr1e-4", "LSTMMTL2CompleteGraph"),
        # ("lstmmtl2cg-use_span_lstm-use_span_att-shed_samp-bs64", "LSTMMTL2CompleteGraph"),
        # ('lstmmtl2cg-use_span_lstm-shed_samp', 'LSTMMTL2CompleteGraph'),
        # ('lstmmtl2cg-sche_samp', 'LSTMMTL2CompleteGraph'),
        # ('trans2cg-biaffine_512_dim-sche_samp-lr1e-4-bs64_32', 'Trans2CompleteGraphModel'),
        # ('trigger3-sche_samp_10_10-tuned', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # 2021年 03月 30日 星期二 17:18:43 CST
        # ('trigger2-sche_samp_10_10-tuned', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # 2021年 03月 30日 星期二 17:19:10 CST
        # ('trigger1-sche_samp_10_10-tuned', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # 2021年 03月 30日 星期二 17:19:32 CST
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-num_triggers1-sche_samp_10_10-tuned-seed2999', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Tue Apr  6 21:41:13 CST 2021
        # ('lstmmtl2cg-tuned-seed2999', 'LSTMMTL2CompleteGraphModel'),    # Tue Apr  6 21:43:47 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-guessing_bk_num_triggers6-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Wed Apr  7 20:15:33 CST 2021
        # ('lstmmtl2sigmoid_multi_role_comb_match_cg-bitwise_and-eval_bk_min_2-span_reverse_att_lstm-sche_samp_10_10-tuned-seed99', 'LSTMMTL2SigmoidMultiRoleCombMatchCG'),    # Fri Apr  9 12:29:46 CST 2021
        # ('lstmmtl2type_specified_sigmoid_multi_role_cg-simplified_event_cls-with_left_pred_comb-bitwise_and-eval_bk_min_2-sche_samp_10_10-tuned-seed99', 'LSTMMTL2TypeSpecifiedSigmoidMultiRoleCG'),    # Fri Apr  9 14:39:38 CST 2021
        # ('lstmmtl2type_specified_sigmoid_multi_role_cg-simplified_event_cls-without_left_pred_comb-bitwise_and-eval_bk_min_2-sche_samp_10_10-tuned-seed99', 'LSTMMTL2TypeSpecifiedSigmoidMultiRoleCG'),    # Fri Apr  9 16:58:09 CST 2021
        # ('lstmmtl2type_specified_sigmoid_multi_role_cg-simplified_event_cls-without_left_pred_comb-without_no_match_pred_comb-bitwise_and-eval_bk_min_2-sche_samp_10_10-tuned-seed99', 'LSTMMTL2TypeSpecifiedSigmoidMultiRoleCG'),    # Sat Apr 10 15:25:41 CST 2021
        # ('lstmmtl2type_specified_sigmoid_multi_role_cg-simplified_event_cls-without_left_pred_comb-batch_span_context-bitwise_and-eval_bk_min_2-sche_samp_10_10-tuned-seed99', 'LSTMMTL2TypeSpecifiedSigmoidMultiRoleCG'),    # Sat Apr 10 15:27:28 CST 2021
        # ('lstmmtl2type_specified_sigmoid_multi_role_cg-simplified_event_cls-without_left_pred_comb-without_no_match_pred_comb-batch_span_context-bitwise_and-eval_bk_min_2-sche_samp_10_10-tuned-seed99', 'LSTMMTL2TypeSpecifiedSigmoidMultiRoleCG'),    # Sat Apr 10 15:31:03 CST 2021
        # ('lstmmtl2type_specified_sigmoid_multi_role_cg-simplified_event_cls-without_left_pred_comb--bitwise_and-eval_bk_min_2-sche_samp_10_10-tuned-seed99', 'LSTMMTL2TypeSpecifiedSigmoidMultiRoleCG'),    # Sun Apr 11 11:05:18 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-bk_num_triggers1-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Thu Apr  8 19:49:10 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-bk_num_triggers2-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Thu Apr  8 19:49:10 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-bk_num_triggers3-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Thu Apr  8 19:49:10 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-bk_num_triggers4-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Thu Apr  8 19:49:10 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-bk_num_triggers5-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Thu Apr  8 19:49:10 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-bk_num_triggers6-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Thu Apr  8 19:49:10 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-bk_num_triggers7-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Thu Apr  8 19:49:10 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-bk_num_triggers8-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Thu Apr  8 19:49:10 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-bk_num_triggers9-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Thu Apr  8 19:49:10 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-bk_num_triggers1-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Wed Apr  7 20:15:33 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-bk_num_triggers2-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Wed Apr  7 20:15:33 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-bk_num_triggers3-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Wed Apr  7 20:15:33 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-bk_num_triggers4-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Wed Apr  7 20:15:33 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-bk_num_triggers5-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Wed Apr  7 20:15:33 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-bk_num_triggers6-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Wed Apr  7 20:15:33 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-bk_num_triggers7-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Wed Apr  7 20:15:33 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-bk_num_triggers8-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Wed Apr  7 20:15:33 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-bk_num_triggers9-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Wed Apr  7 20:15:33 CST 2021('trigger2sigmoid_multi_role_comb_match_cg-directed-bk_num_triggers6-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Mon Apr 12 13:45:57 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-bk_num_triggers7-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Mon Apr 12 13:46:13 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-bk_num_triggers8-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Mon Apr 12 13:46:52 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-bk_num_triggers9-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Mon Apr 12 13:47:10 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-incremental_guessing_triggers3-min_conn0-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Mon Apr 12 22:06:50 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-incremental_guessing_triggers3-min_conn1-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Mon Apr 12 22:07:08 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-incremental_guessing_triggers3-min_conn2-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Mon Apr 12 22:06:50 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-incremental_guessing_triggers3-min_conn3-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Mon Apr 12 22:07:08 CST 2021
        # ('lstmmtl2sigmoid_multi_role_comb_match_cg-span_self_att-sche_samp_10_10-tuned-seed99', 'LSTMMTL2SigmoidMultiRoleCombMatchCG'),    # Mon Apr 12 14:54:40 CST 2021
        # ('lstmmtl2sigmoid_multi_role_comb_match_cg-role_cls_by_span_self_att-sche_samp_10_10-tuned-seed99', 'LSTMMTL2SigmoidMultiRoleCombMatchCG'),    # Mon Apr 12 14:57:19 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-guessing_bk_num_triggers3-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Mon Apr 12 21:29:42 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-guessing_bk_num_triggers3-left_tril-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Mon Apr 12 21:29:42 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-train_triggers1_eval_trigger3-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Tue Apr 13 21:22:36 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-train_triggers3_eval_trigger1-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Tue Apr 13 21:08:39 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-bk_train_triggers1_eval_trigger3-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Tue Apr 13 21:23:12 CST 2021
        # ('trigger2sigmoid_multi_role_comb_match_cg-directed-guessing_bk_train_triggers1_eval_trigger3-sche_samp_10_10-tuned-seed99', 'Trigger2SigmoidMultiRoleCombMatchCG'),    # Tue Apr 13 21:23:46 CST 2021
        # ('smooth_att-sche_samp_10_10-tuned-seed99', 'LSTMMTL2SigmoidMultiRoleSmoothAttCombMatchCG'),    # Thu Apr 15 11:13:42 CST 2021
        # ('smooth_att-sche_samp_10_10-no_span_att-no_span_lstm-tuned-seed99', 'LSTMMTL2SigmoidMultiRoleSmoothAttCombMatchCG'),    # Thu Apr 15 11:14:16 CST 2021
        # ('smooth_att-sche_samp_10_10-ner_lstm_2_layers-no_span_att-no_span_lstm-tuned-seed99', 'LSTMMTL2SigmoidMultiRoleSmoothAttCombMatchCG'),    # Thu Apr 15 11:14:45 CST 2021
        # ('smooth_att-sche_samp_10_10-no_smooth-ner_lstm_2_layers-no_span_att-no_span_lstm-tuned-seed99', 'LSTMMTL2SigmoidMultiRoleSmoothAttCombMatchCG'),    # Thu Apr 15 11:17:48 CST 2021
        # ('smooth_att-sche_samp_10_10-no_smooth-tuned-seed99', 'LSTMMTL2SigmoidMultiRoleSmoothAttCombMatchCG'),    # Fri Apr 16 14:02:14 CST 2021
        # ('smooth_att-sche_samp_10_10-no_smooth-no_span_att-tuned-seed99', 'LSTMMTL2SigmoidMultiRoleSmoothAttCombMatchCG'),    # Fri Apr 16 14:02:49 CST 2021
        # ('smooth_att-sche_samp_10_10-no_span_att-tuned-seed99', 'LSTMMTL2SigmoidMultiRoleSmoothAttCombMatchCG'),    # Fri Apr 16 14:03:43 CST 2021
        # ('no_smooth-sche_samp_10_10-tuned-seed99', 'LSTMMTL2SigmoidMultiRoleSmoothAttCombMatchCG'),
        # ('no_smooth-no_span_att-sche_samp_10_10-tuned-seed99', 'LSTMMTL2SigmoidMultiRoleSmoothAttCombMatchCG'),
        # ('ner_2ly-span_lstm_1ly-sche_samp_10_10-tuned-seed99', 'LSTMMTL2SigmoidMultiRoleSmoothAttCombMatchCG'),
        # ('ner_2ly-sche_samp_10_10-tuned-seed99', 'LSTMMTL2SigmoidMultiRoleSmoothAttCombMatchCG'),
        # ('no_smooth-sche_samp_10_10-tuned-seed99', 'LSTMMTL2SigmoidMultiRoleSmoothAttCombMatchCG'),    # Sat Apr 17 19:18:44 CST 2021
        # ('no_smooth-no_span_att-sche_samp_10_10-tuned-seed99', 'LSTMMTL2SigmoidMultiRoleSmoothAttCombMatchCG'),    # Sat Apr 17 19:19:29 CST 2021
        # ('smooth-no_span_att-sche_samp_10_10-tuned-seed99', 'LSTMMTL2SigmoidMultiRoleSmoothAttCombMatchCG'),    # Sat Apr 17 19:20:04 CST 2021
        # ('ner_2ly-span_lstm_1ly-sche_samp_10_10-tuned-seed99', 'LSTMMTL2SigmoidMultiRoleSmoothAttCombMatchCG'),    # Sat Apr 17 19:21:34 CST 2021
        # ('ner_2ly-sche_samp_10_10-tuned-seed99', 'LSTMMTL2SigmoidMultiRoleSmoothAttCombMatchCG'),    # Sat Apr 17 19:22:19 CST 2021
        # ('directed_trigger1-span_self_att-sche_samp_10_10-tuned-seed99', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Sun Apr 18 16:44:56 CST 2021
        # ('directed_trigger1-span_self_att-ment_type_concat-sche_samp_10_10-tuned-seed99', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Sun Apr 18 21:27:23 CST 2021
        # ('directed_trigger1-span_self_att-ment_type_concat-other_type-sche_samp_10_10-tuned-seed99', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Sun Apr 18 21:28:41 CST 2021
        # ('directed_trigger1-span_self_att-ment_type_concat-sche_samp_10_10-tuned-seed999', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Mon Apr 19 12:23:46 CST 2021
        # ('directed_trigger1-span_self_att-no_ment_type-sche_samp_10_10-tuned-seed999', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Mon Apr 19 12:29:02 CST 2021
        # ('directed_trigger1-span_self_att-ment_type_concat-other_type-no_sche_samp-tuned-seed999', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Mon Apr 19 12:32:10 CST 2021
        # ('directed_trigger1-span_self_att_score_as_adj_mat-ment_type_concat-other_type-sche_samp_10_10-tuned-seed999', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Mon Apr 19 13:02:20 CST 2021
        # ('directed_trigger1-span_self_att_score_as_adj_mat-ment_type_concat-other_type-sche_samp_10_10-tuned-seed99', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Tue Apr 20 10:21:32 CST 2021
        # ('directed_trigger1-span_self_att_score_as_adj_mat-ment_type_concat-other_type-ner_1_ly-sche_samp_10_10-tuned-seed99', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Tue Apr 20 10:22:40 CST 2021
        # ('directed_trigger1-span_self_att_score_as_adj_mat-ment_type_concat-other_type-ner_1_ly-batch_span_context-sche_samp_10_10-tuned-seed99', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Tue Apr 20 10:30:03 CST 2021
        # ('directed_trigger1-span_self_att_score_as_adj_mat-ment_type_concat-other_type-ner_2_ly-batch_span_context-sche_samp_10_10-tuned-seed99', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Tue Apr 20 10:30:27 CST 2021
        # ('directed_trigger1-span_self_att_score_as_adj_mat-ment_type_concat-other_type-ner_2_ly-diagonal_1-sche_samp_10_10-tuned-seed99', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Wed Apr 21 10:32:33 CST 2021
        # ('directed_trigger1-span_self_att_score_as_adj_mat-ment_type_concat-other_type-ner_2_ly-diagonal_1-mse-sche_samp_10_10-tuned-seed99', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Wed Apr 21 10:32:33 CST 2021
        # ('directed_trigger1-score_scaling', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Mon May 24 23:10:04 CST 2021
        # ('directed_trigger1-score_scaling-full', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Tue May 25 11:40:43 CST 2021
        # ('debug', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Tue May 25 11:40:43 CST 2021
        # ('directed_trigger1-score_scaling-quarter', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Wed May 26 16:56:29 CST 2021
        # ('directed_trigger1-score_scaling-1o8', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Thu May 27 22:12:43 CST 2021
        # ('directed_trigger1-score_scaling-quarter', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Wed May 26 16:56:46 CST 2021
        # ('directed_trigger1-score_scaling-half', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Thu May 27 22:03:30 CST 2021
        # ('directed_trigger1-score_scaling-half-sujianlin_bce', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Thu May 27 22:03:30 CST 2021
        # ('directed_trigger1-score_scaling-full', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Wed May 26 16:55:48 CST 2021
        # ('directed_trigger1-score_scaling-full-bce', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Wed May 26 16:55:48 CST 2021
        # ('directed_trigger1-dot_att-bce_loss-role_by_span_lstm', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Sat Jun 12 18:36:18 CST 2021
        # ('directed_trigger1-dot_att-full-bce-lr5e-4', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Wed Jun  9 13:51:00 CST 2021
        # ('doc_graph-no_span_lstm', 'DocGraphTrigger2SigmoidMultiRoleCombMatchCG'),    # Thu Jun 10 22:52:34 CST 2021
        # ('doc_graph-span_lstm_1lyr', 'DocGraphTrigger2SigmoidMultiRoleCombMatchCG'),    # Thu Jun 10 22:52:34 CST 2021
        # ('doc_graph-span_lstm_1lyr-lr5e-4', 'DocGraphTrigger2SigmoidMultiRoleCombMatchCG'),    # Fri Jun 11 22:20:09 CST 2021
        # ('doc_graph-dot_att-bce_loss-role_by_gat', 'DocGraphTrigger2SigmoidMultiRoleCombMatchCG'),    # Sat Jun 12 18:39:03 CST 2021
        # ('doc_graph-gcn-dot_att-bce_loss-no_span_lstm', 'DocGraphTrigger2SigmoidMultiRoleCombMatchCG'),    # Sun Jun 13 15:22:42 CST 2021
        # ('doc_graph-gcn-dot_att-bce_loss-role_by_gcn', 'DocGraphTrigger2SigmoidMultiRoleCombMatchCG'),    # Sun Jun 13 16:08:36 CST 2021
        # ('mention_lstm_2layers-no_span_lstm', 'MentionEncodingTrigger2SigmoidMultiRoleCombMatchCG'),    # Thu Jun 10 16:38:49 CST 2021
        # ('mention_lstm_1lyr-span_lstm_1lyr', 'MentionEncodingTrigger2SigmoidMultiRoleCombMatchCG'),    # Thu Jun 10 22:54:22 CST 2021
        # ('directed_trigger1-score_scaling-full-sujianlin_bce', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Wed May 26 16:55:48 CST 2021
        # ('directed_trigger1-cos_sim-full-mse', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Tue Jun  8 14:18:13 CST 2021
        # ('directed_trigger1-score_scaling-full-mse_clamp', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Sun May 30 19:33:29 CST 2021
        # ('directed_trigger1-score_scaling-full-no_sigmoid', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Wed May 26 16:55:48 CST 2021
        # ('directed_trigger1-score_scaling-full-bce_no_sigmoid', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Sat May 29 21:55:44 CST 2021
        # ('directed_trigger1-cos_sim-full-mse_cl_0.05', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Tue Jun  8 17:06:30 CST 2021
        # ('directed_trigger1-cos_sim-full-cos_emb_loss', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # Wed Jun  9 13:18:29 CST 2021
        # ('n143-directed_trigger1-dot_att-bce_loss-role_by_encoding-mention_lstm_1_lyr-span_lstm_1lyr-mlp_before_adj_measure-lr5e-4', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),    # 2021年 06月 18日 星期五 14:56:27 CST
        ("n143-Tp1CG-bs16", "TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG"),
        # ('iai-trigger-2head-maxmean', 'MultiHeadTriggerGraph'),    # Tue Aug  3 20:10:35 CST 2021
        # ('iai-trigger_hetero_encoding-1head', 'MultiHeadTriggerGraphWithHeteroNodeEncoding'),    # Tue Aug  3 20:54:34 CST 2021
        # ('n143-Tp1CG-comp_ents-bs16', 'TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG'),
        (
            "n143-Tp1CG-bs16-with_left_triggers",
            "TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG",
        ),  # 2021年 08月 07日 星期六 15:16:14 CST
        (
            "n143-Tp1CG-try_to_make_up-with_left_trigger",
            "TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG",
        ),  # 2021年 08月 07日 星期六 15:16:14 CST
        (
            "n143-Tp1CG-try_to_make_up-with_left_trigger-self_loop",
            "TriggerAwarePrunedCompleteGraph",
        ),  # 2021年 08月 07日 星期六 15:17:03 CST
        (
            "n143-Tp22CG-try_to_make_up-with_left_trigger",
            "TypeSpecifiedTrigger2SigmoidMultiRoleCombMatchCG",
        ),  # 2021年 08月 07日 星期六 15:19:21 CST
        (
            "n143-Tp22CG-try_to_make_up-with_left_trigger-self_loop",
            "TriggerAwarePrunedCompleteGraph",
        ),  # 2021年 08月 07日 星期六 15:19:58 CST
    ]
    for task_name, model_name in records:
        print("\n")
        # print_paper_result(task_name, result_type="total")
        # print_paper_result(task_name, result_type="s&m")
        # span_type = 'gold_span'
        doc_type = "overall"
        data_type = "test"
        metric_type = "micro"

        doc_type2data_span_type2model_str2epoch_res_list = aggregate_task_eval_info(
            f"Exps/{task_name}/Output/", max_epoch=total_epoch, dump_flag=True
        )
        dee_task_setting = DEETaskSetting.from_pretrained(
            f"Exps/{task_name}/{model_name}.task_setting.json"
        )
        dee_task = DEETask(
            dee_task_setting,
            load_train=False,
            load_dev=False,
            load_test=True,
            parallel_decorate=False,
        )

        for span_type in ["pred_span", "gold_span"]:
            # if task_name == 'lstmmtl2cg':
            #     total_epoch = 200
            # print_specified_epoch(task_name, model_name, best_epoch, span_type="pred_span")

            mstr_bepoch_list, total_results = print_total_eval_info(
                doc_type2data_span_type2model_str2epoch_res_list,
                dee_task.template,
                metric_type=metric_type,
                span_type=span_type,
                model_strs=model_name.split(","),
                doc_type=doc_type,
                target_set=data_type,
            )
            sm_results = print_single_vs_multi_performance(
                mstr_bepoch_list,
                f"Exps/{task_name}/Output/",
                dee_task.test_features,
                dee_task.event_template,
                dee_task_setting.event_relevant_combination,
                metric_type=metric_type,
                data_type=data_type,
                span_type=span_type,
            )

            best_epoch = print_best_test_via_dev(
                task_name,
                model_name,
                total_epoch,
                span_type=span_type,
                measure_type="overall",
                measure_key="MicroF1",
            )
            print_detailed_specified_epoch(
                task_name, model_name, best_epoch, span_type=span_type
            )

            # eval_res = load_evaluate_results(task_name, model_name, "o2o", "test", span_type, best_epoch)["adj_mat"]["Accuracy"]
            # print("{:.3f}".format(eval_res * 100))
            # eval_res = load_evaluate_results(task_name, model_name, "o2m", "test", span_type, best_epoch)["adj_mat"]["Accuracy"]
            # print("{:.3f}".format(eval_res * 100))
            # eval_res = load_evaluate_results(task_name, model_name, "m2m", "test", span_type, best_epoch)["adj_mat"]["Accuracy"]
            # print("{:.3f}".format(eval_res * 100))
            # eval_res = load_evaluate_results(task_name, model_name, "overall", "test", span_type, best_epoch)["adj_mat"]["Accuracy"]
            # print("{:.3f}".format(eval_res * 100))
            # get_macro_overall(task_name, model_name, total_epoch, span_type=span_type, data_type="overall", verbose=False)
            # print_score_on_each_epoch(task_name, model_name, total_epoch,
            #                           measure_type="overall",
            #                           measure_key="MicroF1",
            #                           data_type="overall",
            #                           span_type=span_type)
            # print_detailed_specified_epoch(task_name, model_name, 50, span_type="gold_span")
            # print_tp_fp_fn(task_name, model_name,
            #                1,
            #                dataset="test",
            #                measure_type="overall",
            #                span_type="gold_span")

    #     best_epoch = print_best_test_via_dev(task_name, model_name, total_epoch, span_type="pred_span", measure_type="overall")
    #     result = load_evaluate_results(task_name, model_name, "overall", "test", "pred_span", best_epoch)
    #     type2results["Overall"].append(result['overall']['MicroF1'])
    #     for events, _ in result['overall']['Events']:
    #         type2results[events['EventType']].append(events['MicroF1'])

    # for i in range(9):
    #     for event_type in ["EquityFreeze", "EquityRepurchase", "EquityUnderweight", "EquityOverweight", "EquityPledge", "Overall"]:
    #         print(type2results[event_type][i], end='\t')
    #     print()
