import numpy as np

from dee.modules import adj_decoding
from dee.utils import (
    extract_combinations_from_event_objs,
    extract_instances_from_event_objs,
)


def agg_event_role_tpfpfn_stats(pred_records, gold_records, role_num):
    """
    Aggregate TP,FP,FN statistics for a single event prediction of one instance.
    A pred_records should be formated as
    [(Record Index)
        ((Role Index)
            argument 1, ...
        ), ...
    ], where argument 1 should support the '=' operation and the empty argument is None.
    """
    role_tpfpfn_stats = [[0] * 3 for _ in range(role_num)]

    if gold_records is None:
        if pred_records is not None:  # FP
            for pred_record in pred_records:
                assert len(pred_record) == role_num
                for role_idx, arg_tup in enumerate(pred_record):
                    if arg_tup is not None:
                        role_tpfpfn_stats[role_idx][1] += 1
        else:  # ignore TN
            pass
    else:
        if pred_records is None:  # FN
            for gold_record in gold_records:
                assert len(gold_record) == role_num
                for role_idx, arg_tup in enumerate(gold_record):
                    if arg_tup is not None:
                        role_tpfpfn_stats[role_idx][2] += 1
        else:
            # True Positive at the event level
            # sort predicted event records by the non-empty count
            # to remove the impact of the record order on evaluation
            pred_records = sorted(
                pred_records,
                key=lambda x: sum(1 for a in x if a is not None),
                reverse=True,
            )
            gold_records = list(gold_records)

            while len(pred_records) > 0 and len(gold_records) > 0:
                pred_record = pred_records[0]
                assert len(pred_record) == role_num

                # pick the most similar gold record
                _tmp_key = lambda gr: sum(
                    [1 for pa, ga in zip(pred_record, gr) if pa == ga]
                )
                best_gr_idx = gold_records.index(max(gold_records, key=_tmp_key))
                gold_record = gold_records[best_gr_idx]

                for role_idx, (pred_arg, gold_arg) in enumerate(
                    zip(pred_record, gold_record)
                ):
                    if gold_arg is None:
                        if pred_arg is not None:  # FP at the role level
                            role_tpfpfn_stats[role_idx][1] += 1
                        else:  # ignore TN
                            pass
                    else:
                        if pred_arg is None:  # FN
                            role_tpfpfn_stats[role_idx][2] += 1
                        else:
                            if pred_arg == gold_arg:  # TP
                                role_tpfpfn_stats[role_idx][0] += 1
                            else:
                                # tzhu: pred and gold are not None, and pred != gold, then this is a FP and FN condition
                                role_tpfpfn_stats[role_idx][1] += 1
                                role_tpfpfn_stats[role_idx][2] += 1

                del pred_records[0]
                del gold_records[best_gr_idx]

            # remaining FP
            for pred_record in pred_records:
                assert len(pred_record) == role_num
                for role_idx, arg_tup in enumerate(pred_record):
                    if arg_tup is not None:
                        role_tpfpfn_stats[role_idx][1] += 1
            # remaining FN
            for gold_record in gold_records:
                assert len(gold_record) == role_num
                for role_idx, arg_tup in enumerate(gold_record):
                    if arg_tup is not None:
                        role_tpfpfn_stats[role_idx][2] += 1

    return role_tpfpfn_stats


def agg_event_level_tpfpfn_stats(pred_records, gold_records, role_num):
    """
    Get event-level TP,FP,FN
    """
    # add role-level statistics as the event-level ones
    role_tpfpfn_stats = agg_event_role_tpfpfn_stats(
        pred_records, gold_records, role_num
    )

    return list(np.sum(role_tpfpfn_stats, axis=0))


def agg_ins_event_role_tpfpfn_stats(
    pred_record_mat, gold_record_mat, event_role_num_list
):
    """
    Aggregate TP,FP,FN statistics for a single instance.
    A record_mat should be formated as
    [(Event Index)
        [(Record Index)
            ((Role Index)
                argument 1, ...
            ), ...
        ], ...
    ], where argument 1 should support the '=' operation and the empty argument is None.
    """
    assert len(pred_record_mat) == len(gold_record_mat)
    # tpfpfn_stat: TP, FP, FN
    event_role_tpfpfn_stats = []
    for event_idx, (pred_records, gold_records) in enumerate(
        zip(pred_record_mat, gold_record_mat)
    ):
        role_num = event_role_num_list[event_idx]
        role_tpfpfn_stats = agg_event_role_tpfpfn_stats(
            pred_records, gold_records, role_num
        )
        event_role_tpfpfn_stats.append(role_tpfpfn_stats)

    return event_role_tpfpfn_stats


def agg_ins_event_level_tpfpfn_stats(
    pred_record_mat, gold_record_mat, event_role_num_list
):
    assert len(pred_record_mat) == len(gold_record_mat)
    # tpfpfn_stat: TP, FP, FN
    event_tpfpfn_stats = []
    for event_idx, (pred_records, gold_records, role_num) in enumerate(
        zip(pred_record_mat, gold_record_mat, event_role_num_list)
    ):
        event_tpfpfn = agg_event_level_tpfpfn_stats(
            pred_records, gold_records, role_num
        )
        event_tpfpfn_stats.append(event_tpfpfn)

    return event_tpfpfn_stats


def get_prec_recall_f1(tp, fp, fn):
    a = tp + fp
    prec = tp / a if a > 0 else 0
    b = tp + fn
    rec = tp / b if b > 0 else 0
    if prec > 0 and rec > 0:
        f1 = 2.0 / (1 / prec + 1 / rec)
    else:
        f1 = 0
    return prec, rec, f1


def get_mcml_prf1(pred_event_types, gold_event_types, event_type_roles_list):
    """get p r f1 measures of classification results"""
    len_events = len(event_type_roles_list)
    event_tp_fp_fn = [[0] * 3 for _ in range(len_events)]
    event_p_r_f1 = [[0.0] * 3 for _ in range(len_events)]
    tot_tp_fp_fn = [0] * 3
    tot_p_r_f1 = [0.0] * 3
    for preds, golds in zip(pred_event_types, gold_event_types):
        for event_idx, (pred, gold) in enumerate(zip(preds, golds)):
            if pred == 0:
                if gold == 0:  # TN
                    pass
                else:  # FN
                    event_tp_fp_fn[event_idx][2] += 1
            else:
                if gold == 0:  # FP
                    event_tp_fp_fn[event_idx][1] += 1
                else:  # TP: if both pred and gold contains paths for this event, then it's TP
                    event_tp_fp_fn[event_idx][0] += 1
    for event_idx, tp_fp_fn in enumerate(event_tp_fp_fn):
        tot_tp_fp_fn[0] += tp_fp_fn[0]
        tot_tp_fp_fn[1] += tp_fp_fn[1]
        tot_tp_fp_fn[2] += tp_fp_fn[2]
        prec, rec, f1 = get_prec_recall_f1(*tp_fp_fn)
        event_p_r_f1[event_idx][0] = prec
        event_p_r_f1[event_idx][1] = rec
        event_p_r_f1[event_idx][2] = f1

    micro_p, micro_r, micro_f1 = get_prec_recall_f1(*tot_tp_fp_fn)
    tot_p_r_f1[0] = micro_p
    tot_p_r_f1[1] = micro_r
    tot_p_r_f1[2] = micro_f1
    macro_p = sum([x[0] for x in event_p_r_f1]) / len_events
    macro_r = sum([x[1] for x in event_p_r_f1]) / len_events
    macro_f1 = sum([x[2] for x in event_p_r_f1]) / len_events

    results = {
        "MacroPrecision": macro_p,
        "MacroRecall": macro_r,
        "MacroF1": macro_f1,
        "MicroPrecision": micro_p,
        "MicroRecall": micro_r,
        "MicroF1": micro_f1,
        "TP": tot_tp_fp_fn[0],
        "FP": tot_tp_fp_fn[1],
        "FN": tot_tp_fp_fn[2],
        "Events": [
            {
                "EventType": event_type_roles_list[event_idx][0],
                "Precision": event_p_r_f1[event_idx][0],
                "Recall": event_p_r_f1[event_idx][1],
                "F1": event_p_r_f1[event_idx][2],
                "TP": event_tp_fp_fn[event_idx][0],
                "FP": event_tp_fp_fn[event_idx][1],
                "FN": event_tp_fp_fn[event_idx][2],
            }
            for event_idx in range(len_events)
        ],
    }
    return results


def get_ent_prf1(pred_spans_token_tuple_list, gold_spans_token_tuple_list):
    """get p r f1 measures of entity prediction results"""
    tot_tp_fp_fn = [0] * 3
    tot_p_r_f1 = [0.0] * 3

    for preds, golds in zip(
        pred_spans_token_tuple_list, gold_spans_token_tuple_list
    ):  # doc
        pred_event_ents = set(preds)
        gold_event_ents = set(golds)
        tot_tp_fp_fn[0] += len(pred_event_ents & gold_event_ents)  # TP
        tot_tp_fp_fn[1] += len(pred_event_ents - gold_event_ents)  # FP
        tot_tp_fp_fn[2] += len(gold_event_ents - pred_event_ents)  # FN

    micro_p, micro_r, micro_f1 = get_prec_recall_f1(*tot_tp_fp_fn)
    tot_p_r_f1[0] = micro_p
    tot_p_r_f1[1] = micro_r
    tot_p_r_f1[2] = micro_f1

    results = {
        "MicroPrecision": micro_p,
        "MicroRecall": micro_r,
        "MicroF1": micro_f1,
        "TP": tot_tp_fp_fn[0],
        "FP": tot_tp_fp_fn[1],
        "FN": tot_tp_fp_fn[2],
    }
    return results


def get_combination_prf1(
    pred_record_mat_list, gold_record_mat_list, input_type="event_obj"
):
    TP, FP, FN = 0, 0, 0
    for pred_record_mat, gold_record_mat in zip(
        pred_record_mat_list, gold_record_mat_list
    ):
        if input_type == "event_obj":
            pred_combinations = extract_combinations_from_event_objs(pred_record_mat)
            gold_combinations = extract_combinations_from_event_objs(gold_record_mat)
        else:
            pred_combinations = pred_record_mat
            gold_combinations = gold_record_mat
        TP += len(pred_combinations & gold_combinations)
        FP += len(pred_combinations - gold_combinations)
        FN += len(gold_combinations - pred_combinations)
    p, r, f1 = get_prec_recall_f1(TP, FP, FN)
    result = {
        "MicroPrecision": p,
        "MicroRecall": r,
        "MicroF1": f1,
        "TP": TP,
        "FP": FP,
        "FN": FN,
    }
    return result


def get_instance_prf1(pred_record_mat_list, gold_record_mat_list):
    TP, FP, FN = 0, 0, 0
    for pred_record_mat, gold_record_mat in zip(
        pred_record_mat_list, gold_record_mat_list
    ):
        pred_instances = extract_instances_from_event_objs(pred_record_mat)
        gold_instances = extract_instances_from_event_objs(gold_record_mat)
        TP += len(pred_instances & gold_instances)
        FP += len(pred_instances - gold_instances)
        FN += len(gold_instances - pred_instances)
    p, r, f1 = get_prec_recall_f1(TP, FP, FN)
    result = {
        "MicroPrecision": p,
        "MicroRecall": r,
        "MicroF1": f1,
        "TP": TP,
        "FP": FP,
        "FN": FN,
    }
    return result


def get_adj_mat_acc(pred_adj_mats, gold_adj_mats):
    assert len(pred_adj_mats) == len(gold_adj_mats)
    num_total = len(pred_adj_mats)
    num_correct = 0
    for pred_adj_mat, gold_adj_mat in zip(pred_adj_mats, gold_adj_mats):
        pred_adj_mat = np.array(pred_adj_mat)
        gold_adj_mat = np.array(gold_adj_mat)
        if gold_adj_mat.shape[0] == 1 and len(gold_adj_mat.shape) == 3:
            gold_adj_mat = np.squeeze(gold_adj_mat, axis=0)
        if pred_adj_mat.shape[0] == 1 and len(pred_adj_mat.shape) == 3:
            pred_adj_mat = np.squeeze(pred_adj_mat, axis=0)
        try:
            np.fill_diagonal(gold_adj_mat, 0)
            np.fill_diagonal(pred_adj_mat, 0)
            if np.array_equal(pred_adj_mat, gold_adj_mat):
                num_correct += 1
        except ValueError:
            # if there is no predicted named entities due to entity reduction, then pass
            pass
    return num_correct / num_total


def get_adj_mat_conn_metrics(batch_pred_adj_mats, batch_gold_adj_mats):
    assert len(batch_pred_adj_mats) == len(batch_gold_adj_mats)
    tp = fp = fn = 0
    for pred_adj_mats, gold_adj_mats in zip(batch_pred_adj_mats, batch_gold_adj_mats):
        for pred_adj_mat, gold_adj_mat in zip(pred_adj_mats, gold_adj_mats):
            pred_connections = adj_decoding.build_single_element_connections(
                pred_adj_mat, tuple_key=False
            )
            pred_tot_connections = set()
            for u, vs in pred_connections.items():
                for v in vs:
                    pred_tot_connections.add((u, v))
            gold_connections = adj_decoding.build_single_element_connections(
                gold_adj_mat, tuple_key=False
            )
            gold_tot_connections = set()
            for u, vs in gold_connections.items():
                for v in vs:
                    gold_tot_connections.add((u, v))
            tp += len(pred_tot_connections & gold_tot_connections)
            fp += len(pred_tot_connections - gold_tot_connections)
            fn += len(gold_tot_connections - pred_tot_connections)

    p, r, f1 = get_prec_recall_f1(tp, fp, fn)
    metrics = {
        "MicroPrecision": p,
        "MicroRecall": r,
        "MicroF1": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }
    return metrics


def get_trigger_identification_metrics(batch_pred_adj_mats, batch_gold_adj_mats):
    assert len(batch_pred_adj_mats) == len(batch_gold_adj_mats)
    tp = fp = fn = 0
    for pred_adj_mats, gold_adj_mats in zip(batch_pred_adj_mats, batch_gold_adj_mats):
        for pred_adj_mat, gold_adj_mat in zip(pred_adj_mats, gold_adj_mats):
            pred_connections = adj_decoding.build_single_element_connections(
                pred_adj_mat, tuple_key=False
            )
            pred_triggers = set()
            for u, vs in pred_connections.items():
                if len(vs) > 0:
                    pred_triggers.add(u)
            gold_connections = adj_decoding.build_single_element_connections(
                gold_adj_mat, tuple_key=False
            )
            gold_triggers = set()
            for u, vs in gold_connections.items():
                if len(vs) > 0:
                    gold_triggers.add(u)
            tp += len(pred_triggers & gold_triggers)
            fp += len(pred_triggers - gold_triggers)
            fn += len(gold_triggers - pred_triggers)

    p, r, f1 = get_prec_recall_f1(tp, fp, fn)
    metrics = {
        "MicroPrecision": p,
        "MicroRecall": r,
        "MicroF1": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }
    return metrics


def measure_event_table_filling(
    pred_record_mat_list,
    gold_record_mat_list,
    event_type_roles_list,
    pred_event_types,
    gold_event_types,
    pred_spans_token_tuple_list,
    gold_spans_token_tuple_list,
    pred_combinations=None,
    gold_combinations=None,
    pred_adj_mats=None,
    gold_adj_mats=None,
    avg_type="micro",
    dict_return=False,
):
    """
    The record_mat_list is formated as
    [(Document Index)
        [(Event Index): None if the event is not detected or not existed
            [(Record Index)
                ((Role Index)
                    argument 1, ...
                ), ...
            ], ...
        ], ...
    ]
    The argument type should support the '==' operation.
    Empty arguments and records are set as None.
    each tuple is a span token ids
    """
    raw_combination_prf1 = None
    event_mcml_prf1 = get_mcml_prf1(
        pred_event_types, gold_event_types, event_type_roles_list
    )
    ent_prf1 = get_ent_prf1(pred_spans_token_tuple_list, gold_spans_token_tuple_list)
    if pred_combinations and gold_combinations:
        raw_combination_prf1 = get_combination_prf1(
            pred_combinations, gold_combinations, input_type="combination"
        )
    combination_prf1 = get_combination_prf1(pred_record_mat_list, gold_record_mat_list)
    instance_prf1 = get_instance_prf1(pred_record_mat_list, gold_record_mat_list)
    event_role_num_list = [len(x[1]) for x in event_type_roles_list]
    # to store total statistics of TP, FP, FN
    total_event_role_stats = [
        [[0] * 3 for _ in range(role_num)]
        for event_idx, role_num in enumerate(event_role_num_list)
    ]

    assert len(pred_record_mat_list) == len(gold_record_mat_list)
    # tzhu: for every documents
    for pred_record_mat, gold_record_mat in zip(
        pred_record_mat_list, gold_record_mat_list
    ):
        event_role_tpfpfn_stats = agg_ins_event_role_tpfpfn_stats(
            pred_record_mat, gold_record_mat, event_role_num_list
        )
        for event_idx, role_num in enumerate(event_role_num_list):
            for role_idx in range(role_num):
                for sid in range(3):
                    total_event_role_stats[event_idx][role_idx][
                        sid
                    ] += event_role_tpfpfn_stats[event_idx][role_idx][sid]

    per_role_metric = []
    per_event_metric = []

    num_events = len(event_role_num_list)
    g_tpfpfn_stat = [0] * 3
    g_prf1_stat = [0] * 3
    event_role_eval_dicts = []
    for event_idx, role_num in enumerate(event_role_num_list):
        event_tpfpfn = [0] * 3  # tp, fp, fn
        event_prf1_stat = [0] * 3
        per_role_metric.append([])
        role_eval_dicts = []
        for role_idx in range(role_num):
            role_tpfpfn_stat = total_event_role_stats[event_idx][role_idx][:3]
            role_prf1_stat = get_prec_recall_f1(*role_tpfpfn_stat)
            per_role_metric[event_idx].append(role_prf1_stat)
            for mid in range(3):
                event_tpfpfn[mid] += role_tpfpfn_stat[mid]
                event_prf1_stat[mid] += role_prf1_stat[mid]

            role_eval_dict = {
                "RoleType": event_type_roles_list[event_idx][1][role_idx],
                "Precision": role_prf1_stat[0],
                "Recall": role_prf1_stat[1],
                "F1": role_prf1_stat[2],
                "TP": role_tpfpfn_stat[0],
                "FP": role_tpfpfn_stat[1],
                "FN": role_tpfpfn_stat[2],
            }
            role_eval_dicts.append(role_eval_dict)

        for mid in range(3):
            event_prf1_stat[mid] /= role_num
            g_tpfpfn_stat[mid] += event_tpfpfn[mid]
            g_prf1_stat[mid] += event_prf1_stat[mid]

        micro_event_prf1 = get_prec_recall_f1(*event_tpfpfn)
        macro_event_prf1 = tuple(event_prf1_stat)
        if avg_type.lower() == "micro":
            event_prf1_stat = micro_event_prf1
        elif avg_type.lower() == "macro":
            event_prf1_stat = macro_event_prf1
        else:
            raise Exception("Unsupported average type {}".format(avg_type))

        per_event_metric.append(event_prf1_stat)

        event_eval_dict = {
            "EventType": event_type_roles_list[event_idx][0],
            "MacroPrecision": macro_event_prf1[0],
            "MacroRecall": macro_event_prf1[1],
            "MacroF1": macro_event_prf1[2],
            "MicroPrecision": micro_event_prf1[0],
            "MicroRecall": micro_event_prf1[1],
            "MicroF1": micro_event_prf1[2],
            "TP": event_tpfpfn[0],
            "FP": event_tpfpfn[1],
            "FN": event_tpfpfn[2],
        }
        event_role_eval_dicts.append((event_eval_dict, role_eval_dicts))

    micro_g_prf1 = get_prec_recall_f1(*g_tpfpfn_stat)
    macro_g_prf1 = tuple(s / num_events for s in g_prf1_stat)
    if avg_type.lower() == "micro":
        g_metric = micro_g_prf1
    elif avg_type.lower() == "macro":
        g_metric = macro_g_prf1
    else:
        raise ValueError("Unsupported average type {}".format(avg_type))

    g_eval_dict = {
        "MacroPrecision": macro_g_prf1[0],
        "MacroRecall": macro_g_prf1[1],
        "MacroF1": macro_g_prf1[2],
        "MicroPrecision": micro_g_prf1[0],
        "MicroRecall": micro_g_prf1[1],
        "MicroF1": micro_g_prf1[2],
        "TP": g_tpfpfn_stat[0],
        "FP": g_tpfpfn_stat[1],
        "FN": g_tpfpfn_stat[2],
        "Events": event_role_eval_dicts,
    }
    eval_dict = {
        "classification": event_mcml_prf1,
        "entity": ent_prf1,
        "combination": combination_prf1,
        "overall": g_eval_dict,
        "instance": instance_prf1,
    }
    if pred_adj_mats and gold_adj_mats:
        eval_dict.update(
            {
                "adj_mat": {"Accuracy": get_adj_mat_acc(pred_adj_mats, gold_adj_mats)},
                "connection": get_adj_mat_conn_metrics(pred_adj_mats, gold_adj_mats),
                "trigger": get_trigger_identification_metrics(
                    pred_adj_mats, gold_adj_mats
                ),
            }
        )
    if raw_combination_prf1:
        eval_dict.update({"rawCombination": raw_combination_prf1})

    if dict_return:
        return eval_dict
    else:
        return g_metric, per_event_metric, per_role_metric
