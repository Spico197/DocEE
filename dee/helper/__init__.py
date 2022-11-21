import copy
import os
import re
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

from dee.metrics import measure_event_table_filling
from dee.utils import (
    convert_role_fea_event_obj_to_standard,
    default_dump_json,
    default_dump_pkl,
    default_load_json,
    default_load_pkl,
    extract_combinations_from_event_objs,
    remove_combination_roles,
    remove_event_obj_roles,
)

from .arg_rel import (
    DEEArgRelFeature,
    DEEArgRelFeatureConverter,
    convert_dee_arg_rel_features_to_dataset,
)
from .dee import (
    DEEExample,
    DEEExampleLoader,
    DEEFeature,
    DEEFeatureConverter,
    convert_dee_features_to_dataset,
)
from .deppn import (
    DEPPNFeature,
    DEPPNFeatureConverter,
    convert_deppn_features_to_dataset,
)
from .ner import (
    NERExample,
    NERFeature,
    NERFeatureConverter,
    convert_ner_features_to_dataset,
)


def render_sentences(
    mspans,
    sentences,
    join_char: Optional[str] = "",
    ent_class: Optional[str] = None,
    ent_type_class: Optional[str] = None,
):
    rendered_sentences = []
    start_pos2mspan = {tuple(x["drange"][:2]): (x["mspan"], x["mtype"]) for x in mspans}
    for sent_idx, sent in enumerate(sentences):
        new_sent = []
        char_idx = 0
        while char_idx < len(sent):
            if (sent_idx, char_idx) in start_pos2mspan:
                ent, ent_type = start_pos2mspan[(sent_idx, char_idx)]
                if ent_class and ent_type_class:
                    span = f'<span class="{ent_class}">{ent}<span class="{ent_type_class}">{ent_type}</span></span>'
                elif ent_class and not ent_type_class:
                    span = f'<span class="{ent_class}">{ent}-{ent_type}</span>'
                elif ent_type_class and not ent_class:
                    span = f'{ent}<span class="{ent_type_class}">{ent_type}</span>'
                else:
                    span = f"{ent}_{{{ent_type}}}"
                new_sent.append(span)
                char_idx += len(ent)
            else:
                new_sent.append(sent[char_idx])
                char_idx += 1
        rendered_sentences.append(join_char.join(new_sent))
    return rendered_sentences


def match_arg(
    sentences: List[str],
    doc_token_ids: np.ndarray,
    arg: Tuple[int],
    offset: Optional[int] = 0,
):
    arg_arr = np.array(arg)
    arg_len = len(arg)
    tokens_ravel = np.ravel(doc_token_ids)
    seq_len = doc_token_ids.shape[1]
    row = col = None
    for i in range(0, tokens_ravel.shape[0] - arg_arr.shape[0] + 1):
        if np.array_equal(arg_arr, tokens_ravel[i : i + arg_len]):
            row = i // seq_len
            col = i - row * seq_len
    if row is not None and col is not None:
        return "".join(sentences[row][col - offset : col + arg_len - offset]), [
            row,
            col - offset,
            col + arg_len - offset,
        ]
    else:
        return None, None


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


def convert_string_to_raw_input(guid, sents: List[str]):
    data = [
        guid,
        {
            "doc_type": "o2o",
            "sentences": sents,
            "ann_valid_mspans": [],
            "ann_valid_dranges": [],
            "ann_mspan2dranges": {},
            "ann_mspan2guess_field": {},
            "recguid_eventname_eventdict_list": [],
        },
    ]
    return data


def prepare_doc_batch_dict(doc_fea_list):
    doc_batch_keys = [
        "ex_idx",
        "doc_type",
        "doc_token_ids",
        "doc_token_masks",
        "doc_token_labels",
        "valid_sent_num",
    ]
    doc_batch_dict = {}
    for key in doc_batch_keys:
        doc_batch_dict[key] = [getattr(doc_fea, key) for doc_fea in doc_fea_list]

    return doc_batch_dict


def measure_dee_prediction(
    event_type_fields_pairs,
    features,
    event_decode_results,
    event_independent,
    dump_json_path=None,
):
    """tzhu comments: get measurements
    Args:
        event_decode_results: [(doc_fea.ex_idx, event_pred_list,
                    event_idx2obj_idx2field_idx2token_tup,
                    doc_span_info, event_idx2event_decode_paths), (), ...]
    """
    is_cg = False
    if len(features) > 0:
        if all(isinstance(feature, DEEArgRelFeature) for feature in features):
            is_cg = True
        elif all(isinstance(feature, DEEFeature) for feature in features):
            is_cg = False
        elif all(isinstance(feature, DEPPNFeature) for feature in features):
            is_cg = False
        else:
            raise ValueError("Not all the features are in the same type!")

    all_results = {}
    # x2m means documents with multi event records (including o2m and m2m)
    for doc_type in ["o2o", "o2m", "m2m", "x2m", "overall"]:
        new_event_decode_results = copy.deepcopy(event_decode_results)
        filtered_event_decode_results = []
        for doc_fea, decode_result in zip(features, new_event_decode_results):
            if (
                doc_type in ["o2o", "o2m", "m2m"]
                and doc_fea.doc_type != {"o2o": 0, "o2m": 1, "m2m": 2}[doc_type]
            ) or (doc_type == "x2m" and doc_fea.doc_type == 0):
                continue
            filtered_event_decode_results.append(decode_result)

        pred_record_mat_list = []
        gold_record_mat_list = []
        pred_event_types = []
        gold_event_types = []
        pred_spans_token_tuple_list = []
        gold_spans_token_tuple_list = []
        pred_adj_mats = []
        gold_adj_mats = []
        pred_combinations = []
        gold_combinations = []
        for term in filtered_event_decode_results:
            ex_idx, pred_event_type_labels, pred_record_mat, doc_span_info = term[:4]
            doc_fea = features[ex_idx]

            if is_cg:
                pred_adj_mat, event_idx2combinations = term[4:6]
                pred_adj_mats.append(pred_adj_mat)
                if event_independent:
                    gold_adj_mats.append(
                        [mat.reveal_adj_mat() for mat in doc_fea.span_rel_mats]
                    )
                else:
                    gold_adj_mats.append([doc_fea.whole_arg_rel_mat.reveal_adj_mat()])
                tmp_pred_combinations = set()
                for combinations in event_idx2combinations:
                    combinations = [
                        tuple(
                            sorted(
                                [doc_span_info.span_token_tup_list[arg] for arg in comb]
                            )
                        )
                        for comb in combinations
                    ]
                    tmp_pred_combinations.update(set(combinations))
                pred_combinations.append(tmp_pred_combinations)
                # convert doc_fea.event_arg_idxs_objs_list and remove the role labels
                doc_fea.event_arg_idxs_objs_list = remove_event_obj_roles(
                    doc_fea.event_arg_idxs_objs_list, event_type_fields_pairs
                )
                tmp_gold_combinations = extract_combinations_from_event_objs(
                    doc_fea.event_arg_idxs_objs_list
                )
                tmp_gold_combinations = set(
                    [
                        tuple(
                            sorted([doc_fea.span_token_ids_list[arg] for arg in comb])
                        )
                        for comb in tmp_gold_combinations
                    ]
                )
                gold_combinations.append(tmp_gold_combinations)

            pred_event_types.append(pred_event_type_labels)
            gold_event_types.append(doc_fea.event_type_labels)
            pred_spans_token_tuple_list.append(doc_span_info.span_token_tup_list)
            gold_spans_token_tuple_list.append(doc_fea.span_token_ids_list)

            pred_record_mat = [
                [
                    [
                        tuple(arg_tup) if arg_tup is not None else None
                        for arg_tup in pred_record
                    ]
                    for pred_record in pred_records
                ]
                if pred_records is not None
                else None
                for pred_records in pred_record_mat
            ]
            gold_record_mat = [
                [
                    [
                        tuple(doc_fea.span_token_ids_list[arg_idx])
                        if arg_idx is not None
                        else None
                        for arg_idx in event_arg_idxs
                    ]
                    for event_arg_idxs in event_arg_idxs_objs
                ]
                if event_arg_idxs_objs is not None
                else None  # for events in each event type
                for event_arg_idxs_objs in doc_fea.event_arg_idxs_objs_list
            ]
            pred_record_mat_list.append(pred_record_mat)
            gold_record_mat_list.append(gold_record_mat)

        if event_independent:
            g_eval_res = measure_event_table_filling(
                pred_record_mat_list,
                gold_record_mat_list,
                event_type_fields_pairs,
                pred_event_types,
                gold_event_types,
                pred_spans_token_tuple_list,
                gold_spans_token_tuple_list,
                pred_adj_mats=None,
                gold_adj_mats=None,
                pred_combinations=None,
                gold_combinations=None,
                dict_return=True,
            )
        else:
            g_eval_res = measure_event_table_filling(
                pred_record_mat_list,
                gold_record_mat_list,
                event_type_fields_pairs,
                pred_event_types,
                gold_event_types,
                pred_spans_token_tuple_list,
                gold_spans_token_tuple_list,
                pred_adj_mats=pred_adj_mats,
                gold_adj_mats=gold_adj_mats,
                pred_combinations=pred_combinations,
                gold_combinations=gold_combinations,
                dict_return=True,
            )
        all_results[doc_type] = g_eval_res

    if dump_json_path is not None:
        default_dump_json(all_results, dump_json_path)

    return all_results


def aggregate_task_eval_info(
    eval_dir_path,
    max_epoch=100,
    target_file_pre="dee_eval",
    target_file_suffix=".json",
    dump_name="total_task_eval.pkl",
    dump_flag=False,
):
    """Enumerate the evaluation directory to collect all dumped evaluation results"""
    logger.info("Aggregate task evaluation info from {}".format(eval_dir_path))
    doc_type2data_span_type2model_str2epoch_res_list = {}
    for fn in os.listdir(eval_dir_path):
        fn_splits = fn.split(".")
        if (
            fn.startswith(target_file_pre)
            and fn.endswith(target_file_suffix)
            and len(fn_splits) == 6
        ):
            _, data_type, span_type, model_str, epoch, _ = fn_splits
            epoch = int(epoch)
            if epoch > max_epoch:
                continue

            fp = os.path.join(eval_dir_path, fn)
            eval_res = default_load_json(fp)

            for doc_type in ["o2o", "o2m", "m2m", "overall"]:
                if doc_type not in doc_type2data_span_type2model_str2epoch_res_list:
                    doc_type2data_span_type2model_str2epoch_res_list[doc_type] = {}

                data_span_type = (data_type, span_type)
                if (
                    data_span_type
                    not in doc_type2data_span_type2model_str2epoch_res_list[doc_type]
                ):
                    doc_type2data_span_type2model_str2epoch_res_list[doc_type][
                        data_span_type
                    ] = {}
                model_str2epoch_res_list = (
                    doc_type2data_span_type2model_str2epoch_res_list[doc_type][
                        data_span_type
                    ]
                )

                if model_str not in model_str2epoch_res_list:
                    model_str2epoch_res_list[model_str] = []
                epoch_res_list = model_str2epoch_res_list[model_str]

                epoch_res_list.append((epoch, eval_res[doc_type]))

    for (
        doc_type,
        data_span_type2model_str2epoch_res_list,
    ) in doc_type2data_span_type2model_str2epoch_res_list.items():
        for (
            data_span_type,
            model_str2epoch_res_list,
        ) in data_span_type2model_str2epoch_res_list.items():
            for model_str, epoch_res_list in model_str2epoch_res_list.items():
                epoch_res_list.sort(key=lambda x: x[0])

    if dump_flag:
        dump_fp = os.path.join(eval_dir_path, dump_name)
        logger.info("Dumping {} into {}".format(dump_name, eval_dir_path))
        default_dump_pkl(doc_type2data_span_type2model_str2epoch_res_list, dump_fp)

    return doc_type2data_span_type2model_str2epoch_res_list


def print_total_eval_info(
    doc_type2data_span_type2model_str2epoch_res_list,
    template,
    doc_type="overall",
    metric_type="micro",
    span_type="pred_span",
    model_strs=("DCFEE-O", "DCFEE-M", "GreedyDec", "Doc2EDAG", "LSTMMTL"),
    target_set="test",
    part="overall",
):
    """Print the final performance by selecting the best epoch on dev set and emitting performance on test set

    Args:
        part: ``overall``, ``classification`` or ``entity``
    """
    dev_type = "dev"
    test_type = "test"
    avg_type2prf1_keys = {
        "macro": ("MacroPrecision", "MacroRecall", "MacroF1"),
        "micro": ("MicroPrecision", "MicroRecall", "MicroF1"),
    }

    name_key = "EventType"
    p_key, r_key, f_key = avg_type2prf1_keys[metric_type]
    total_results = []

    def get_avg_event_score(epoch_res):
        eval_res = epoch_res[1][part]
        avg_event_score = eval_res[f_key]
        return avg_event_score

    dev_model_str2epoch_res_list = doc_type2data_span_type2model_str2epoch_res_list[
        doc_type
    ][(dev_type, span_type)]
    test_model_str2epoch_res_list = doc_type2data_span_type2model_str2epoch_res_list[
        doc_type
    ][(test_type, span_type)]

    has_header = False
    mstr_bepoch_list = []
    logger.info(
        f"{'-'*15} doc_type: {doc_type}, part: {part}, span_type: {span_type} {'-'*15}"
    )
    logger.info(
        "=" * 15, "Final Performance (%) (avg_type={})".format(metric_type), "=" * 15
    )
    for model_str in model_strs:
        if (
            model_str not in dev_model_str2epoch_res_list
            or model_str not in test_model_str2epoch_res_list
        ):
            continue

        tmp_results = {
            "ModelType": model_str,
            "Average": {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
            "Total": {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
        }
        for event_type in template.event_type2event_class:
            tmp_results[event_type] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }
        # get the best epoch on dev set
        dev_epoch_res_list = dev_model_str2epoch_res_list[model_str]
        best_dev_epoch, best_dev_res = max(dev_epoch_res_list, key=get_avg_event_score)
        test_epoch_res_list = test_model_str2epoch_res_list[model_str]
        best_test_epoch = None
        best_test_res = None
        for test_epoch, test_res in test_epoch_res_list:
            if test_epoch == best_dev_epoch:
                best_test_epoch = test_epoch
                best_test_res = test_res
        assert best_test_epoch is not None
        mstr_bepoch_list.append((model_str, best_test_epoch))

        if target_set == "test":
            target_eval_res = best_test_res
        else:
            target_eval_res = best_dev_res
        target_eval_res = target_eval_res[part]

        align_temp = "{:20}"
        head_str = align_temp.format("ModelType")
        eval_str = align_temp.format(model_str)
        head_temp = " \t {}"
        eval_temp = " \t & {:.1f} & {:.1f} & {:.1f}"
        ps = []
        rs = []
        fs = []
        for tgt_event_res in target_eval_res["Events"]:
            head_str += align_temp.format(head_temp.format(tgt_event_res[0][name_key]))
            p, r, f1 = (100 * tgt_event_res[0][key] for key in [p_key, r_key, f_key])
            eval_str += align_temp.format(eval_temp.format(p, r, f1))
            ps.append(p)
            rs.append(r)
            fs.append(f1)
            tmp_results[tgt_event_res[0][name_key]]["precision"] = "{:.1f}".format(p)
            tmp_results[tgt_event_res[0][name_key]]["recall"] = "{:.1f}".format(r)
            tmp_results[tgt_event_res[0][name_key]]["f1"] = "{:.1f}".format(f1)

        head_str += align_temp.format(head_temp.format("Average"))
        ap, ar, af1 = (x for x in [np.mean(ps), np.mean(rs), np.mean(fs)])
        eval_str += align_temp.format(eval_temp.format(ap, ar, af1))
        tmp_results["Average"]["precision"] = "{:.1f}".format(ap)
        tmp_results["Average"]["recall"] = "{:.1f}".format(ar)
        tmp_results["Average"]["f1"] = "{:.1f}".format(af1)

        head_str += align_temp.format(
            head_temp.format("Total ({})".format(metric_type))
        )
        g_avg_res = target_eval_res
        ap, ar, af1 = (100 * g_avg_res[key] for key in [p_key, r_key, f_key])
        eval_str += align_temp.format(eval_temp.format(ap, ar, af1))
        tmp_results["Total"]["precision"] = "{:.1f}".format(ap)
        tmp_results["Total"]["recall"] = "{:.1f}".format(ar)
        tmp_results["Total"]["f1"] = "{:.1f}".format(af1)

        total_results.append(tmp_results)

        if not has_header:
            logger.info(head_str)
            has_header = True
        logger.info(eval_str)

    return mstr_bepoch_list, total_results


# evaluation dump file name template
# dee_eval~~.[DocType: o2o|o2m|m2m|overall]~~.[DataType].[SpanType].[ModelStr].[Epoch].(pkl|json)
decode_dump_template = "dee_eval.{}.{}.{}.{}.pkl"
eval_dump_template = "dee_eval.{}.{}.{}.{}.json"


def resume_decode_results(base_dir, data_type, span_type, model_str, epoch):
    decode_fn = decode_dump_template.format(data_type, span_type, model_str, epoch)
    decode_fp = os.path.join(base_dir, decode_fn)
    logger.info("Resume decoded results from {}".format(decode_fp))
    decode_results = default_load_pkl(decode_fp)

    return decode_results


def resume_eval_results(base_dir, data_type, span_type, model_str, epoch):
    eval_fn = eval_dump_template.format(data_type, span_type, model_str, epoch)
    eval_fp = os.path.join(base_dir, eval_fn)
    logger.info("Resume eval results from {}".format(eval_fp))
    eval_results = default_load_json(eval_fp)

    return eval_results


def print_single_vs_multi_performance(
    mstr_bepoch_list,
    base_dir,
    features,
    template,
    event_independent,
    metric_type="micro",
    data_type="test",
    span_type="pred_span",
    part="overall",
):
    """
    Args:
        part: classification, entity, combination, rawCombination, overall
    """
    model_str2decode_results = {}
    for model_str, best_epoch in mstr_bepoch_list:
        model_str2decode_results[model_str] = resume_decode_results(
            base_dir, data_type, span_type, model_str, best_epoch
        )

    single_eid_set = set(
        [doc_fea.ex_idx for doc_fea in features if not doc_fea.is_multi_event()]
    )
    multi_eid_set = set(
        [doc_fea.ex_idx for doc_fea in features if doc_fea.is_multi_event()]
    )
    event_type_fields_pairs = DEEExample.get_event_type_fields_pairs(template)
    event_type_list = [x[0] for x in event_type_fields_pairs]

    name_key = "EventType"
    avg_type2f1_key = {
        "micro": "MicroF1",
        "macro": "MacroF1",
    }
    f1_key = avg_type2f1_key[metric_type]
    sm_results = []

    model_str2etype_sf1_mf1_list = {}
    for model_str, _ in mstr_bepoch_list:
        total_decode_results = model_str2decode_results[model_str]

        single_decode_results = [
            dec_res for dec_res in total_decode_results if dec_res[0] in single_eid_set
        ]
        assert len(single_decode_results) == len(single_eid_set)
        single_eval_res = measure_dee_prediction(
            event_type_fields_pairs, features, single_decode_results, event_independent
        )["overall"]

        multi_decode_results = [
            dec_res for dec_res in total_decode_results if dec_res[0] in multi_eid_set
        ]
        assert len(multi_decode_results) == len(multi_eid_set)
        multi_eval_res = measure_dee_prediction(
            event_type_fields_pairs, features, multi_decode_results, event_independent
        )["overall"]

        etype_sf1_mf1_list = []
        for event_idx, (se_res, me_res) in enumerate(
            zip(single_eval_res[part]["Events"], multi_eval_res[part]["Events"])
        ):
            assert (
                se_res[0][name_key] == me_res[0][name_key] == event_type_list[event_idx]
            )
            event_type = event_type_list[event_idx]
            single_f1 = se_res[0][f1_key]
            multi_f1 = me_res[0][f1_key]

            etype_sf1_mf1_list.append((event_type, single_f1, multi_f1))
        g_avg_se_res = single_eval_res[part]
        g_avg_me_res = multi_eval_res[part]
        etype_sf1_mf1_list.append(
            (
                "Total ({})".format(metric_type),
                g_avg_se_res[f1_key],
                g_avg_me_res[f1_key],
            )
        )
        model_str2etype_sf1_mf1_list[model_str] = etype_sf1_mf1_list

    logger.info(
        "=" * 15, "Single vs. Multi (%) (avg_type={})".format(metric_type), "=" * 15
    )
    align_temp = "{:20}"
    head_str = align_temp.format("ModelType")
    head_temp = " \t {}"
    eval_temp = " \t & {:.1f} & {:.1f} "
    for event_type in event_type_list:
        head_str += align_temp.format(head_temp.format(event_type))
    head_str += align_temp.format(head_temp.format("Average"))
    head_str += align_temp.format(head_temp.format("Total ({})".format(metric_type)))
    logger.info(head_str)

    for model_str, _ in mstr_bepoch_list:
        tmp_results = {
            "ModelType": model_str,
            "Average": {"Single": 0.0, "Multi": 0.0},
            "Total": {"Single": 0.0, "Multi": 0.0},
        }
        for event_type in template.event_type2event_class:
            tmp_results[event_type] = {"Single": 0.0, "Multi": 0.0}
        eval_str = align_temp.format(model_str)
        sf1s = []
        mf1s = []
        for event_type, single_f1, multi_f1 in model_str2etype_sf1_mf1_list[model_str]:
            eval_str += align_temp.format(
                eval_temp.format(single_f1 * 100, multi_f1 * 100)
            )
            sf1s.append(single_f1)
            mf1s.append(multi_f1)
            if "Total" in event_type:
                event_type = "Total"
            tmp_results[event_type]["Single"] = "{:.1f}".format(single_f1 * 100)
            tmp_results[event_type]["Multi"] = "{:.1f}".format(multi_f1 * 100)
        avg_sf1 = np.mean(sf1s[:-1])
        avg_mf1 = np.mean(mf1s[:-1])
        tmp_results["Average"]["Single"] = "{:.1f}".format(avg_sf1 * 100)
        tmp_results["Average"]["Multi"] = "{:.1f}".format(avg_mf1 * 100)
        eval_str += align_temp.format(eval_temp.format(avg_sf1 * 100, avg_mf1 * 100))
        sm_results.append(tmp_results)
        new_eval_str = align_temp.format(model_str)
        for event_type in event_type_list + ["Average", "Total"]:
            single = tmp_results[event_type]["Single"]
            multi = tmp_results[event_type]["Multi"]
            new_eval_str += align_temp.format(" \t & {} & {} ".format(single, multi))
        logger.info(new_eval_str)

    return sm_results


def print_ablation_study(
    mstr_bepoch_list,
    base_dir,
    base_mstr,
    other_mstrs,
    template,
    doc_type="overall",
    metric_type="micro",
    data_type="test",
    span_type="pred_span",
):
    model_str2best_epoch = dict(mstr_bepoch_list)
    if base_mstr not in model_str2best_epoch:
        logger.info("No base model type {}".format(base_mstr))
        return

    logger.info(f"{'-'*15} doc_type: {doc_type} {'-'*15}")
    base_eval = resume_eval_results(
        base_dir, data_type, span_type, base_mstr, model_str2best_epoch[base_mstr]
    )[doc_type]
    model_str2eval_res = {
        model_str: resume_eval_results(
            base_dir, data_type, span_type, model_str, model_str2best_epoch[model_str]
        )[doc_type]
        for model_str in other_mstrs
        if model_str in model_str2best_epoch
    }

    event_type_fields_pairs = DEEExample.get_event_type_fields_pairs(template)
    event_type_list = [x[0] for x in event_type_fields_pairs]
    # name_key = 'EventType'
    # f1_key = 'AvgFieldF1'
    avg_type2f1_key = {"micro": "MicroF1", "macro": "MacroF1"}
    f1_key = avg_type2f1_key[metric_type]

    logger.info("=" * 15, "Ablation Study (avg_type={})".format(metric_type), "=" * 15)
    align_temp = "{:20}"
    head_str = align_temp.format("ModelType")
    head_temp = " \t {}"
    for event_type in event_type_list:
        head_str += align_temp.format(head_temp.format(event_type))
    head_str += align_temp.format(head_temp.format("Average ({})".format(metric_type)))
    head_str += align_temp.format(head_temp.format("Average"))
    logger.info(head_str)

    eval_temp = " \t & {:.1f}"
    eval_str = align_temp.format(base_mstr)
    bf1s = []
    for base_event_res in base_eval[:-1]:
        base_f1 = base_event_res[0][f1_key]
        eval_str += align_temp.format(eval_temp.format(base_f1 * 100))
        bf1s.append(base_f1)
    g_avg_bf1 = base_eval[-1][f1_key]
    eval_str += align_temp.format(eval_temp.format(g_avg_bf1 * 100))
    avg_bf1 = np.mean(bf1s)
    eval_str += align_temp.format(eval_temp.format(avg_bf1 * 100))
    logger.info(eval_str)

    inc_temp = " \t & +{:.1f}"
    dec_temp = " \t & -{:.1f}"
    for model_str in other_mstrs:
        if model_str in model_str2eval_res:
            eval_str = align_temp.format(model_str)
            cur_eval = model_str2eval_res[model_str]
            f1ds = []
            for base_event_res, cur_event_res in zip(base_eval[:-1], cur_eval[:-1]):
                base_f1 = base_event_res[0][f1_key]
                cur_f1 = cur_event_res[0][f1_key]
                f1_diff = cur_f1 - base_f1
                f1ds.append(f1_diff)
                f1_abs = abs(f1_diff)
                if f1_diff >= 0:
                    eval_str += align_temp.format(inc_temp.format(f1_abs * 100))
                else:
                    eval_str += align_temp.format(dec_temp.format(f1_abs * 100))

            g_avg_f1_diff = cur_eval[-1][f1_key] - base_eval[-1][f1_key]
            g_avg_f1_abs = abs(g_avg_f1_diff)
            if g_avg_f1_diff >= 0:
                eval_str += align_temp.format(inc_temp.format(g_avg_f1_abs * 100))
            else:
                eval_str += align_temp.format(dec_temp.format(g_avg_f1_abs * 100))

            avg_f1_diff = np.mean(f1ds)
            avg_f1_abs = abs(avg_f1_diff)
            if avg_f1_diff >= 0:
                eval_str += align_temp.format(inc_temp.format(avg_f1_abs * 100))
            else:
                eval_str += align_temp.format(dec_temp.format(avg_f1_abs * 100))

            logger.info(eval_str)
