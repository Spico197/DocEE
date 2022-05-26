import functools
import importlib
import itertools
import json
import logging
import math
import os
import pickle
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib as mpl
import networkx as nx
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from loguru import logger  # noqa: E402
from torch.optim.lr_scheduler import LambdaLR  # noqa: E402
from torch.optim.optimizer import Optimizer  # noqa: E402
from transformers import BertTokenizer  # noqa: E402

mpl.rcParams["font.sans-serif"] = ["SimHei"]  # 指定默认字体
mpl.rcParams["axes.unicode_minus"] = False  # 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams["axes.titlesize"] = 20

EPS = 1e-10


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def default_load_json(json_file_path, encoding="utf-8", **kwargs):
    with open(json_file_path, "r", encoding=encoding) as fin:
        tmp_json = json.load(fin, **kwargs)
    return tmp_json


def default_dump_json(
    obj, json_file_path, encoding="utf-8", ensure_ascii=False, indent=2, **kwargs
):
    with open(json_file_path, "w", encoding=encoding) as fout:
        json.dump(obj, fout, ensure_ascii=ensure_ascii, indent=indent, **kwargs)


def default_load_pkl(pkl_file_path, **kwargs):
    with open(pkl_file_path, "rb") as fin:
        obj = pickle.load(fin, **kwargs)

    return obj


def default_dump_pkl(obj, pkl_file_path, **kwargs):
    with open(pkl_file_path, "wb") as fout:
        pickle.dump(obj, fout, **kwargs)


def set_basic_log_config():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


class BertTokenizerForDocEE(BertTokenizer):
    """Customized tokenizer"""

    def __init__(
        self,
        vocab_file,
        doc_lang="zh",
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )
        self.doc_lang = doc_lang

        if self.doc_lang == "zh":
            self.dee_tokenize = self.dee_char_tokenize
        elif self.doc_lang == "en":
            self.dee_tokenize = self.dee_space_tokenize

    def dee_space_tokenize(self, text):
        """perform space tokenization"""
        tokens = text.split()
        out_tokens = []
        for token in tokens:
            if token in self.vocab:
                out_tokens.append(token)
            else:
                out_tokens.append(self.unk_token)

        return out_tokens

    def dee_wordpiece_tokenize(self, text):
        """perform wordpiece tokenization"""
        tokens = text.split()
        out_tokens = []
        for token in tokens:
            pieces = self.tokenize(token)
            if len(pieces) < 1:
                pieces = [self.unk_token]
            out_tokens += pieces

        return out_tokens

    def dee_char_tokenize(self, text):
        """perform pure character-based tokenization"""
        tokens = list(text)
        out_tokens = []
        for token in tokens:
            if token in self.vocab:
                out_tokens.append(token)
            else:
                out_tokens.append(self.unk_token)

        return out_tokens


def recursive_print_grad_fn(grad_fn, prefix="", depth=0, max_depth=50):
    if depth > max_depth:
        return
    logger.info(prefix, depth, grad_fn.__class__.__name__)
    if hasattr(grad_fn, "next_functions"):
        for nf in grad_fn.next_functions:
            ngfn = nf[0]
            recursive_print_grad_fn(
                ngfn, prefix=prefix + "  ", depth=depth + 1, max_depth=max_depth
            )


def strtobool(str_val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    str_val = str_val.lower()
    if str_val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif str_val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError("invalid truth value %r" % (str_val,))


def plot_graph_from_adj_mat(adj_mat, directory=".", title="No Title"):
    fig = plt.figure(figsize=(16, 9), dpi=350)
    adj_mat = np.array(adj_mat)
    np.fill_diagonal(adj_mat, 0)
    rows, cols = np.where(adj_mat == 1)
    edges = zip(rows.tolist(), cols.tolist())

    G = nx.Graph()
    G.add_edges_from(edges)

    options = {
        "font_size": 36,
        "node_size": 3000,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 5,
        "width": 5,
    }
    nx.draw_networkx(G, **options)

    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    path = os.path.join(directory, f"{title}.png")
    fig.savefig(path, format="png")
    plt.close()


def extract_combinations_from_event_objs(event_objs):
    combinations = set()
    for events in event_objs:
        if events is not None:
            for instance in events:
                combination = set()
                for arg in instance:
                    if arg is not None:
                        combination.add(arg)
                if len(combination) > 0:
                    combinations.add(tuple(sorted(list(combination))))
    return combinations


def extract_instances_from_event_objs(event_objs):
    """has a role type in the final combination compared with `extract_combinations_from_event_objs`"""
    instances = set()
    for event_type_idx, events in enumerate(event_objs):
        if events is not None:
            for instance in events:
                combination = set()
                for role, arg in enumerate(instance):
                    if arg is not None:
                        combination.add((arg, role))
                if len(combination) > 0:
                    instances.add((event_type_idx,) + tuple(sorted(list(combination))))
    return instances


def remove_combination_roles(combinations):
    ret_combs = set()
    for comb in combinations:
        if isinstance(comb[0], int):
            ret_combs.add(comb)
            continue
        new_comb = set()
        for arg_role in comb:
            if len(arg_role) == 2:
                arg, _ = arg_role
            else:
                arg = arg_role
            if arg is not None:
                new_comb.add(arg)
        new_comb = sorted(list(new_comb))
        ret_combs.add(tuple(new_comb))
    return ret_combs


def contain_role_type(event_instance):
    return any(arg is not None and isinstance(arg, tuple) for arg in event_instance)


def remove_event_obj_roles(event_objs_list, event_type_fields_pairs):
    result_event_arg_idxs_objs_list = []
    for event_idx, events in enumerate(event_objs_list):
        if events is None:
            result_event_arg_idxs_objs_list.append(None)
            continue
        tmp_events = []
        for event in events:
            # if the event_arg_idxs_objs_list has already been fixed, then pass
            if not contain_role_type(event):
                tmp_events.append(event)
                continue
            tmp_span_idxs = [
                None for _ in range(len(event_type_fields_pairs[event_idx][1]))
            ]
            for span, field in event:
                tmp_span_idxs[field] = span
            tmp_events.append(tuple(tmp_span_idxs))
        result_event_arg_idxs_objs_list.append(tmp_events)
    return result_event_arg_idxs_objs_list


def negative_sampling(gold_combinations, len_spans):
    negative_combination = set(list(range(len_spans)))
    for gc in gold_combinations:
        negative_combination = negative_combination - set(gc)

    if len(negative_combination) > 0:
        return tuple(sorted(list(negative_combination)))
    else:
        return None


def random_sampling(
    whole_arg_rel_mat, len_spans, min_num_span=2, num_samp=5, max_samp_times=20
):
    """
    random sampling part of the whole combination graph

    Returns:
        [[combination (List), adj matrix (List[List])], [...], ...]
    """
    ret_pairs = []

    combinations = []
    for _ in range(max_samp_times):
        if len(combinations) >= num_samp:
            break
        tmp_combination = []
        for i in range(len_spans):
            if random.random() >= 0.5:
                tmp_combination.append(i)
        if len(tmp_combination) >= min_num_span:
            combinations.append(tmp_combination)

    for combination in combinations[:num_samp]:
        adj_mat = whole_arg_rel_mat.get_sub_graph_adj_mat(combination)
        ret_pairs.append([combination, adj_mat])

    return ret_pairs


def fill_diag(mat, num):
    for i in range(len(mat)):
        mat[i][i] = num
    return mat


def fold_and(mat):
    r"""
    mat[j, i] = mat[i, j] = 1 iff mat[i, j] == mat[j, i] == 1
    """
    new_mat = [[0] * len(mat[0]) for _ in range(len(mat))]
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] == mat[j][i] == 1:
                new_mat[i][j] = new_mat[j][i] = 1
    return new_mat


def left_tril(mat):
    new_mat = np.array(mat)
    new_mat = np.tril(new_mat, k=-1)
    return new_mat.tolist()


def tril_fold_or(mat):
    new_mat = np.array(mat)
    new_mat = np.tril(new_mat, k=-1)
    new_mat = np.bitwise_or(new_mat, new_mat.T)
    return new_mat.tolist()


def assign_role_from_gold_to_comb(comb, gold_comb):
    r"""
    pass the roles in gold combination to pred combination
    role will be `None` if there's no such mapping

    Returns:
        [(0, {1, 2}), (1, None)]
    """
    span_idx2roles = defaultdict(set)
    for span_idx, role in gold_comb:
        span_idx2roles[span_idx].add(role)
    new_comb = []
    for span_idx in comb:
        new_comb.append((span_idx, span_idx2roles.get(span_idx, None)))
    return new_comb


def reveal_event_obj_from_comb_instance(comb_instance, num_fields):
    r"""
    from role-assgined comb to event obj
    """
    ret_results = [None] * num_fields
    for span_idx, roles in comb_instance:
        if roles is None:
            continue
        for role_idx in roles:
            ret_results[role_idx] = span_idx
    return ret_results


def closest_match(comb, gold_combs):
    r"""
    get the closest combination with intersection match

    Returns:
        combination
        similarity: 0 if there's no matched span
    """
    idx2match = []
    for gold_comb_idx, gold_comb in enumerate(gold_combs):
        num_match = 0
        if isinstance(gold_comb[0], tuple):
            num_match = len(set(comb) & set(span_idx[0] for span_idx in gold_comb))
        else:
            num_match = len(set(comb) & set(span_idx for span_idx in gold_comb))
        idx2match.append((gold_comb_idx, num_match))
    idx2match.sort(key=lambda x: x[1], reverse=True)
    return gold_combs[idx2match[0][0]], idx2match[0][1]


def recover_ins(event_type_fields_list, convert_ids_to_tokens_func, record_mat):
    inses = []
    for event_idx, events in enumerate(record_mat):
        if events is not None:
            for ins in events:
                tmp_ins = {
                    "EventType": event_type_fields_list[event_idx][0],
                    "Arguments": {
                        event_type_fields_list[event_idx][1][field_idx]: "".join(
                            convert_ids_to_tokens_func(arg)
                        )
                        if arg is not None
                        else None
                        for field_idx, arg in enumerate(ins)
                    },
                }
                inses.append(tmp_ins)
    return inses


def convert_role_fea_event_obj_to_standard(event_type_fields_list, event_objs):
    new_event_objs = []
    for event_idx, events in enumerate(event_objs):
        if events is None:
            new_event_objs.append(None)
            continue
        num_fields = len(event_type_fields_list[event_idx][1])
        new_inses = []
        for ins in events:
            tmp_ins = [None for _ in range(num_fields)]
            for arg in ins:
                span_idx, field_idx = arg
                tmp_ins[field_idx] = span_idx
            new_inses.append(tmp_ins)
        new_event_objs.append(new_inses)
    return new_event_objs


def list_models():
    models = dir(importlib.import_module("dee.models"))
    models = list(filter(lambda x: x[0].upper() == x[0] and x[0] != "_", models))
    return models


def merge_non_conflicting_ins_objs(instances, min_coo=1):
    final = []
    final_merged = []
    merged = set()
    for ins1, ins2 in itertools.combinations(instances, 2):
        mergable_values = []
        coo = 0
        for field1, field2 in zip(ins1, ins2):
            if field1 is None or field2 is None:
                mergable_values.append(True)
                continue
            if field1 == field2:
                coo += 1
                mergable_values.append(True)
                continue
            mergable_values.append(False)

        if all(mergable_values) and coo >= min_coo:
            # mergable
            new_obj = []
            for field1, field2 in zip(ins1, ins2):
                merged_field = None
                if field1 is None:
                    merged_field = field2
                elif field2 is None:
                    merged_field = field1
                else:
                    # or field2 (here, field1 == field2)
                    merged_field = field1
                new_obj.append(merged_field)
            final_merged.append(new_obj)
            merged.add(tuple(ins1))
            merged.add(tuple(ins2))

    for ins in instances:
        if tuple(ins) not in merged:
            final.append(ins)
    return final + final_merged


def list_flatten(lst):
    total_list = []
    len_mapping = []
    for idx, elements in enumerate(lst):
        len_mapping += [[idx, i] for i in range(len(elements))]
        total_list += elements
    return total_list, len_mapping


class RegexEntExtractor(object):
    def __init__(self) -> None:
        self.field2type = {
            # shares
            "TotalHoldingShares": "share",
            "TotalPledgedShares": "share",
            "PledgedShares": "share",
            "FrozeShares": "share",
            "RepurchasedShares": "share",
            "TradedShares": "share",
            "LaterHoldingShares": "share",
            # ratio
            "TotalHoldingRatio": "ratio",
            # date
            "StartDate": "date",
            "ReleasedDate": "date",
            "EndDate": "date",
            "ClosingDate": "date",
            "UnfrozeDate": "date",
            # money
            "RepurchaseAmount": "money",
            "HighestTradingPrice": "money",
            "LowestTradingPrice": "money",
            "AveragePrice": "money",
            # shares
            "质押股票/股份数量": "share",
            "回购股份数量": "share",
            "交易股票/股份数量": "share",
            # ratio
            "质押物占总股比": "ratio",
            "质押物占持股比": "ratio",
            "占公司总股本比例": "ratio",
            "增持部分占总股本比例": "ratio",
            "增持部分占所持比例": "ratio",
            "减持部分占总股本比例": "ratio",
            "减持部分占所持比例": "ratio",
            # date
            "披露时间": "date",
            "披露日期": "date",
            "中标日期": "date",
            "事件时间": "date",
            "回购完成时间": "date",
            "被约谈时间": "date",
            "收购完成时间": "date",
            "交易完成时间": "date",
            "破产时间": "date",
            # money
            "每股交易价格": "money",
            "交易金额": "money",
            "募资金额": "money",
            "发行价格": "money",
            "市值": "money",
            "融资金额": "money",
            "净亏损": "money",
        }
        self.field_id2field_name = {}
        self.basic_type_id = None  # id of `O` label
        self.type2func = {
            "share": self.extract_share,
            "ratio": self.extract_ratio,
            "date": self.extract_date,
            "money": self.extract_money,
        }

    @classmethod
    def _extract(cls, regex, text, group=0):
        results = []
        matches = re.finditer(regex, text)
        for match in matches:
            results.append([match.group(group), match.span(group)])
        return results

    @classmethod
    def extract_share(cls, text):
        regex = r"(\d+股)[^票]"
        results = cls._extract(regex, text, group=1)
        return results

    @classmethod
    def extract_ratio(cls, text):
        regex = r"\d+(\.\d+)?%"
        results = cls._extract(regex, text)
        return results

    @classmethod
    def extract_date(cls, text):
        regex = r"\d{4}年\d{1,2}月\d{1,2}日"
        results = cls._extract(regex, text)
        return results

    @classmethod
    def extract_money(cls, text):
        regex = r"\d+(\.\d+)?元"
        results = cls._extract(regex, text)
        return results

    def extract(self, text):
        r"""
        extract ents from one sentence

        Returns:
            {
                "ratio": [[ent, (start pos, end pos)], ...],
                ...
            }
        """
        field2results = defaultdict(list)
        for field, func in self.type2func.items():
            results = func(text)
            if len(results) > 0:
                field2results[field].extend(results)
        return field2results

    def extract_doc(
        self, doc: List[str], exclude_ents: Optional[List[str]] = []
    ) -> Dict[str, List]:
        r"""
        extract ents from the whole document (multiple lines)

        Returns:
            {
                "ratio": [[ent, (sentence idx, start pos, end pos)], ...],
                ...
            }
        """
        field2results = defaultdict(list)
        for sent_idx, line in enumerate(doc):
            results = self.extract(line)
            for field, fr in results.items():
                for match_text, match_span in fr:
                    if match_text not in exclude_ents:
                        field2results[field].append(
                            [match_text, [sent_idx, match_span[0], match_span[1]]]
                        )
        return field2results


regex_extractor = RegexEntExtractor()


def chain_prod(num_list: List):
    return functools.reduce(lambda x, y: x * y, num_list)
