import json
import pprint
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from dee.helper.dee import DEEExample
from dee.helper.ner import NERExample, NERFeatureConverter
from dee.modules import adj_decoding
from dee.utils import (
    default_dump_json,
    extract_combinations_from_event_objs,
    regex_extractor,
    remove_event_obj_roles,
)


def build_span_rel_mat(event_arg_idx_objs_list, len_spans):
    span_rel_mat = []
    for event_idx, events in enumerate(event_arg_idx_objs_list):
        if events is None:
            span_rel_mat.append(None)
            continue
        rel_mat = [[0] * len_spans for _ in range(len_spans)]
        for event in events:
            for arg_field_i in event:
                for arg_field_j in event:
                    if arg_field_i != arg_field_j:
                        rel_mat[arg_field_i][arg_field_j] = rel_mat[arg_field_j][
                            arg_field_i
                        ] = 1
        span_rel_mat.append(rel_mat)
    return span_rel_mat


def build_span_rel_connection_for_each_event(event_arg_idx_objs, len_spans):
    if event_arg_idx_objs is None or len(event_arg_idx_objs) == 0:
        return None
    connections = {span_idx: set() for span_idx in range(len_spans)}
    for event_args in event_arg_idx_objs:
        args = set([x[0] if isinstance(x, tuple) else x for x in event_args])
        if None in args:
            args.remove(None)
        for arg1 in args:
            for arg2 in args:
                if arg1 != arg2:
                    connections[arg1].add(arg2)
    return connections


def build_span_rel_connections(event_arg_idx_objs_list, len_spans):
    connections = []
    for event_idx, events in enumerate(event_arg_idx_objs_list):
        connections.append(build_span_rel_connection_for_each_event(events, len_spans))
    return connections


def build_span_rel_adj_mat_from_connection(connection):
    rel_mat = [[0] * len(connection) for _ in range(len(connection))]
    for arg, connected_args in connection.items():
        for connected_arg in connected_args:
            rel_mat[arg][connected_arg] = 1
    return rel_mat


class SpanRelAdjMat(object):
    """
    Relation Matrix for each Event

    Args:
        event_arg_idx_objs: list of span idxes in each event
            (event-relevant or whole event objs)
        len_spans: the number of spans
        whole_graph: whether to build the whole graph from the event objs
    """

    def __init__(self, event_arg_idx_objs, len_spans, whole_graph=False):
        # num of spans
        len_spans = int(len_spans)
        self.len_spans = len_spans

        # half of the adj matrix to save memory storage
        # [1,2], [1,3], [1,4], [2,3], [2,4], [3,4]
        self.rel_mat = [0] * int((len_spans * (len_spans - 1) / 2))

        # fill in the rel_mat
        if event_arg_idx_objs is not None:
            if whole_graph:
                for events in event_arg_idx_objs:
                    if events is not None:
                        self.build_rel_mat(events)
            else:
                self.build_rel_mat(event_arg_idx_objs)

    def build_rel_mat(self, event_arg_idx_objs):
        connections = build_span_rel_connection_for_each_event(
            event_arg_idx_objs, self.len_spans
        )
        for arg1, connected_args in connections.items():
            for arg2 in connected_args:
                self[arg1, arg2] = 1

    def linear_index2array_index(self, index):
        if index == 0 and len(self.rel_mat) == 0:
            return 0, 0
        elif index == 0 and self.len_spans == 2:
            return 0, 1
        x, y = 0, 0
        start, end = 0, self.len_spans - 1
        while start != end:
            if start <= index < end:
                y = index - start + x + 1
                break
            else:
                start = end
                end = start + self.len_spans - x - 2
                x += 1
        if start == end:
            y = x + 1
        return x, y

    def array_index2linear_index(self, x, y):
        assert x != y
        x, y = min(x, y), max(x, y)
        accum = ((self.len_spans - x) + (self.len_spans - 1)) * x / 2
        left = y - (x + 1)
        index = int(accum + left)
        return index

    def check_index(self, index):
        if isinstance(index, tuple):
            assert len(index) == 2
            assert 0 <= index[0] < self.len_spans
            assert 0 <= index[1] < self.len_spans
        else:
            if len(self.rel_mat) == 0:
                assert index == 0
            else:
                assert 0 <= index < len(self.rel_mat)

    def __getitem__(self, index):
        self.check_index(index)
        if isinstance(index, tuple):
            if index[0] == index[1]:
                return -1  # diag line
            index = self.array_index2linear_index(*index)
        if index == 0 and len(self.rel_mat) == 0:
            return -1
        return self.rel_mat[index]

    def __setitem__(self, index, value):
        self.check_index(index)
        if isinstance(index, tuple):
            if index[0] == index[1]:
                pass  # diag line
            index = self.array_index2linear_index(*index)
        if index == 0 and len(self.rel_mat) == 0:
            pass
        self.rel_mat[index] = value

    def reveal_adj_mat(self):
        adj_mat = [[0] * self.len_spans for _ in range(self.len_spans)]
        for index, value in enumerate(self.rel_mat):
            x, y = self.linear_index2array_index(index)
            adj_mat[x][y] = value
            adj_mat[y][x] = value
        for span_idx in range(self.len_spans):
            adj_mat[span_idx][span_idx] = -1
        return adj_mat

    def get_sub_graph_adj_mat(self, combination):
        """
        get sub-graph and returns the adjacent matrix

        Returns:
            List[List]
        """
        len_comb = len(set(combination))

        for span_idx in combination:
            if span_idx >= self.len_spans:
                raise ValueError(
                    f"span_idx: {span_idx} is greater than the maximum value: {self.len_spans}"
                )

        sub_adj_mat = [[0] * len_comb for _ in range(len_comb)]
        for i in range(len_comb):
            for j in range(len_comb):
                sub_adj_mat[i][j] = self[i, j]

        return sub_adj_mat

    def __repr__(self):
        return f"<SpanRelAdjMat: #{self.len_spans}>"

    def __str__(self):
        string = ""
        adj_mat = self.reveal_adj_mat()
        string += self.__repr__() + "\n"
        for line in adj_mat:
            string += str(line) + "\n"
        return string


class AdjMat(object):
    """
    Adjacent Matrix for building relation graph

    Args:
        event_arg_idx_objs: list of span idxes in each event
            (event-relevant or whole event objs)
        len_spans: the number of spans
        whole_graph: whether to build the whole graph from the event objs
        trigger_aware_graph: use trigger words in `event_type_fields_list`
            to build trigger-based graph
        num_triggers: number of triggers to participate in the trigger-aware graph
        directed_graph: if `trigger_aware_graph`, build directed graph or not
        event_type_idx: if `whole_graph` is False and `trigger_aware_graph` is True,
            assign the event type idx you wanna build for the type-specified graph
    """

    def __init__(
        self,
        event_arg_idx_objs,
        len_spans,
        event_type_fields_list,
        whole_graph=False,
        trigger_aware_graph=False,
        num_triggers=-1,
        directed_graph=False,
        event_type_idx=None,
        try_to_make_up=False,
    ):
        # num of spans
        len_spans = int(len_spans)
        self.len_spans = len_spans
        self.num_triggers = num_triggers
        self.try_to_make_up = try_to_make_up
        self.triggers = set()

        self.adj_mat = torch.zeros(
            len_spans, len_spans, requires_grad=False, dtype=torch.int8
        )

        # fill in the rel_mat
        if event_arg_idx_objs is not None:
            if whole_graph:
                for event_idx, events in enumerate(event_arg_idx_objs):
                    if events is not None:
                        if trigger_aware_graph:
                            self.build_directed_graph(
                                events, event_idx, event_type_fields_list
                            )
                            if not directed_graph:
                                self.fold()
                        else:
                            self.build_undirected_graph(events)
            else:
                if trigger_aware_graph:
                    self.build_directed_graph(
                        event_arg_idx_objs, event_type_idx, event_type_fields_list
                    )
                    if not directed_graph:
                        self.fold()
                else:
                    self.build_undirected_graph(event_arg_idx_objs)

    def build_undirected_graph(self, event_arg_idx_objs):
        connections = build_span_rel_connection_for_each_event(
            event_arg_idx_objs, self.len_spans
        )
        for arg1, connected_args in connections.items():
            for arg2 in connected_args:
                self[arg1, arg2] = 1
                self[arg2, arg1] = 1

    def build_directed_graph(
        self, event_args_objs, event_idx, event_type_fields_list, at_least_one=False
    ):
        event_type_triggers = event_type_fields_list[event_idx][2]
        if self.num_triggers > len(event_type_triggers) - 1:
            num_triggers = len(event_type_triggers) - 1
        else:
            num_triggers = self.num_triggers
        triggers = [
            event_type_fields_list[event_idx][1].index(x)
            for x in event_type_triggers[num_triggers]
        ]
        for obj in event_args_objs:
            if isinstance(obj[0], tuple):
                # with role
                args = [None] * len(event_type_fields_list[event_idx][1])
                for arg, role_idx in obj:
                    args[role_idx] = arg
            else:
                args = obj
            # trigger fields
            trigger_args = list(filter(lambda x: args[x] is not None, triggers))

            if self.try_to_make_up:
                # if there's no enough triggers, try to make up to the num_triggers
                if len(trigger_args) < num_triggers:
                    available_triggers = []
                    for tt in event_type_triggers["all"]:
                        tt_field = event_type_fields_list[event_idx][1].index(tt)
                        if tt_field not in trigger_args and args[tt_field] is not None:
                            available_triggers.append(tt_field)
                    absent_num = num_triggers - len(trigger_args)
                    trigger_args = trigger_args + available_triggers[:absent_num]

            if at_least_one:
                if len(trigger_args) == 0:
                    for tt in event_type_triggers["all"]:
                        tt_field = event_type_fields_list[event_idx][1].index(tt)
                        if args[tt_field] is not None:
                            trigger_args = [tt_field]

            for trigger_arg in trigger_args:
                for arg in args:
                    if arg is not None:
                        self[args[trigger_arg], arg] = 1
                        self.triggers.add(args[trigger_arg])

    def fold(self):
        self.adj_mat = torch.bitwise_or(self.adj_mat, self.adj_mat.t())

    def __getitem__(self, index):
        return self.adj_mat[index]

    def __setitem__(self, index, value):
        self.adj_mat[index] = value

    def reveal_adj_mat(self, masked_diagonal=-1, tolist=True):
        if masked_diagonal is not None:
            mat = self.adj_mat.clone().fill_diagonal_(masked_diagonal)
        else:
            mat = self.adj_mat

        if tolist:
            return mat.tolist()
        else:
            return mat

    def tolist(self, masked_diagonal=None):
        return self.reveal_adj_mat(masked_diagonal, tolist=True)

    def smooth_tensor_rel_mat(self, diagonal=0.5, dim=1) -> torch.Tensor:
        r"""get smoothed rel mat in tensor format"""
        new_mat = torch.clone(
            self.reveal_adj_mat(masked_diagonal=0, tolist=False)
        ).float()
        num_ones = new_mat.sum(dim=dim, keepdim=True)
        diagonals = diagonal * torch.ones_like(num_ones, dtype=torch.float)
        num_ones[num_ones <= 0.0001] = -1.0
        diagonals[num_ones < 0] = 1.0
        new_mat.mul_((1.0 - diagonal) / num_ones)
        new_mat.scatter_(
            -1,
            torch.arange(0, num_ones.shape[0], device=new_mat.device).unsqueeze(1),
            diagonals,
        )
        return new_mat.abs()

    def get_sub_graph_adj_mat(self, combination):
        """
        get sub-graph and returns the adjacent matrix

        Returns:
            List[List]
        """
        len_comb = len(set(combination))

        for span_idx in combination:
            if span_idx >= self.len_spans:
                raise ValueError(
                    f"span_idx: {span_idx} is greater than the maximum value: {self.len_spans}"
                )

        sub_adj_mat = torch.zeros(
            len_comb, len_comb, requires_grad=False, dtype=torch.int8
        )
        for i in range(len_comb):
            for j in range(len_comb):
                sub_adj_mat[i, j] = self[i, j]

        return sub_adj_mat

    def __repr__(self):
        return f"<AdjMat: #{self.len_spans}>"

    def __str__(self):
        string = ""
        adj_mat = self.reveal_adj_mat()
        string += self.__repr__() + "\n"
        string += str(adj_mat)
        return string


class DEEArgRelFeature(object):
    def __init__(
        self,
        guid,
        ex_idx,
        event_type_fields_list,
        doc_type,
        doc_token_id_mat,
        doc_token_mask_mat,
        doc_token_label_mat,
        span_token_ids_list,
        span_dranges_list,
        exist_span_token_tup_set,
        span_token_tup2type,
        event_type_labels,
        event_arg_idxs_objs_list,
        complementary_field2ents,
        valid_sent_num=None,
        trigger_aware=False,
        num_triggers=-1,
        directed_graph=False,
        try_to_make_up=False,
    ):
        self.guid = guid
        self.ex_idx = ex_idx  # example row index, used for backtracking
        self.bak_ex_idx = ex_idx
        self.doc_type = doc_type
        self.valid_sent_num = valid_sent_num

        self.trigger_aware = trigger_aware
        self.num_triggers = num_triggers
        self.directed_graph = directed_graph
        self.try_to_make_up = try_to_make_up

        # directly set tensor for dee feature to save memory
        self.doc_token_ids = torch.tensor(doc_token_id_mat, dtype=torch.long)
        self.doc_token_masks = torch.tensor(
            doc_token_mask_mat, dtype=torch.uint8
        )  # uint8 for mask
        self.doc_token_labels = torch.tensor(doc_token_label_mat, dtype=torch.long)

        # sorted by the first drange tuple
        # [(token_id, ...), ...]
        # span_idx -> span_token_id tuple
        # [(124, 121, 121, 125, 127, 126), (7770, 836, 6809), (7770, 836, 6809, 6763, 816, 5500, 819, 3300, 7361, 1062, 1385), (7916, 4059, 2356, 7916, 7770, 2832, 6598, 1486, 6418, 3300, 7361, 1062, 1385), (2398, 2128, 6395, 1171, 5500, 819, 3300, 7361, 1062, 1385), (123, 121, 122, 127, 2399, 122, 122, 3299, 123, 124, 3189), (122, 122, 124, 128, 121, 121, 121, 121, 5500), (123, 121, 122, 128, 2399, 126, 3299, 123, 125, 3189), (122, 129, 123, 122, 124, 125, 123, 5500), (123, 121, 122, 128, 2399, 127, 3299, 127, 3189), (129, 129, 126, 125, 128, 129, 5500), (122, 125, 121, 129, 125, 124, 128, 125, 121, 5500), (124, 122, 119, 124, 122, 110), (124, 130, 121, 121, 121, 121, 121, 121, 5500), (123, 121, 122, 128, 2399, 122, 122, 3299, 123, 125, 3189)]
        self.span_token_ids_list = span_token_ids_list

        # all span token tuples where all the spans exist in instances
        self.exist_span_token_tup_set = exist_span_token_tup_set

        # span types
        # `0`: non exist (dependent nodes, 0-degree)
        # `1`: exist (not shared nodes, regular sub-graph)
        # `2`: exist and shared (more degree than sub-graph nodes)
        # `3`: non exist and wrongly predicted (not shared nodes, wrongly predicted, 0-degree)
        self.span_token_tup2type = span_token_tup2type

        # span_token_ids -> span_idx
        # span_idx starts from 0, span_token_ids is depend on the span contents, not the dranges, so it's an end-to-end process
        self.span_token_ids2span_idx = {
            token_ids: idx for idx, token_ids in enumerate(self.span_token_ids_list)
        }

        # [[(sent_idx, char_s, char_e), ...], ...]
        # span_idx -> [drange tuple, ...]
        # [[(0, 5, 11)], [(0, 16, 19)], [(1, 0, 11), (3, 0, 11), (12, 0, 11)], [(3, 30, 43)], [(3, 69, 79)], [(5, 0, 11)], [(5, 25, 34)], [(5, 35, 45)], [(5, 61, 69)], [(5, 70, 79)], [(5, 95, 102)], [(7, 20, 30)], [(7, 38, 44)], [(7, 57, 66)], [(13, 0, 11)]]
        self.span_dranges_list = span_dranges_list

        # [event_type_label, ...]
        # length = the total number of events to be considered
        # event_type_label \in {0, 1}, 0: no 1: yes
        # [0, 0, 0, 0, 1]
        self.event_type_labels = event_type_labels

        # event_type is denoted by the index of event_type_labels
        # event_type_idx -> event_obj_idx -> event_arg_idx -> (span_idx, field_type)
        # if no event objects, event_type_idx -> None
        # [None, None, None, None, [((3, 1), (10, 2), (4, 0)), ()]]
        self.event_arg_idxs_objs_list = event_arg_idxs_objs_list

        # complementary ents extracted by regex matcher with fields like `ratio`, `money`, `share` and `date`
        self.complementary_field2ents = complementary_field2ents

        # build span relation connections
        len_spans = len(span_token_ids_list)
        # self.span_rel_mats = [SpanRelAdjMat(es, len_spans) for es in event_arg_idxs_objs_list]
        # self.whole_arg_rel_mat = SpanRelAdjMat(event_arg_idxs_objs_list, len_spans, whole_graph=True)
        self.span_rel_mats = [
            AdjMat(
                es,
                len_spans,
                event_type_fields_list,
                trigger_aware_graph=trigger_aware,
                num_triggers=num_triggers,
                directed_graph=directed_graph,
                event_type_idx=es_idx,
                try_to_make_up=self.try_to_make_up,
            )
            for es_idx, es in enumerate(event_arg_idxs_objs_list)
        ]
        self.whole_arg_rel_mat = AdjMat(
            event_arg_idxs_objs_list,
            len_spans,
            event_type_fields_list,
            whole_graph=True,
            trigger_aware_graph=trigger_aware,
            num_triggers=num_triggers,
            directed_graph=directed_graph,
            try_to_make_up=self.try_to_make_up,
        )

    def get_event_args_objs_list(self):
        event_args_objs_list = []
        for event_arg_idxs_objs in self.event_arg_idxs_objs_list:
            if event_arg_idxs_objs is None:
                event_args_objs_list.append(None)
            else:
                event_args_objs = []
                for event_arg_idxs in event_arg_idxs_objs:
                    event_args = []
                    for arg_idx in event_arg_idxs:
                        if arg_idx is None:
                            token_tup = None
                        else:
                            token_tup = self.span_token_ids_list[arg_idx]
                        event_args.append(token_tup)
                    event_args_objs.append(event_args)
                event_args_objs_list.append(event_args_objs)

        return event_args_objs_list

    @staticmethod
    def build_arg_rel_info(
        event_arg_idxs_objs_list,
        num_spans,
        event_type_fields_list,
        whole_graph=False,
        trigger_aware=False,
        num_triggers=-1,
        directed_graph=False,
        try_to_make_up=False,
    ):
        if whole_graph:
            # event_idx2arg_rel_info = SpanRelAdjMat(event_arg_idxs_objs_list, num_spans, whole_graph=True)
            event_idx2arg_rel_info = AdjMat(
                event_arg_idxs_objs_list,
                num_spans,
                event_type_fields_list,
                whole_graph=True,
                trigger_aware_graph=trigger_aware,
                num_triggers=num_triggers,
                directed_graph=directed_graph,
                try_to_make_up=try_to_make_up,
            )
        else:
            # here, the span idx has changed to the predicted span_idx if in predict mode
            # if the gold spans are used for training, stay the idxes unchanged
            # event_idx2arg_rel_info = [SpanRelAdjMat(es, num_spans) for es in event_arg_idxs_objs_list]
            event_idx2arg_rel_info = [
                AdjMat(
                    es,
                    num_spans,
                    event_type_fields_list,
                    trigger_aware_graph=trigger_aware,
                    num_triggers=num_triggers,
                    directed_graph=directed_graph,
                    event_type_idx=es_idx,
                    try_to_make_up=try_to_make_up,
                )
                for es_idx, es in enumerate(event_arg_idxs_objs_list)
            ]
        return event_idx2arg_rel_info

    def generate_arg_rel_mat_for(
        self, pred_span_token_tup_list, event_type_fields_list, return_miss=False
    ):
        token_tup2pred_span_idx = {
            token_tup: pred_span_idx
            for pred_span_idx, token_tup in enumerate(pred_span_token_tup_list)
        }
        gold_span_idx2pred_span_idx = {}
        missed_span_idx_list = []  # in terms of self
        missed_sent_idx_list = []  # in terms of self
        for gold_span_idx, token_tup in enumerate(self.span_token_ids_list):
            # tzhu: token_tup: token ids for each span
            if token_tup in token_tup2pred_span_idx:
                pred_span_idx = token_tup2pred_span_idx[token_tup]
                gold_span_idx2pred_span_idx[gold_span_idx] = pred_span_idx
            else:  # tzhu: not predicted
                missed_span_idx_list.append(gold_span_idx)
                for gold_drange in self.span_dranges_list[gold_span_idx]:
                    missed_sent_idx_list.append(gold_drange[0])
        missed_sent_idx_list = list(set(missed_sent_idx_list))

        pred_event_arg_idxs_objs_list = []
        for event_arg_idxs_objs in self.event_arg_idxs_objs_list:
            if event_arg_idxs_objs is None:
                pred_event_arg_idxs_objs_list.append(None)
            else:
                pred_event_arg_idxs_objs = []
                for event_arg_idxs in event_arg_idxs_objs:
                    pred_event_arg_idxs = []
                    for gold_span_idx, field_type in event_arg_idxs:
                        if gold_span_idx in gold_span_idx2pred_span_idx:
                            pred_event_arg_idxs.append(
                                (gold_span_idx2pred_span_idx[gold_span_idx], field_type)
                            )
                            # there is no none field during training to
                            # get all the types for each role.
                            # while evaluating, we should convert it to
                            # add the none fields, so that's why we remove
                            # the `None` adding operation here.
                    if len(pred_event_arg_idxs) != 0:
                        pred_event_arg_idxs_objs.append(tuple(pred_event_arg_idxs))
                if len(pred_event_arg_idxs_objs) == 0:
                    pred_event_arg_idxs_objs = None
                pred_event_arg_idxs_objs_list.append(pred_event_arg_idxs_objs)

        num_spans = len(pred_span_token_tup_list)
        pred_arg_rel_mats = self.build_arg_rel_info(
            pred_event_arg_idxs_objs_list,
            num_spans,
            event_type_fields_list,
            trigger_aware=self.trigger_aware,
            num_triggers=self.num_triggers,
            directed_graph=self.directed_graph,
            try_to_make_up=self.try_to_make_up,
        )
        whole_arg_rel_mat = self.build_arg_rel_info(
            pred_event_arg_idxs_objs_list,
            num_spans,
            event_type_fields_list,
            whole_graph=True,
            trigger_aware=self.trigger_aware,
            num_triggers=self.num_triggers,
            directed_graph=self.directed_graph,
            try_to_make_up=self.try_to_make_up,
        )
        if return_miss:
            return (
                pred_arg_rel_mats,
                whole_arg_rel_mat,
                pred_event_arg_idxs_objs_list,
                missed_span_idx_list,
                missed_sent_idx_list,
            )
        else:
            return pred_arg_rel_mats, whole_arg_rel_mat

    def generate_arg_rel_mat_with_none_for(
        self, pred_span_token_tup_list, event_type_fields_list, return_miss=False
    ):
        token_tup2pred_span_idx = {
            token_tup: pred_span_idx
            for pred_span_idx, token_tup in enumerate(pred_span_token_tup_list)
        }
        gold_span_idx2pred_span_idx = {}
        missed_span_idx_list = []  # in terms of self
        missed_sent_idx_list = []  # in terms of self
        for gold_span_idx, token_tup in enumerate(self.span_token_ids_list):
            # tzhu: token_tup: token ids for each span
            if token_tup in token_tup2pred_span_idx:
                pred_span_idx = token_tup2pred_span_idx[token_tup]
                gold_span_idx2pred_span_idx[gold_span_idx] = pred_span_idx
            else:  # tzhu: not predicted
                missed_span_idx_list.append(gold_span_idx)
                for gold_drange in self.span_dranges_list[gold_span_idx]:
                    missed_sent_idx_list.append(gold_drange[0])
        missed_sent_idx_list = list(set(missed_sent_idx_list))

        pred_event_arg_idxs_objs_list = []
        for event_arg_idxs_objs in self.event_arg_idxs_objs_list:
            if event_arg_idxs_objs is None:
                pred_event_arg_idxs_objs_list.append(None)
            else:
                pred_event_arg_idxs_objs = []
                for event_arg_idxs in event_arg_idxs_objs:
                    pred_event_arg_idxs = []
                    for gold_span_idx, field_type in event_arg_idxs:
                        if gold_span_idx in gold_span_idx2pred_span_idx:
                            pred_event_arg_idxs.append(
                                (gold_span_idx2pred_span_idx[gold_span_idx], field_type)
                            )
                        else:
                            # not one predicted entity can express this role
                            pred_event_arg_idxs.append((None, field_type))
                    if len(pred_event_arg_idxs) != 0:
                        pred_event_arg_idxs_objs.append(tuple(pred_event_arg_idxs))
                if len(pred_event_arg_idxs_objs) == 0:
                    pred_event_arg_idxs_objs = None
                pred_event_arg_idxs_objs_list.append(pred_event_arg_idxs_objs)

        num_spans = len(pred_span_token_tup_list)
        pred_arg_rel_mats = self.build_arg_rel_info(
            pred_event_arg_idxs_objs_list,
            num_spans,
            event_type_fields_list,
            trigger_aware=self.trigger_aware,
            num_triggers=self.num_triggers,
            directed_graph=self.directed_graph,
            try_to_make_up=self.try_to_make_up,
        )
        whole_arg_rel_mat = self.build_arg_rel_info(
            pred_event_arg_idxs_objs_list,
            num_spans,
            event_type_fields_list,
            whole_graph=True,
            trigger_aware=self.trigger_aware,
            num_triggers=self.num_triggers,
            directed_graph=self.directed_graph,
            try_to_make_up=self.try_to_make_up,
        )
        if return_miss:
            return (
                pred_arg_rel_mats,
                whole_arg_rel_mat,
                pred_event_arg_idxs_objs_list,
                missed_span_idx_list,
                missed_sent_idx_list,
            )
        else:
            return pred_arg_rel_mats, whole_arg_rel_mat

    def is_multi_event(self):
        event_cnt = 0
        for event_objs in self.event_arg_idxs_objs_list:
            if event_objs is not None:
                event_cnt += len(event_objs)
                if event_cnt > 1:
                    return True

        return False


class DEEArgRelFeatureConverter(object):
    def __init__(
        self,
        entity_label_list,
        template,
        max_sent_len,
        max_sent_num,
        tokenizer,
        ner_fea_converter=None,
        include_cls=True,
        include_sep=True,
        trigger_aware=False,
        num_triggers=-1,
        directed_graph=False,
        try_to_make_up=False,
    ):
        self.entity_label_list = entity_label_list
        self.template = template
        self.event_type_fields_pairs = template.event_type_fields_list
        self.max_sent_len = max_sent_len
        self.max_sent_num = max_sent_num
        self.tokenizer = tokenizer
        self.truncate_doc_count = (
            0  # track how many docs have been truncated due to max_sent_num
        )
        self.truncate_span_count = 0  # track how may spans have been truncated

        self.trigger_aware = trigger_aware
        self.num_triggers = num_triggers
        self.directed_graph = directed_graph
        self.try_to_make_up = try_to_make_up

        # label not in entity_label_list will be default 'O'
        # sent_len > max_sent_len will be truncated, and increase ner_fea_converter.truncate_freq
        if ner_fea_converter is None:
            self.ner_fea_converter = NERFeatureConverter(
                entity_label_list,
                self.max_sent_len,
                tokenizer,
                include_cls=include_cls,
                include_sep=include_sep,
            )
        else:
            self.ner_fea_converter = ner_fea_converter

        self.include_cls = include_cls
        self.include_sep = include_sep

        # prepare entity_label -> entity_index mapping
        self.entity_label2index = {}
        for entity_idx, entity_label in enumerate(self.entity_label_list):
            self.entity_label2index[entity_label] = entity_idx

        # HACK: inject to regex_extractor
        for field_name in regex_extractor.field2type:
            if self.entity_label2index.get("B-" + field_name) is None:
                continue
            regex_extractor.field_id2field_name[
                self.entity_label2index["B-" + field_name]
            ] = field_name
        regex_extractor.basic_type_id = self.entity_label2index["O"]

        # prepare event_type -> event_index and event_index -> event_fields mapping
        self.event_type2index = {}
        self.event_type_list = []
        self.event_fields_list = []
        for event_idx, (event_type, event_fields, _, _) in enumerate(
            self.event_type_fields_pairs
        ):
            self.event_type2index[event_type] = event_idx
            self.event_type_list.append(event_type)
            self.event_fields_list.append(event_fields)

    def convert_example_to_feature(self, ex_idx, dee_example, log_flag=False):
        annguid = dee_example.guid
        assert isinstance(dee_example, DEEExample)

        # 1. prepare doc token-level feature

        # Size(num_sent_num, num_sent_len)
        doc_token_id_mat = []  # [[token_idx, ...], ...]
        doc_token_mask_mat = []  # [[token_mask, ...], ...]
        doc_token_label_mat = []  # [[token_label_id, ...], ...]

        for sent_idx, sent_text in enumerate(dee_example.sentences):
            if sent_idx >= self.max_sent_num:
                # truncate doc whose number of sentences is longer than self.max_sent_num
                self.truncate_doc_count += 1
                break

            if sent_idx in dee_example.sent_idx2srange_mspan_mtype_tuples:
                srange_mspan_mtype_tuples = (
                    dee_example.sent_idx2srange_mspan_mtype_tuples[sent_idx]
                )
            else:
                srange_mspan_mtype_tuples = []

            ner_example = NERExample(
                "{}-{}".format(annguid, sent_idx),
                sent_text,
                self.tokenizer.dee_tokenize(sent_text),
                srange_mspan_mtype_tuples,
            )
            # sentence truncated count will be recorded incrementally
            ner_feature = self.ner_fea_converter.convert_example_to_feature(
                ner_example, log_flag=log_flag
            )

            doc_token_id_mat.append(ner_feature.input_ids)
            doc_token_mask_mat.append(ner_feature.input_masks)
            doc_token_label_mat.append(ner_feature.label_ids)

        assert (
            len(doc_token_id_mat)
            == len(doc_token_mask_mat)
            == len(doc_token_label_mat)
            <= self.max_sent_num
        )
        valid_sent_num = len(doc_token_id_mat)

        # 2. prepare span feature
        # spans are sorted by the first drange
        span_token_ids_list = []
        span_dranges_list = []
        mspan2span_idx = {}
        for mspan in dee_example.ann_valid_mspans:
            if mspan in mspan2span_idx:
                continue

            raw_dranges = dee_example.ann_mspan2dranges[mspan]
            char_base_s = 1 if self.include_cls else 0
            char_max_end = (
                self.max_sent_len - 1 if self.include_sep else self.max_sent_len
            )
            span_dranges = []
            for sent_idx, char_s, char_e in raw_dranges:
                if (
                    char_base_s + char_e <= char_max_end
                    and sent_idx < self.max_sent_num
                ):
                    span_dranges.append(
                        (sent_idx, char_base_s + char_s, char_base_s + char_e)
                    )
                else:
                    self.truncate_span_count += 1
            if len(span_dranges) == 0:
                # span does not have any valid location in truncated sequences
                continue

            span_tokens = self.tokenizer.dee_tokenize(mspan)
            span_token_ids = tuple(self.tokenizer.convert_tokens_to_ids(span_tokens))

            mspan2span_idx[mspan] = len(span_token_ids_list)
            span_token_ids_list.append(span_token_ids)
            span_dranges_list.append(span_dranges)
        assert len(span_token_ids_list) == len(span_dranges_list) == len(mspan2span_idx)

        if len(span_token_ids_list) == 0 and not dee_example.only_inference:
            logger.warning("Neglect example {}".format(ex_idx))
            return None

        # 3. prepare doc-level event feature
        # event_type_labels: event_type_index -> event_type_exist_sign (1: exist, 0: no)
        # event_arg_idxs_objs_list: event_type_index -> event_obj_index -> event_arg_index -> arg_span_token_ids
        exist_span_token_tup_set = set()
        # prepared for span sharing checking
        span2shared_times = defaultdict(lambda: 0)
        event_type_labels = []  # event_type_idx -> event_type_exist_sign (1 or 0)
        event_arg_idxs_objs_list = (
            []
        )  # event_type_idx -> event_obj_idx -> event_arg_idx -> tuple(span_idx, argument_role)
        for event_idx, event_type in enumerate(self.event_type_list):
            event_fields = self.event_fields_list[event_idx]

            if event_type not in dee_example.event_type2event_objs:
                event_type_labels.append(0)
                event_arg_idxs_objs_list.append(None)
            else:
                event_objs = dee_example.event_type2event_objs[event_type]

                event_arg_idxs_objs = []
                for event_obj in event_objs:
                    assert isinstance(event_obj, self.template.BaseEvent)
                    tmp_span_stat = set()
                    event_arg_idxs = []
                    any_valid_flag = False
                    for field_idx, field in enumerate(event_fields):
                        arg_span = event_obj.field2content[field]

                        if arg_span is not None and arg_span in mspan2span_idx:
                            # when constructing data files,
                            # must ensure event arg span is covered by the total span collections
                            arg_span_idx = mspan2span_idx[arg_span]
                            any_valid_flag = True
                            event_arg_idxs.append((arg_span_idx, field_idx))
                            exist_span_token_tup_set.add(
                                span_token_ids_list[arg_span_idx]
                            )
                            tmp_span_stat.add(span_token_ids_list[arg_span_idx])

                    for token_tup in tmp_span_stat:
                        span2shared_times[token_tup] += 1

                    if any_valid_flag:
                        event_arg_idxs_objs.append(tuple(event_arg_idxs))

                if event_arg_idxs_objs:
                    event_type_labels.append(1)
                    event_arg_idxs_objs_list.append(event_arg_idxs_objs)
                else:
                    event_type_labels.append(0)
                    event_arg_idxs_objs_list.append(None)

        # span types
        # `0`: non exist (dependent nodes, 0-degree)
        # `1`: exist (not shared nodes, regular sub-graph)
        # `2`: exist and shared (more degree than sub-graph nodes)
        # `3`: non exist and wrongly predicted (not shared nodes, wrongly predicted, 0-degree)
        #      generated during training to check whether the NER module predictions are right or not
        span_token_tup2type = dict()
        for x in span_token_ids_list:
            if span2shared_times[x] == 0:
                span_token_tup2type[x] = 0
            elif span2shared_times[x] == 1:
                span_token_tup2type[x] = 1
            elif span2shared_times[x] > 1:
                span_token_tup2type[x] = 2
            else:
                raise RuntimeError("span_token_tup existence < 0!")

        doc_type = {
            "o2o": 0,
            "o2m": 1,
            "m2m": 2,
            "unk": 3,
        }[dee_example.doc_type]

        complementary_field2ents = defaultdict(list)  # converted
        comp_field2ents = dee_example.complementary_field2ents
        for field, ents in comp_field2ents.items():
            for ent, ent_span in ents:
                complementary_field2ents[field].append(
                    [
                        self.tokenizer.convert_tokens_to_ids(
                            self.tokenizer.dee_tokenize(ent)
                        ),
                        ent_span,
                    ]
                )

        dee_feature = DEEArgRelFeature(
            annguid,
            ex_idx,
            self.event_type_fields_pairs,
            doc_type,
            doc_token_id_mat,
            doc_token_mask_mat,
            doc_token_label_mat,
            span_token_ids_list,
            span_dranges_list,
            exist_span_token_tup_set,
            span_token_tup2type,
            event_type_labels,
            event_arg_idxs_objs_list,
            complementary_field2ents,
            valid_sent_num=valid_sent_num,
            trigger_aware=self.trigger_aware,
            num_triggers=self.num_triggers,
            directed_graph=self.directed_graph,
            try_to_make_up=self.try_to_make_up,
        )

        return dee_feature

    def __call__(self, dee_examples, log_example_num=0):
        """Convert examples to features suitable for document-level event extraction"""
        dee_features = []
        self.truncate_doc_count = 0
        self.truncate_span_count = 0
        self.ner_fea_converter.truncate_count = 0

        remove_ex_cnt = 0

        # idx2doc_type = {
        #     0: "o2o",
        #     1: "o2m",
        #     2: "m2m"
        # }
        # failed_cnt = {'o2o': 0, 'o2m': 0, 'm2m': 0, 'tot': 0}
        # tot_cnt = {'o2o': 0, 'o2m': 0, 'm2m': 0, 'tot': 0}
        # failed_results = []
        # failed_reason = defaultdict(lambda: defaultdict(lambda: 0))
        num_connections = 0
        num_tot_rels = 0

        pbar = tqdm(dee_examples, ncols=80, ascii=True)
        for ex_idx, dee_example in enumerate(pbar):
            # """setting for stats"""
            # num_triggers = 5
            # eval_num_triggers = -1
            # max_clique = True
            # with_left_trigger = True
            # with_all_one_trigger_comb = False
            # self_loop = True

            # self.num_triggers = num_triggers
            # """end of setting for stats"""

            if ex_idx < log_example_num:
                dee_feature = self.convert_example_to_feature(
                    ex_idx - remove_ex_cnt, dee_example, log_flag=True
                )
            else:
                dee_feature = self.convert_example_to_feature(
                    ex_idx - remove_ex_cnt, dee_example, log_flag=False
                )

            if dee_feature is None:
                remove_ex_cnt += 1
                continue

            dee_features.append(dee_feature)

            num_connections += dee_feature.whole_arg_rel_mat.reveal_adj_mat(
                masked_diagonal=None, tolist=False
            ).sum()
            num_tot_rels += dee_feature.whole_arg_rel_mat.len_spans**2

            # """begin of stats"""
            # event_arg_idxs_objs_list = remove_event_obj_roles(dee_feature.event_arg_idxs_objs_list, self.event_type_fields_pairs)
            # gold_combinations = extract_combinations_from_event_objs(event_arg_idxs_objs_list)

            # adj_mat = dee_feature.whole_arg_rel_mat.reveal_adj_mat(masked_diagonal=None)
            # gold_triggers = dee_feature.whole_arg_rel_mat.triggers
            # decoded = adj_decoding.directed_trigger_graph_decode(
            #     adj_mat,
            #     num_triggers=eval_num_triggers,
            #     self_loop=self_loop,
            #     max_clique=max_clique,
            #     with_left_trigger=with_left_trigger,
            #     with_all_one_trigger_comb=with_all_one_trigger_comb
            # )
            # doc_type = idx2doc_type[dee_feature.doc_type]
            # if set(decoded) != gold_combinations:
            #     failed_cnt['tot'] += 1
            #     failed_cnt[doc_type] += 1

            #     decoded_triggers = set()
            #     decoded_connections = adj_decoding.build_single_element_connections(adj_mat, tuple_key=False, self_loop=self_loop)
            #     for u, vs in decoded_connections.items():
            #         if len(vs) > 0:
            #             decoded_triggers.add(u)

            #     if len(gold_combinations) == 1 and len(list(gold_combinations)[0]) == 1:
            #         reason = 'single_element'
            #     elif np.sum(dee_feature.whole_arg_rel_mat.reveal_adj_mat(masked_diagonal=None)) == 0:
            #         reason = 'no_trigger'
            #     elif set(list(filter(lambda x: len(x) > 1, gold_combinations))) == set(decoded):
            #         reason = 'single_element_with_other_comb'
            #     elif len(gold_triggers - decoded_triggers) > 0 and len(decoded_triggers - gold_triggers) == 0:
            #         reason = 'decode_absent_trigger'
            #     elif len(decoded_triggers - gold_triggers) > 0 and len(gold_triggers - decoded_triggers) == 0:
            #         reason = 'decode_more_trigger'
            #     elif len(decoded_triggers - gold_triggers) > 0 and len(gold_triggers - decoded_triggers) > 0:
            #         reason = 'decode_more_less_trigger'
            #     elif gold_triggers == decoded_triggers and len(gold_combinations) > len(decoded) and all(t in comb for t in decoded_triggers for comb in decoded):
            #         reason = 'num_limit_cannot_distinguish'
            #     elif len(gold_combinations - set(decoded)) > 0 and any(set(pc) & set(pc2) == set(pc) or set(pc) & set(pc2) == set(pc2) for pc in decoded for pc2 in decoded):
            #         reason = 'no_attached_common_nodes'
            #     else:
            #         reason = 'others'
            #     failed_reason[doc_type][reason] += 1
            #     failed_results.append(OrderedDict({
            #         "guid": dee_feature.guid,
            #         "doc_type": doc_type,
            #         "reason": reason,
            #         "gold_triggers": gold_triggers,
            #         "pred_triggers": decoded_triggers,
            #         "gold_combs": list(gold_combinations),
            #         "pred_combs": decoded,
            #         "adj_mat": adj_mat,
            #     }))

            # tot_cnt['tot'] += 1
            # tot_cnt[doc_type] += 1
            # """end of stats"""
        logger.info(f"num_tot_rels={num_tot_rels}, num_connections={num_connections}")
        logger.info(pbar)
        logger.info(
            "{} documents, ignore {} examples, truncate {} docs, {} sents, {} spans".format(
                len(dee_examples),
                remove_ex_cnt,
                self.truncate_doc_count,
                self.ner_fea_converter.truncate_count,
                self.truncate_span_count,
            )
        )

        # """stats"""
        # # with open("failed_decoding_22TTF.json", "wt", encoding="utf-8") as fout:
        # #     for ins in failed_results:
        # #         string = pprint.pformat(ins, indent=2)
        # #         fout.write(f'{string}\n')
        # # default_dump_json(failed_results, "failed_decoding_22TTF.json")
        # logger.info(f"num_triggers={num_triggers}, eval_num_triggers={eval_num_triggers}, max_clique={max_clique}, with_left_trigger={with_left_trigger}, with_all_one_trigger_comb={with_all_one_trigger_comb}, self_loop={self_loop}")
        # logger.info(f"o2o failed decoding number: {failed_cnt['o2o']}/{tot_cnt['o2o']} = {100 * failed_cnt['o2o'] / tot_cnt['o2o']:.3f}%")
        # logger.info(f"o2m failed decoding number: {failed_cnt['o2m']}/{tot_cnt['o2m']} = {100 * failed_cnt['o2m'] / tot_cnt['o2m']:.3f}%")
        # logger.info(f"m2m failed decoding number: {failed_cnt['m2m']}/{tot_cnt['m2m']} = {100 * failed_cnt['m2m'] / tot_cnt['m2m']:.3f}%")
        # logger.info(f"total failed decoding number: {failed_cnt['tot']}/{tot_cnt['tot']} = {100 * failed_cnt['tot'] / tot_cnt['tot']:.3f}%")
        # logger.info(f"failed reason: {json.dumps(failed_reason, indent=2, ensure_ascii=False)}")
        # assert 0
        # """end of stats"""

        return dee_features


def convert_dee_arg_rel_features_to_dataset(dee_arg_rel_features):
    # just view a list of doc_fea as the dataset, that only requires __len__, __getitem__
    assert len(dee_arg_rel_features) > 0 and isinstance(
        dee_arg_rel_features[0], DEEArgRelFeature
    )
    return dee_arg_rel_features
