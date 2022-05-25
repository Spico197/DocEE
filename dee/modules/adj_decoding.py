import functools
import random
from collections import defaultdict
from typing import List, Tuple

from dee.utils import fold_and


def build_single_element_connections(adj_mat, tuple_key=True, self_loop=False):
    r"""
    Establish out-degree connections for directed graph,
    or bi-directional links for undirected graph.
    """
    connections = defaultdict(set)
    for i in range(len(adj_mat)):
        if tuple_key:
            key = (i,)
        else:
            key = i
        for j in range(len(adj_mat)):
            if adj_mat[i][j] == 1:
                if self_loop or (not self_loop and i != j):
                    connections[key].add(j)
        connections[key] = connections[key]
    return connections


def brute_force_adj_decode(adj_mat, min_num_arg):
    """
    Brute-Force Algorithm for Complete Sub-Graph Extracting from Adjacent Matrix
    Initialization: M = set()
    Input: Adjacent Matrix: A, Argument Set: args
    Output: All complete graph with the number of spans greater `than min_num_arg`: M

    Args:
        adj_mat: adjacent matrix
        min_num_arg: event instance with a minimum number of spans will be kept

    Returns:
        all the span combinations
    """
    M = defaultdict(lambda: defaultdict(set))
    M_record = defaultdict(set)  # avoid redundant
    num_spans = len(adj_mat)
    single_ele_connections = build_single_element_connections(adj_mat)
    M[1].update(single_ele_connections)
    for num_of_span in range(2, num_spans + 1):
        for key_args, connected_args in M[num_of_span - 1].items():
            for connected_arg in connected_args:
                connected_connected_arg_set = set(
                    single_ele_connections[(connected_arg,)]
                )
                key_args_set = set(key_args)
                if (key_args_set & connected_connected_arg_set) == key_args_set:
                    connected_args_set = set(connected_args)
                    new_connected_args = (
                        connected_args_set & connected_connected_arg_set
                    )
                    new_key_args = tuple(sorted(list(key_args) + [connected_arg]))
                    if new_key_args not in M_record[num_of_span]:
                        M[num_of_span][new_key_args] = new_connected_args
                        M_record[num_of_span].add(new_key_args)
    all_complete_graph = []
    for num_of_span in M:
        if num_of_span >= min_num_arg:
            all_complete_graph.extend(list(M[num_of_span].keys()))
    all_complete_graph.sort(key=lambda x: len(x), reverse=True)
    return all_complete_graph


"""
Bron-Kerbosch Algorithm Implementation

References:
    - https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm
    - https://stackoverflow.com/questions/13904636/implementing-bron-kerbosch-algorithm-in-python
    - https://github.com/cornchz/Bron-Kerbosch
"""


def bron_kerbosch_decode(adj_mat: List, min_num_arg: int):
    neighbour_nodes = build_single_element_connections(adj_mat, tuple_key=False)

    def algo(clique: set, candidates: set, excluded: set):
        if not any((candidates, excluded)):
            if len(clique) >= min_num_arg:
                yield clique
        while candidates:
            v = candidates.pop()
            new_candidates = candidates.intersection(neighbour_nodes[v])
            new_excluded = excluded.intersection(neighbour_nodes[v])
            yield from algo(clique.union({v}), new_candidates, new_excluded)
            excluded.add(v)

    all_cliques = []
    for clique in algo(set(), set(neighbour_nodes.keys()), set()):
        all_cliques.append(tuple(sorted(clique)))
    all_cliques.sort(key=lambda x: len(x), reverse=True)
    return all_cliques


def bron_kerbosch_pivoting_decode(adj_mat: List, min_num_arg: int):
    neighbour_nodes = build_single_element_connections(adj_mat, tuple_key=False)

    def algo(clique: set, candidates: set, excluded: set):
        if not any((candidates, excluded)):
            if len(clique) >= min_num_arg:
                yield clique
        if not candidates.union(excluded):
            # the first call may bring the empty indexing issue
            S = candidates
        else:
            u = random.choice(list(candidates.union(excluded)))
            S = candidates.difference(neighbour_nodes[u])
        for v in S:
            new_candidates = candidates.intersection(neighbour_nodes[v])
            new_excluded = excluded.intersection(neighbour_nodes[v])
            yield from algo(clique.union({v}), new_candidates, new_excluded)
            candidates.remove(v)
            excluded.add(v)

    all_cliques = []
    for clique in algo(set(), set(neighbour_nodes.keys()), set()):
        all_cliques.append(tuple(sorted(clique)))
    all_cliques.sort(key=lambda x: len(x), reverse=True)
    return all_cliques


def linked_decode(adj_mat: List):
    r"""
    as long as there is a link between nodes, these are in the same combination
    """
    neighbour_nodes = build_single_element_connections(adj_mat, tuple_key=False)
    combs = []

    def bfs(node):
        visited = set()
        latter = {node}
        while latter:
            curr = latter
            latter = set()
            for node in curr:
                if node not in visited:
                    yield node
                    visited.add(node)
                    latter.update(neighbour_nodes[node])

    visited = set()
    for node in neighbour_nodes:
        if node not in visited:
            c = set(bfs(node))
            combs.append(c)
            visited.update(c)

    return [tuple(sorted(list(comb))) for comb in combs]


def undirected_trigger_graph_decode(adj_mat: List):
    raise NotImplementedError


def get_common_and_trigger_connections(connections):
    N_c = defaultdict(set)
    N_t = defaultdict(set)
    for v, conn in connections.items():
        for u in conn:
            if len(connections[u]) > 0:
                N_t[v].add(u)
            else:
                N_c[v].add(u)
    return N_c, N_t


def directed_trigger_graph_decode(
    adj_mat: List[List[int]],
    num_triggers: int,
    self_loop=False,
    max_clique=False,
    with_left_trigger=False,
    with_all_one_trigger_comb=False,
) -> List[Tuple[int]]:
    r"""get decoded combinations from adjacent mat

    Args:
        adj_mat: adjacent matrix
        num_triggers: number of triggers.
            guessing mode if `num_triggers` < 1
        self_loop: whether diag entries should be taken into consideration
        max_clique: whether to return maximal cliques instead of all combiantions,
            BK algorithm if `max_clique`, else BF algorithm
        with_left_trigger: after updating BK cliques into final results,
            whether to take left triggered-combinations into consideration
        with_all_one_trigger_comb: with all 1-trigger combinations

    Returns:
        list of combinations in tuple format
    """
    connections = build_single_element_connections(
        adj_mat, tuple_key=False, self_loop=self_loop
    )
    triggers = set()
    for u, vs in connections.items():
        if len(vs) > 0:
            triggers.add(u)

    combs = list()
    if num_triggers < 1:
        # guessing mode
        # num_triggers = sum(len(val) > 0 for val in connections.values())
        num_triggers = len(triggers)

    if num_triggers == 1:
        for v in connections:
            comb = set()
            if len(connections[v]) > 0:
                comb.add(v)
                comb.update(connections[v])
            if len(comb) > 0 and comb not in combs:
                combs.append(comb)
    else:
        fold_adj_mat = fold_and(adj_mat)
        # fold_and_adj_mat = left_tril(adj_mat)
        if max_clique:
            trigger_combs = bron_kerbosch_pivoting_decode(fold_adj_mat, 2)
        else:
            trigger_combs = brute_force_adj_decode(fold_adj_mat, 2)
        trigger_combs = list(filter(lambda c: len(c) <= num_triggers, trigger_combs))
        used_triggers = set()
        for tc in trigger_combs:
            used_triggers.update(set(tc))
            comb = set(tc)
            # here, all node in `comb` are triggers all pointing to the same successors
            comb.update(
                functools.reduce(
                    lambda A, v: A & connections[v], tc[1:], connections[tc[0]]
                )
            )
            if len(comb) > 0 and comb not in combs:
                combs.append(comb)

        left_triggers = triggers - used_triggers
        if with_left_trigger:
            for v in left_triggers:
                comb = set()
                if len(connections[v]) > 0:
                    comb.add(v)
                    comb.update(connections[v])
                if len(comb) > 0 and comb not in combs:
                    combs.append(comb)

        if with_all_one_trigger_comb:
            for v in triggers:
                comb = {v}
                comb.update(set(filter(lambda u: u not in triggers, connections[v])))
                if len(comb) > 0 and comb not in combs:
                    combs.append(comb)

    ret_combs = []
    for comb in combs:
        ret_combs.append(tuple(sorted(list(comb))))
    ret_combs.sort(key=lambda x: len(x), reverse=True)
    return ret_combs


def directed_trigger_graph_incremental_decode(
    adj_mat: List[List[int]], num_triggers: int, min_conn=1
) -> List[Tuple[int]]:
    r"""get decoded combinations from adjacent mat with incrementally expanding

    Args:
        adj_mat: adjacent matrix
        num_triggers: number of triggers.
            guessing mode if `num_triggers` < 1
        min_conn: minimal number of connections. To absorb a trigger into
            the current combination, the trigger must have `>=min_conn`
            non-trigger neighbours connected with current combination.

    Returns:
        list of combinations in tuple format
    """
    connections = build_single_element_connections(adj_mat, tuple_key=False)
    combs = list()
    if num_triggers < 1:
        # guessing mode
        num_triggers = sum(len(val) > 0 for val in connections.values())

    if num_triggers == 1:
        for v in connections:
            comb = set()
            if len(connections[v]) > 0:
                comb.add(v)
                comb.update(connections[v])
            if len(comb) > 0 and comb not in combs:
                combs.append(comb)
    else:
        for v in connections:
            comb = set()
            if len(connections[v]) > 0:
                comb.add(v)
                # update leaves into current combination
                non_trigger_neighbours = set(
                    filter(lambda u: len(connections[u]) <= 0, connections[v])
                )
                comb.update(non_trigger_neighbours)
                # for trigger neighbours, check the min_conn restriction
                for u in filter(lambda u: len(connections[u]) > 0, connections[v]):
                    if (
                        len(connections[u].intersection(non_trigger_neighbours))
                        >= min_conn
                    ):
                        comb.add(u)
            if len(comb) > 0 and comb not in combs:
                combs.append(comb)

    ret_combs = []
    for comb in combs:
        ret_combs.append(tuple(sorted(list(comb))))
    ret_combs.sort(key=lambda x: len(x), reverse=True)
    return ret_combs


def directed_mst_graph_decode(adj_mats: List[List[List[int]]]) -> List[Tuple[int]]:
    r"""get multi-step decoded combinations from adjacent mat

    Args:
        adj_mats: adjacent matrices

    Returns:
        list of combinations in tuple format
    """
    # def recursive_trigger_decode(base_trigger: int, former_connections: Dict[int, set], curr_connections: Dict[int, set], former_comb: set):
    #     triggers = [x[0] for x in filter(lambda key, val: len(val) > 0, curr_connections.items())]
    #     combs = []
    #     for trigger in triggers:
    #         if trigger in former_connections[base_trigger] and base_trigger in curr_connections[trigger]:
    #             comb = former_comb.intersection(former_connections[base_trigger] & curr_connections[trigger])
    #             comb.update({trigger, base_trigger})
    #             yield comb
    #             # for comb in recursive_trigger_decode(trigger, curr_connections)
    #             # combs.append(comb)

    # step_connections = [build_single_element_connections(mat, tuple_key=False) for mat in adj_mats]
    # combs = list()
    # # for connections in step_connections:
    # #     triggers = [x[0] for x in filter(lambda key, val: len(val) > 0, connections.items())]

    # # if num_triggers == 1:
    # #     for v in connections:
    # #         comb = set()
    # #         if len(connections[v]) > 0:
    # #             comb.add(v)
    # #             comb.update(connections[v])
    # #         if len(comb) > 0 and comb not in combs:
    # #             combs.append(comb)
    # # else:
    # #     fold_adj_mat = fold_and(adj_mat)
    # #     # fold_and_adj_mat = left_tril(adj_mat)
    # #     if max_clique:
    # #         trigger_combs = bron_kerbosch_pivoting_decode(fold_adj_mat, 2)
    # #     else:
    # #         trigger_combs = brute_force_adj_decode(fold_adj_mat, 2)
    # #     trigger_combs = list(filter(lambda c: len(c) <= num_triggers, trigger_combs))
    # #     for tc in trigger_combs:
    # #         comb = set(tc)
    # #         comb.update(functools.reduce(lambda A, v: A & connections[v], tc[1:], connections[tc[0]]))
    # #         if len(comb) > 0 and comb not in combs:
    # #             combs.append(comb)

    # ret_combs = []
    # for comb in combs:
    #     ret_combs.append(tuple(sorted(list(comb))))
    # ret_combs.sort(key=lambda x: len(x), reverse=True)
    # return ret_combs

    raise NotImplementedError
