import time

import pytest

from dee.event_types import get_event_template
from dee.helper.arg_rel import AdjMat, SpanRelAdjMat
from dee.modules import adj_decoding


@pytest.fixture
def tmp_event_type_fields_list_for_test():
    template = get_event_template("zheng2019_trigger_graph")
    return template.event_type_fields_list


def test_brute_force():
    span_rel_adj_obj = SpanRelAdjMat([(0, 1, 2, 3), (0, 4, 5, 6), (2, 5, 6, 7)], 8)
    adj_mat = span_rel_adj_obj.reveal_adj_mat()
    brute_force_decoding_results = adj_decoding.brute_force_adj_decode(adj_mat, 4)
    assert set(brute_force_decoding_results) == {
        (0, 1, 2, 3),
        (0, 2, 5, 6),
        (0, 4, 5, 6),
        (2, 5, 6, 7),
    }


def test_complex_brute_force_decoding():
    complex_adj_rels = [
        (4, 6, 71, 72, 73),
        (3, 7, 8, 9),
        (4, 7, 8, 9),
        (4, 8, 69, 70),
        (3, 66, 67, 68),
        (3, 51, 54, 55),
        (4, 51, 52, 53),
        (3, 46, 49, 50),
        (4, 46, 47, 48),
        (3, 5, 44, 45),
        (4, 5, 43, 44),
        (3, 38, 41, 42),
        (4, 38, 39, 40),
        (3, 33, 36, 37),
        (4, 33, 34, 35),
        (3, 28, 31, 32),
        (4, 28, 29, 30),
        (3, 23, 26, 27),
        (4, 23, 24, 25),
        (3, 18, 21, 22),
        (4, 18, 19, 20),
        (4, 15, 16, 17),
        (3, 10, 13, 14),
        (4, 10, 11, 12),
    ]
    start = time.time()
    span_rel_adj_obj = SpanRelAdjMat(complex_adj_rels, 74)
    end = time.time() - start
    print("complex (constructed): ", end)

    start = time.time()
    adj_mat = span_rel_adj_obj.reveal_adj_mat()
    print("complex (reveal_adj_mat): ", time.time() - start)

    times = 1000
    start = time.time()
    for _ in range(times):
        adj_decoding.brute_force_adj_decode(adj_mat, 4)
    end = time.time() - start
    print(f"complex ({times} decoding: total, avg): ", end, end / 1000)


def test_complex_brute_force_worst_case_decoding():
    num_spans = 20
    times = 1
    complex_adj_rels = [tuple(range(0, num_spans))]
    span_rel_adj_obj = SpanRelAdjMat(complex_adj_rels, num_spans)
    adj_mat = span_rel_adj_obj.reveal_adj_mat()
    start = time.time()
    for _ in range(times):
        adj_decoding.brute_force_adj_decode(adj_mat, 1)
    end = time.time() - start
    print(f"{num_spans} worst ({times} decoding: total, avg): ", end, end / times)
    # assert set(brute_force_decoding_results), set(complex_adj_rels))


def test_bron_kerbosch():
    span_rel_adj_obj = SpanRelAdjMat([(0, 1, 2, 3), (0, 4, 5, 6), (2, 5, 6, 7)], 8)
    adj_mat = span_rel_adj_obj.reveal_adj_mat()
    brute_force_decoding_results = adj_decoding.bron_kerbosch_decode(adj_mat, 4)
    assert set(brute_force_decoding_results) == {
        (0, 1, 2, 3),
        (0, 2, 5, 6),
        (0, 4, 5, 6),
        (2, 5, 6, 7),
    }


def test_complex_bron_kerbosch_decoding():
    complex_adj_rels = [
        (4, 6, 71, 72, 73),
        (3, 7, 8, 9),
        (4, 7, 8, 9),
        (4, 8, 69, 70),
        (3, 66, 67, 68),
        (3, 51, 54, 55),
        (4, 51, 52, 53),
        (3, 46, 49, 50),
        (4, 46, 47, 48),
        (3, 5, 44, 45),
        (4, 5, 43, 44),
        (3, 38, 41, 42),
        (4, 38, 39, 40),
        (3, 33, 36, 37),
        (4, 33, 34, 35),
        (3, 28, 31, 32),
        (4, 28, 29, 30),
        (3, 23, 26, 27),
        (4, 23, 24, 25),
        (3, 18, 21, 22),
        (4, 18, 19, 20),
        (4, 15, 16, 17),
        (3, 10, 13, 14),
        (4, 10, 11, 12),
    ]
    start = time.time()
    span_rel_adj_obj = SpanRelAdjMat(complex_adj_rels, 74)
    end = time.time() - start
    print("complex (constructed): ", end)

    start = time.time()
    adj_mat = span_rel_adj_obj.reveal_adj_mat()
    print("complex (reveal_adj_mat): ", time.time() - start)

    times = 1000
    start = time.time()
    for _ in range(times):
        adj_decoding.bron_kerbosch_decode(adj_mat, 4)
    end = time.time() - start
    print(f"complex ({times} decoding: total, avg): ", end, end / 1000)


def test_complex_bron_kerbosch_worst_case_decoding():
    num_spans = 20
    times = 10
    complex_adj_rels = [tuple(range(0, num_spans))]
    span_rel_adj_obj = SpanRelAdjMat(complex_adj_rels, num_spans)
    adj_mat = span_rel_adj_obj.reveal_adj_mat()
    start = time.time()
    for _ in range(times):
        adj_decoding.bron_kerbosch_decode(adj_mat, 1)
    end = time.time() - start
    print(f"{num_spans} worst ({times} decoding: total, avg): ", end, end / times)


def test_bron_kerbosch_pivoting():
    span_rel_adj_obj = SpanRelAdjMat([(0, 1, 2, 3), (0, 4, 5, 6), (2, 5, 6, 7)], 8)
    adj_mat = span_rel_adj_obj.reveal_adj_mat()
    brute_force_decoding_results = adj_decoding.bron_kerbosch_pivoting_decode(
        adj_mat, 4
    )
    assert set(brute_force_decoding_results) == {
        (0, 1, 2, 3),
        (0, 2, 5, 6),
        (0, 4, 5, 6),
        (2, 5, 6, 7),
    }


def test_complex_bron_kerbosch_pivoting_decoding():
    complex_adj_rels = [
        (4, 6, 71, 72, 73),
        (3, 7, 8, 9),
        (4, 7, 8, 9),
        (4, 8, 69, 70),
        (3, 66, 67, 68),
        (3, 51, 54, 55),
        (4, 51, 52, 53),
        (3, 46, 49, 50),
        (4, 46, 47, 48),
        (3, 5, 44, 45),
        (4, 5, 43, 44),
        (3, 38, 41, 42),
        (4, 38, 39, 40),
        (3, 33, 36, 37),
        (4, 33, 34, 35),
        (3, 28, 31, 32),
        (4, 28, 29, 30),
        (3, 23, 26, 27),
        (4, 23, 24, 25),
        (3, 18, 21, 22),
        (4, 18, 19, 20),
        (4, 15, 16, 17),
        (3, 10, 13, 14),
        (4, 10, 11, 12),
    ]
    start = time.time()
    span_rel_adj_obj = SpanRelAdjMat(complex_adj_rels, 74)
    end = time.time() - start
    print("complex (constructed): ", end)

    start = time.time()
    adj_mat = span_rel_adj_obj.reveal_adj_mat()
    print("complex (reveal_adj_mat): ", time.time() - start)

    times = 1000
    start = time.time()
    for _ in range(times):
        adj_decoding.bron_kerbosch_pivoting_decode(adj_mat, 4)
    end = time.time() - start
    print(f"complex ({times} decoding: total, avg): ", end, end / 1000)


def test_complex_bron_kerbosch_pivoting_worst_case_decoding():
    num_spans = 20
    times = 100
    complex_adj_rels = [tuple(range(0, num_spans))]
    span_rel_adj_obj = SpanRelAdjMat(complex_adj_rels, num_spans)
    adj_mat = span_rel_adj_obj.reveal_adj_mat()
    cliques = None
    start = time.time()
    for _ in range(times):
        cliques = adj_decoding.bron_kerbosch_pivoting_decode(adj_mat, 1)
    end = time.time() - start
    print(f"{num_spans} worst ({times} decoding: total, avg): ", end, end / times)
    print(cliques)


def test_linked_decode():
    adj_mat = [[0] * 8 for _ in range(8)]
    adj_mat[0][1] = adj_mat[1][3] = adj_mat[2][3] = adj_mat[4][5] = adj_mat[5][
        6
    ] = adj_mat[5][7] = 1
    adj_mat[1][0] = adj_mat[3][1] = adj_mat[3][2] = adj_mat[5][4] = adj_mat[6][
        5
    ] = adj_mat[7][5] = 1
    combs = adj_decoding.linked_decode(adj_mat)
    assert combs == [(0, 1, 2, 3), (4, 5, 6, 7)]


def test_directed_trigger_graph_decode(tmp_event_type_fields_list_for_test):
    event_obj_list = [
        [[None, 5, 6, 2, 3, None, 4, None]],
        None,
        [[1, 0, None, 3, 2, None]],
        None,
        None,
    ]
    adj = AdjMat(
        event_obj_list,
        7,
        tmp_event_type_fields_list_for_test,
        whole_graph=True,
        trigger_aware_graph=True,
        directed_graph=True,
        num_triggers=2,
    )
    assert adj.tolist(-1) == [
        [-1, 0, 0, 0, 0, 0, 0],
        [1, -1, 1, 1, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0],
        [1, 1, 1, -1, 0, 0, 0],
        [0, 0, 0, 0, -1, 0, 0],
        [0, 0, 1, 1, 1, -1, 1],
        [0, 0, 1, 1, 1, 1, -1],
    ]
    assert adj_decoding.directed_trigger_graph_decode(adj.tolist(-1), 2) == [
        (2, 3, 4, 5, 6),
        (0, 1, 2, 3),
    ]


def test_directed_complex_combination_decoding():
    """
    Combinations:
        [1, 4, 5, 6]
        [2, 3, 5, 6]
        [3, 5, 6, 7]
        [2, 4, 5 ,6]
    """
    adj = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    decoded = adj_decoding.directed_trigger_graph_decode(
        adj,
        num_triggers=2,
        max_clique=True,
        with_left_trigger=True,
        with_all_one_trigger_comb=True,
    )
    assert set(decoded) == {(2, 3, 5, 6), (1, 4, 5, 6), (2, 4, 5, 6), (3, 5, 6, 7)}
    decoded = adj_decoding.directed_trigger_graph_decode(
        adj,
        num_triggers=2,
        max_clique=True,
        with_left_trigger=True,
        with_all_one_trigger_comb=False,
    )
    assert set(decoded) == {(2, 3, 5, 6), (1, 4, 5, 6)}
    decoded = adj_decoding.directed_trigger_graph_decode(
        adj,
        num_triggers=2,
        max_clique=True,
        with_left_trigger=False,
        with_all_one_trigger_comb=False,
    )
    assert set(decoded) == {(2, 3, 5, 6)}
