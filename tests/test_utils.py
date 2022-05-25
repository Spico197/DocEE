from dee.utils import (
    extract_combinations_from_event_objs,
    list_flatten,
    merge_non_conflicting_ins_objs,
    remove_combination_roles,
    tril_fold_or,
)


def test_list_flatten():
    sents = [
        [
            "Three",
            "specific",
            "points",
            "illustrate",
            "why",
            "Americans",
            "see",
            "Trump",
            "as",
            "the",
            "problem",
            ":",
        ],
        [
            "1",
            ")",
            "Trump",
            "has",
            "trouble",
            "working",
            "with",
            "people",
            "beyond",
            "his",
            "base",
            ".",
        ],
        [
            "In",
            "Saddam",
            "Hussein",
            "'s",
            "Iraq",
            "that",
            "might",
            "work",
            "when",
            "opponents",
            "can",
            "be",
            "thrown",
            "in",
            "jail",
            "or",
            "exterminated",
            ".",
        ],
        [
            "In",
            "the",
            "United",
            "States",
            "that",
            "wo",
            "n't",
            "fly",
            ":",
            "presidents",
            "must",
            "build",
            "bridges",
            "within",
            "and",
            "beyond",
            "their",
            "core",
            "support",
            "to",
            "resolve",
            "challenges",
            ".",
        ],
        [
            "Without",
            "alliances",
            ",",
            "a",
            "president",
            "ca",
            "n't",
            "get",
            "approval",
            "to",
            "get",
            "things",
            "done",
            ".",
        ],
    ]
    total_sents, len_mapping = list_flatten(sents)
    gold_sents = (
        [
            "Three",
            "specific",
            "points",
            "illustrate",
            "why",
            "Americans",
            "see",
            "Trump",
            "as",
            "the",
            "problem",
            ":",
        ]
        + [
            "1",
            ")",
            "Trump",
            "has",
            "trouble",
            "working",
            "with",
            "people",
            "beyond",
            "his",
            "base",
            ".",
        ]
        + [
            "In",
            "Saddam",
            "Hussein",
            "'s",
            "Iraq",
            "that",
            "might",
            "work",
            "when",
            "opponents",
            "can",
            "be",
            "thrown",
            "in",
            "jail",
            "or",
            "exterminated",
            ".",
        ]
        + [
            "In",
            "the",
            "United",
            "States",
            "that",
            "wo",
            "n't",
            "fly",
            ":",
            "presidents",
            "must",
            "build",
            "bridges",
            "within",
            "and",
            "beyond",
            "their",
            "core",
            "support",
            "to",
            "resolve",
            "challenges",
            ".",
        ]
        + [
            "Without",
            "alliances",
            ",",
            "a",
            "president",
            "ca",
            "n't",
            "get",
            "approval",
            "to",
            "get",
            "things",
            "done",
            ".",
        ]
    )
    sent_lens = [len(sent) for sent in sents]
    total_lens = []
    for sent_idx, length in enumerate(sent_lens):
        total_lens += [[sent_idx, i] for i in range(length)]

    assert total_sents == gold_sents
    assert len_mapping == total_lens


def test_combination_extraction():
    # with roles
    event_obj_list = [
        None,
        None,
        None,
        None,
        [
            ((3, 0), (7, 1), (5, 2), (9, 3), (10, 4), (11, 5), (4, 6), (8, 7)),
            ((3, 0), (5, 2), (9, 3), (11, 5), (4, 6), (12, 7), (6, 8)),
        ],
    ]
    gold_combinations = {(3, 4, 5, 7, 8, 9, 10, 11), (3, 4, 5, 6, 9, 11, 12)}
    extracted_combinations = extract_combinations_from_event_objs(event_obj_list)
    removed_roles_combinations = remove_combination_roles(extracted_combinations)
    assert gold_combinations == removed_roles_combinations


def test_comb_case2():
    event_obj_list = [
        None,
        None,
        None,
        None,
        [
            [
                (3330, 1290, 7471),
                (122, 122, 129, 129, 127, 121, 121, 5500),
                (3862, 6858, 6395, 1171, 5500, 819, 3300, 7361, 1062, 1385),
                (123, 123, 127, 122, 130, 130, 130, 130, 5500),
                (127, 119, 125, 122, 110),
                (122, 129, 123, 121, 121, 121, 121, 121, 5500),
                (123, 121, 122, 129, 2399, 130, 3299, 127, 3189),
                None,
                None,
            ],
            [
                (3330, 1290, 7471),
                (122, 123, 122, 126, 122, 121, 121, 121, 5500),
                (3862, 6858, 6395, 1171, 5500, 819, 3300, 7361, 1062, 1385),
                None,
                (127, 119, 125, 122, 110),
                (122, 123, 122, 126, 122, 121, 121, 121, 5500),
                (123, 121, 122, 128, 2399, 122, 123, 3299, 128, 3189),
                None,
                None,
            ],
        ],
    ]
    extracted_combinations = extract_combinations_from_event_objs(event_obj_list)
    assert extracted_combinations == {
        (
            (122, 122, 129, 129, 127, 121, 121, 5500),
            (122, 129, 123, 121, 121, 121, 121, 121, 5500),
            (123, 121, 122, 129, 2399, 130, 3299, 127, 3189),
            (123, 123, 127, 122, 130, 130, 130, 130, 5500),
            (127, 119, 125, 122, 110),
            (3330, 1290, 7471),
            (3862, 6858, 6395, 1171, 5500, 819, 3300, 7361, 1062, 1385),
        ),
        (
            (122, 123, 122, 126, 122, 121, 121, 121, 5500),
            (123, 121, 122, 128, 2399, 122, 123, 3299, 128, 3189),
            (127, 119, 125, 122, 110),
            (3330, 1290, 7471),
            (3862, 6858, 6395, 1171, 5500, 819, 3300, 7361, 1062, 1385),
        ),
    }


def test_conflicting_merge_ins_objs():
    objs = [
        [
            (3330, 1290, 7471),
            (122, 122, 129, 129, 127, 121, 121, 5500),
            (3862, 6858, 6395, 1171, 5500, 819, 3300, 7361, 1062, 1385),
            (123, 123, 127, 122, 130, 130, 130, 130, 5500),
            (127, 119, 125, 122, 110),
            (122, 129, 123, 121, 121, 121, 121, 121, 5500),
            (123, 121, 122, 129, 2399, 130, 3299, 127, 3189),
            None,
            None,
        ],
        [
            (3330, 1290, 7471),
            (122, 123, 122, 126, 122, 121, 121, 121, 5500),
            (3862, 6858, 6395, 1171, 5500, 819, 3300, 7361, 1062, 1385),
            None,
            (127, 119, 125, 122, 110),
            (122, 123, 122, 126, 122, 121, 121, 121, 5500),
            (123, 121, 122, 128, 2399, 122, 123, 3299, 128, 3189),
            None,
            None,
        ],
    ]
    new_objs = merge_non_conflicting_ins_objs(objs)
    assert objs == new_objs


def test_mergable_merge_ins_objs():
    objs = [
        [
            (3330, 1290, 7471),
            (122, 122, 129, 129, 127, 121, 121, 5500),
            (3862, 6858, 6395, 1171, 5500, 819, 3300, 7361, 1062, 1385),
            (123, 123, 127, 122, 130, 130, 130, 130, 5500),
            (127, 119, 125, 122, 110),
            (122, 129, 123, 121, 121, 121, 121, 121, 5500),
            (123, 121, 122, 129, 2399, 130, 3299, 127, 3189),
            None,
            None,
        ],
        [
            (3330, 1290, 7471),
            (122, 122, 129, 129, 127, 121, 121, 5500),
            (3862, 6858, 6395, 1171, 5500, 819, 3300, 7361, 1062, 1385),
            None,
            (127, 119, 125, 122, 110),
            (122, 129, 123, 121, 121, 121, 121, 121, 5500),
            (123, 121, 122, 129, 2399, 130, 3299, 127, 3189),
            None,
            None,
        ],
    ]
    result_objs = [
        [
            (3330, 1290, 7471),
            (122, 122, 129, 129, 127, 121, 121, 5500),
            (3862, 6858, 6395, 1171, 5500, 819, 3300, 7361, 1062, 1385),
            (123, 123, 127, 122, 130, 130, 130, 130, 5500),
            (127, 119, 125, 122, 110),
            (122, 129, 123, 121, 121, 121, 121, 121, 5500),
            (123, 121, 122, 129, 2399, 130, 3299, 127, 3189),
            None,
            None,
        ]
    ]
    merged_objs = merge_non_conflicting_ins_objs(objs, min_coo=2)
    assert merged_objs == result_objs


def test_tril_fold_or():
    mat = [
        [-1, 1, 1, 1, 0, 0, 0],
        [1, -1, 1, 1, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, 0, -1, 0, 0],
        [0, 0, 1, 1, 1, -1, 1],
        [0, 0, 1, 1, 1, 1, -1],
    ]

    fold_mat = tril_fold_or(mat)
    assert fold_mat == [
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 1, 0, 1],
        [0, 0, 1, 1, 1, 1, 0],
    ]
