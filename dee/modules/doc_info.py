from collections import OrderedDict, defaultdict, namedtuple

from dee.utils import regex_extractor


def get_span_mention_info(span_dranges_list, doc_token_type_list):
    span_mention_range_list = []
    mention_drange_list = []
    mention_type_list = []
    for span_dranges in span_dranges_list:
        ment_idx_s = len(mention_drange_list)
        for drange in span_dranges:
            mention_drange_list.append(drange)
            sent_idx, char_s, char_e = drange
            mention_type_list.append(doc_token_type_list[sent_idx][char_s])
        ment_idx_e = len(mention_drange_list)
        span_mention_range_list.append((ment_idx_s, ment_idx_e))

    return span_mention_range_list, mention_drange_list, mention_type_list


def extract_doc_valid_span_info(doc_token_type_mat, doc_fea):
    # tzhu: extract span from predicted/gold token type labels
    doc_token_id_mat = doc_fea.doc_token_ids.tolist()
    doc_token_mask_mat = doc_fea.doc_token_masks.tolist()

    # [(token_id_tuple, (sent_idx, char_s, char_e)), ...]
    span_token_drange_list = []

    valid_sent_num = doc_fea.valid_sent_num
    for sent_idx in range(valid_sent_num):
        seq_token_id_list = doc_token_id_mat[sent_idx]
        seq_token_mask_list = doc_token_mask_mat[sent_idx]
        seq_token_type_list = doc_token_type_mat[sent_idx]
        seq_len = len(seq_token_id_list)

        char_s = 0
        while char_s < seq_len:
            # tzhu: if token is pad, then it means the sentence has come to an end
            if seq_token_mask_list[char_s] == 0:
                break

            entity_idx = seq_token_type_list[char_s]

            if entity_idx % 2 == 1:  # tzhu: if entity is start with `B-`
                char_e = char_s + 1
                # tzhu: when former entity label is started with `I-`
                while (
                    char_e < seq_len
                    and seq_token_mask_list[char_e] == 1
                    and seq_token_type_list[char_e] == entity_idx + 1
                ):
                    char_e += 1

                token_tup = tuple(seq_token_id_list[char_s:char_e])
                drange = (sent_idx, char_s, char_e)

                # if len(token_tup) >= 2:
                # if token_tup in doc_fea.span_token_ids_list:
                span_token_drange_list.append((token_tup, drange))

                char_s = char_e
            else:
                char_s += 1

    span_token_drange_list.sort(
        key=lambda x: x[-1]
    )  # sorted by drange = (sent_idx, char_s, char_e)
    # drange is exclusive and sorted
    token_tup2dranges = OrderedDict()
    for token_tup, drange in span_token_drange_list:
        if token_tup not in token_tup2dranges:
            token_tup2dranges[token_tup] = []
        token_tup2dranges[token_tup].append(drange)

    span_token_tup_list = list(token_tup2dranges.keys())
    span_dranges_list = list(token_tup2dranges.values())

    return span_token_tup_list, span_dranges_list


DocSpanInfo = namedtuple(
    "DocSpanInfo",
    (
        "span_token_tup_list",  # [(span_token_id, ...), ...], num_spans
        "gold_span_token_ids_list",  # list of gold span token-ids-tuple
        "span_dranges_list",  # [[(sent_idx, char_s, char_e), ...], ...], num_spans
        "span_mention_range_list",  # [(mention_idx_s, mention_idx_e), ...], num_spans
        "mention_drange_list",  # [(sent_idx, char_s, char_e), ...], num_mentions
        "mention_type_list",  # [mention_type_id, ...], num_mentions
        "event_dag_info",  # event_idx -> field_idx -> pre_path -> cur_span_idx_set
        "missed_sent_idx_list",  # index list of sentences where gold spans are not extracted
    ),
)


def get_doc_span_info_list(doc_token_types_list, doc_fea_list, use_gold_span=False):
    assert len(doc_token_types_list) == len(doc_fea_list)
    doc_span_info_list = []
    for doc_token_types, doc_fea in zip(doc_token_types_list, doc_fea_list):
        doc_token_type_mat = doc_token_types.tolist()  # [[token_type, ...], ...]

        # using extracted results is also ok
        # span_token_tup_list, span_dranges_list = extract_doc_valid_span_info(doc_token_type_mat, doc_fea)
        if use_gold_span:
            span_token_tup_list = doc_fea.span_token_ids_list
            span_dranges_list = doc_fea.span_dranges_list
        else:
            span_token_tup_list, span_dranges_list = extract_doc_valid_span_info(
                doc_token_type_mat, doc_fea
            )
            """
            DONE(tzhu): check the availability to use gold_span while evaluating
            it is ok to write this, although there is still an evaluation risk,
            refer to: [Discussion in Github](https://github.com/dolphin-zs/Doc2EDAG/issues/19)
            """
            if len(span_token_tup_list) == 0:
                # do not get valid entity span results,
                # just use gold spans to avoid crashing at earlier iterations
                # TODO: consider generate random negative spans
                span_token_tup_list = doc_fea.span_token_ids_list
                span_dranges_list = doc_fea.span_dranges_list

        # one span may have multiple mentions
        # tzhu: just flatten the dranges and mentions from sentence-independent data orgnisation format to flat list format
        (
            span_mention_range_list,
            mention_drange_list,
            mention_type_list,
        ) = get_span_mention_info(span_dranges_list, doc_token_type_mat)

        # generate event decoding dag graph for model training
        event_dag_info, _, missed_sent_idx_list = doc_fea.generate_dag_info_for(
            span_token_tup_list, return_miss=True
        )

        # doc_span_info will incorporate all span-level information needed for the event extraction
        doc_span_info = DocSpanInfo(
            span_token_tup_list,
            doc_fea.span_token_ids_list,
            span_dranges_list,
            span_mention_range_list,
            mention_drange_list,
            mention_type_list,
            event_dag_info,
            missed_sent_idx_list,
        )

        doc_span_info_list.append(doc_span_info)

    return doc_span_info_list


DocArgRelInfo = namedtuple(
    "DocArgRelInfo",
    (
        # [(span_token_id, ...), ...], num_spans
        "span_token_tup_list",
        # [1, 0, 1, ...], num_spans, span exist in instances
        "span_token_tup_exist_list",
        # span types
        # `0`: non exist (dependent nodes, 0-degree)
        # `1`: exist (not shared nodes, regular sub-graph)
        # `2`: exist and shared (more degree than sub-graph nodes)
        # `3`: non exist and wrongly predicted (not shared nodes, wrongly predicted, 0-degree)
        "span_token_tup_type_list",
        # list of gold span token-ids-tuple
        "gold_span_token_ids_list",
        # [[(sent_idx, char_s, char_e), ...], ...], num_spans
        "span_dranges_list",
        # [(mention_idx_s, mention_idx_e), ...], num_spans
        "span_mention_range_list",
        # [(sent_idx, char_s, char_e), ...], num_mentions
        "mention_drange_list",
        # [mention_type_id, ...], num_mentions
        "mention_type_list",
        # [SpanRelAdjMat(), None, SpanRelAdjMat(), ...]
        "event_arg_rel_mats",
        # for event-irrelevant scenario, SpanRelAdjMat()
        "whole_arg_rel_mat",
        # predictions and gold intersection for further role classification
        # [None, None, ..., [((1, 0), (2, 1)), (...)]]
        "pred_event_arg_idxs_objs_list",
        # missed span idx list
        "missed_span_idx_list",
        # index list of sentences where gold spans are not extracted
        "missed_sent_idx_list",
    ),
)


def _is_overlapping(part, whole):
    return part == whole[0 : len(part)] or part == whole[len(whole) - len(part) :]


def _check_and_fix(span_token_tup, span_drange, pred_field, complementary_field2ents):
    """
    check if span_token_tup is in complementary_field2ents and fix result

    Returns:
        bool: if span_token_tup is in complementary ents
        span_token_tup: fixed result
        span_drange: fixed drange
    """
    sent_idx = span_drange[0]
    field_type = regex_extractor.field2type[
        regex_extractor.field_id2field_name[pred_field]
    ]
    ents = complementary_field2ents[field_type]
    in_ents = False
    # entities in the same sentence
    ents_same_sentence = []
    for ent, ent_drange in filter(lambda x: x[1][0] == sent_idx, ents):
        ents_same_sentence.append([ent, ent_drange])
        if ent == span_token_tup:
            in_ents = True
            # does not need to fix
            return in_ents, span_token_tup, ent_drange
    # if ent is not in complementary_field2ents, consider fixing
    for ent, ent_drange in ents_same_sentence:
        if _is_overlapping(span_token_tup, ent):
            return in_ents, ent, ent_drange
    return in_ents, span_token_tup, span_drange


def fix_ent(
    span_token_tup_list, span_dranges_list, doc_token_type_mat, doc_fea, ent_fix_mode
):
    if ent_fix_mode == "n":
        return span_token_tup_list, span_dranges_list

    span2dranges = defaultdict(set)
    for span_token_tup, span_dranges in zip(span_token_tup_list, span_dranges_list):
        for span_drange in span_dranges:
            pred_field = doc_token_type_mat[span_drange[0]][span_drange[1]]
            if pred_field in regex_extractor.field_id2field_name:
                in_ents, fixed_span_token_tup, fixed_ent_drange = _check_and_fix(
                    span_token_tup,
                    span_drange,
                    pred_field,
                    doc_fea.complementary_field2ents,
                )
                if in_ents:
                    span2dranges[span_token_tup].add(span_drange)
                else:
                    if ent_fix_mode == "f":
                        span2dranges[fixed_span_token_tup].add(fixed_ent_drange)
                    elif ent_fix_mode == "-":
                        pass
            else:
                span2dranges[span_token_tup].add(span_drange)
    for span, dranges in span2dranges.items():
        span2dranges[span] = sorted(dranges)
    return list(span2dranges.keys()), list(span2dranges.values())


def get_doc_arg_rel_info_list(
    doc_token_types_list,
    doc_fea_list,
    event_type_fields_list,
    use_gold_span=False,
    ent_fix_mode="n",
    force_return_none_role_entity=False,
):
    assert len(doc_token_types_list) == len(doc_fea_list)
    doc_arg_rel_info_list = []
    for doc_token_types, doc_fea in zip(doc_token_types_list, doc_fea_list):
        doc_token_type_mat = doc_token_types.tolist()  # [[token_type, ...], ...]

        # using extracted results is also ok
        # span_token_tup_list, span_dranges_list = extract_doc_valid_span_info(doc_token_type_mat, doc_fea)
        if use_gold_span:
            span_token_tup_list = doc_fea.span_token_ids_list
            span_dranges_list = doc_fea.span_dranges_list
        else:
            span_token_tup_list, span_dranges_list = extract_doc_valid_span_info(
                doc_token_type_mat, doc_fea
            )
            # DONE(tzhu): check the availability to use gold_span while evaluating
            # it is ok to write this, although there is still an evaluation risk,
            # refer to: [Discussion in Github](https://github.com/dolphin-zs/Doc2EDAG/issues/19)
            if len(span_token_tup_list) == 0:
                # do not get valid entity span results,
                # just use gold spans to avoid crashing at earlier iterations
                # TODO: consider generate random negative spans
                span_token_tup_list = doc_fea.span_token_ids_list
                span_dranges_list = doc_fea.span_dranges_list
            else:
                if ent_fix_mode != "n":
                    span_token_tup_list, span_dranges_list = fix_ent(
                        span_token_tup_list,
                        span_dranges_list,
                        doc_token_type_mat,
                        doc_fea,
                        ent_fix_mode,
                    )

        # one span may have multiple mentions
        # tzhu: just flatten the dranges and mentions from sentence-independent data orgnisation format to flat list format
        (
            span_mention_range_list,
            mention_drange_list,
            mention_type_list,
        ) = get_span_mention_info(span_dranges_list, doc_token_type_mat)

        # generate event decoding adj mat for model training
        # if using the predicted results, the span list must has been changed
        # to keep the harmony, must generate new arg rel mat from the predicted spans

        # if force_return_none_role_entity:
        #     event_arg_rel_mats, whole_arg_rel_mat, pred_event_arg_idxs_objs_list, \
        #         _, missed_sent_idx_list = doc_fea.generate_arg_rel_mat_with_none_for(span_token_tup_list, return_miss=True)
        # else:
        (
            event_arg_rel_mats,
            whole_arg_rel_mat,
            pred_event_arg_idxs_objs_list,
            missed_span_idx_list,
            missed_sent_idx_list,
        ) = doc_fea.generate_arg_rel_mat_for(
            span_token_tup_list, event_type_fields_list, return_miss=True
        )

        # span exist in any sub-graphs
        span_token_tup_exist_list = []

        # span types
        # `0`: non exist (dependent nodes, 0-degree)
        # `1`: exist (not shared nodes, regular sub-graph)
        # `2`: exist and shared (more degree than sub-graph nodes)
        # `3`: non exist and wrongly predicted (not shared nodes, wrongly predicted, 0-degree)
        span_token_tup_type_list = []
        for x in span_token_tup_list:
            span_token_tup_exist_list.append(x in doc_fea.exist_span_token_tup_set)
            if x not in doc_fea.span_token_ids_list:
                # wrongly predicted
                span_token_tup_type_list.append(3)
            else:
                span_token_tup_type_list.append(doc_fea.span_token_tup2type[x])

        # doc_span_info will incorporate all span-level information needed for the event extraction
        doc_arg_rel_info = DocArgRelInfo(
            span_token_tup_list,
            span_token_tup_exist_list,
            span_token_tup_type_list,
            doc_fea.span_token_ids_list,
            span_dranges_list,
            span_mention_range_list,
            mention_drange_list,
            mention_type_list,
            event_arg_rel_mats,
            whole_arg_rel_mat,
            pred_event_arg_idxs_objs_list,
            missed_span_idx_list,
            missed_sent_idx_list,
        )

        doc_arg_rel_info_list.append(doc_arg_rel_info)

    return doc_arg_rel_info_list


DEPPNDocSpanInfo = namedtuple(
    "DEPPNDocSpanInfo",
    (
        "span_token_tup_list",  # [(span_token_id, ...), ...], num_spans
        "span_dranges_list",  # [[(sent_idx, char_s, char_e), ...], ...], num_spans
        "span_mention_range_list",  # [(mention_idx_s, mention_idx_e), ...], num_spans
        "mention_drange_list",  # [(sent_idx, char_s, char_e), ...], num_mentions
        "mention_type_list",  # [mention_type_id, ...], num_mentions
        "gold_span_idx2pred_span_idx",
        "pred_event_arg_idxs_objs_list",
        "pred_event_type_idxs_list",
    ),
)


def get_deppn_doc_span_info_list(
    doc_token_types_list, doc_fea_list, use_gold_span=False
):
    assert len(doc_token_types_list) == len(doc_fea_list)
    doc_span_info_list = []
    for doc_token_types, doc_fea in zip(doc_token_types_list, doc_fea_list):
        doc_token_type_mat = doc_token_types.tolist()  # [[token_type, ...], ...]
        # print(doc_token_type_mat)
        # using extracted results is also ok
        # span_token_tup_list, span_dranges_list = extract_doc_valid_span_info(doc_token_type_mat, doc_fea)
        if use_gold_span:
            span_token_tup_list = doc_fea.span_token_ids_list
            span_dranges_list = doc_fea.span_dranges_list
        else:
            span_token_tup_list, span_dranges_list = extract_doc_valid_span_info(
                doc_token_type_mat, doc_fea
            )
            # span_token_tup_list
            # print(len(span_token_tup_list), span_token_tup_list)
            # print(len(span_dranges_list), span_dranges_list)
            if len(span_token_tup_list) == 0:
                # do not get valid entity span results,
                # just use gold spans to avoid crashing at earlier iterations
                # TODO: consider generate random negative spans
                span_token_tup_list = doc_fea.span_token_ids_list
                span_dranges_list = doc_fea.span_dranges_list

        # one span may have multiple mentions
        (
            span_mention_range_list,
            mention_drange_list,
            mention_type_list,
        ) = get_span_mention_info(span_dranges_list, doc_token_type_mat)
        # generate event decoding dag graph for model training
        (
            gold_span_idx2pred_span_idx,
            pred_event_arg_idxs_objs_list,
            pred_event_type_idxs_list,
        ) = doc_fea.generate_dag_info_for(span_token_tup_list, return_miss=True)

        # doc_span_info will incorporate all span-level information needed for the event extraction
        doc_span_info = DEPPNDocSpanInfo(
            span_token_tup_list,
            span_dranges_list,
            span_mention_range_list,
            mention_drange_list,
            mention_type_list,
            gold_span_idx2pred_span_idx,
            pred_event_arg_idxs_objs_list,
            pred_event_type_idxs_list,
        )
        doc_span_info_list.append(doc_span_info)

    return doc_span_info_list
