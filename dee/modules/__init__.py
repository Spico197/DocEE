import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import transformer
from .adj_decoding import (
    bron_kerbosch_decode,
    bron_kerbosch_pivoting_decode,
    brute_force_adj_decode,
    directed_trigger_graph_decode,
    directed_trigger_graph_incremental_decode,
    linked_decode,
)
from .biaffine import (
    Biaffine,
    SymmetricBiaffine,
    SymmetricWeightBiaffine,
    SymmetricWeightComponentBiaffine,
    Triaffine,
)
from .doc_info import (
    DocArgRelInfo,
    DocSpanInfo,
    get_doc_arg_rel_info_list,
    get_doc_span_info_list,
    get_span_mention_info,
)
from .dropout import SharedDropout
from .event_table import (
    EventTable,
    EventTableForArgRel,
    EventTableForSigmoidMultiArgRel,
    EventTableWithRNNCell,
)
from .gnn import GAT, GCN, normalize_adj
from .mlp import MLP, SharedDropoutMLP
from .ner_model import (
    BertForBasicNER,
    LSTMBiaffineNERModel,
    LSTMCRFAttNERModel,
    LSTMCRFNERModel,
    LSTMMaskedCRFNERModel,
    NERModel,
    judge_ner_prediction,
)


def get_batch_span_label(num_spans, cur_span_idx_set, device):
    # prepare span labels for this field and this path
    span_field_labels = [
        1 if span_idx in cur_span_idx_set else 0 for span_idx in range(num_spans)
    ]

    batch_field_label = torch.tensor(
        span_field_labels, dtype=torch.long, device=device, requires_grad=False
    )  # [num_spans], val \in {0, 1}

    return batch_field_label


def append_top_span_only(
    last_token_path_list, field_idx, field_idx2span_token_tup2dranges
):
    new_token_path_list = []
    span_token_tup2dranges = field_idx2span_token_tup2dranges[field_idx]
    token_min_drange_list = [
        (token_tup, dranges[0]) for token_tup, dranges in span_token_tup2dranges.items()
    ]
    token_min_drange_list.sort(key=lambda x: x[1])

    for last_token_path in last_token_path_list:
        new_token_path = list(last_token_path)
        if len(token_min_drange_list) == 0:
            new_token_path.append(None)
        else:
            token_tup = token_min_drange_list[0][0]
            new_token_path.append(token_tup)

        new_token_path_list.append(new_token_path)

    return new_token_path_list


def append_all_spans(last_token_path_list, field_idx, field_idx2span_token_tup2dranges):
    new_token_path_list = []
    span_token_tup2dranges = field_idx2span_token_tup2dranges[field_idx]

    for last_token_path in last_token_path_list:
        for token_tup in span_token_tup2dranges.keys():
            new_token_path = list(last_token_path)
            new_token_path.append(token_tup)
            new_token_path_list.append(new_token_path)

        if len(span_token_tup2dranges) == 0:  # ensure every last path will be extended
            new_token_path = list(last_token_path)
            new_token_path.append(None)
            new_token_path_list.append(new_token_path)

    return new_token_path_list


class AttentiveReducer(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(AttentiveReducer, self).__init__()

        self.hidden_size = hidden_size
        self.att_norm = math.sqrt(self.hidden_size)

        self.fc = nn.Linear(hidden_size, 1, bias=False)
        self.att = None

        self.layer_norm = transformer.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_token_emb, masks=None, keepdim=False):
        # batch_token_emb: Size([*, seq_len, hidden_size])
        # masks: Size([*, seq_len]), 1: normal, 0: pad

        query = self.fc.weight
        if masks is None:
            att_mask = None
        else:
            att_mask = masks.unsqueeze(-2)  # [*, 1, seq_len]

        # batch_att_emb: Size([*, 1, hidden_size])
        # self.att: Size([*, 1, seq_len])
        batch_att_emb, self.att = transformer.attention(
            query, batch_token_emb, batch_token_emb, mask=att_mask
        )

        batch_att_emb = self.dropout(self.layer_norm(batch_att_emb))

        if keepdim:
            return batch_att_emb
        else:
            return batch_att_emb.squeeze(-2)

    def extra_repr(self):
        return "hidden_size={}, att_norm={}".format(self.hidden_size, self.att_norm)


class SentencePosEncoder(nn.Module):
    def __init__(self, hidden_size, max_sent_num=100, dropout=0.1):
        super(SentencePosEncoder, self).__init__()

        self.embedding = nn.Embedding(max_sent_num, hidden_size)
        self.layer_norm = transformer.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_elem_emb, sent_pos_ids=None):
        if sent_pos_ids is None:
            num_elem = batch_elem_emb.size(-2)
            sent_pos_ids = torch.arange(
                num_elem,
                dtype=torch.long,
                device=batch_elem_emb.device,
                requires_grad=False,
            )
        elif not isinstance(sent_pos_ids, torch.Tensor):
            sent_pos_ids = torch.tensor(
                sent_pos_ids,
                dtype=torch.long,
                device=batch_elem_emb.device,
                requires_grad=False,
            )

        batch_pos_emb = self.embedding(sent_pos_ids)
        out = batch_elem_emb + batch_pos_emb
        out = self.dropout(self.layer_norm(out))

        return out


class MentionTypeEncoder(nn.Module):
    def __init__(self, hidden_size, num_ment_types, dropout=0.1):
        super(MentionTypeEncoder, self).__init__()

        self.embedding = nn.Embedding(num_ment_types, hidden_size)
        self.layer_norm = transformer.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_mention_emb, mention_type_ids):
        if not isinstance(mention_type_ids, torch.Tensor):
            mention_type_ids = torch.tensor(
                mention_type_ids,
                dtype=torch.long,
                device=batch_mention_emb.device,
                requires_grad=False,
            )

        batch_mention_type_emb = self.embedding(mention_type_ids)
        out = batch_mention_emb + batch_mention_type_emb
        out = self.dropout(self.layer_norm(out))

        return out


class MentionTypePluser(nn.Module):
    def __init__(self, hidden_size, num_ment_types):
        super().__init__()
        self.embedding = nn.Embedding(num_ment_types, hidden_size)

    def forward(self, batch_mention_emb, mention_type_ids):
        if not isinstance(mention_type_ids, torch.Tensor):
            mention_type_ids = torch.tensor(
                mention_type_ids,
                dtype=torch.long,
                device=batch_mention_emb.device,
                requires_grad=False,
            )

        batch_mention_type_emb = self.embedding(mention_type_ids)
        out = batch_mention_emb + batch_mention_type_emb

        return out


class MentionTypeConcatEncoder(nn.Module):
    def __init__(self, hidden_size, num_ment_types, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(num_ment_types, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_mention_emb, mention_type_ids):
        if not isinstance(mention_type_ids, torch.Tensor):
            mention_type_ids = torch.tensor(
                mention_type_ids,
                dtype=torch.long,
                device=batch_mention_emb.device,
                requires_grad=False,
            )

        batch_mention_type_emb = self.embedding(mention_type_ids)
        out = torch.cat([batch_mention_emb, batch_mention_type_emb], dim=-1)
        out = self.dropout(out)

        return out


class MentionTypeEncoderWithMentionEmbReturning(nn.Module):
    def __init__(self, hidden_size, num_ment_types, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(num_ment_types, hidden_size)
        self.layer_norm = transformer.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_mention_emb, mention_type_ids):
        if not isinstance(mention_type_ids, torch.Tensor):
            mention_type_ids = torch.tensor(
                mention_type_ids,
                dtype=torch.long,
                device=batch_mention_emb.device,
                requires_grad=False,
            )

        batch_mention_type_emb = self.embedding(mention_type_ids)
        out = batch_mention_emb + batch_mention_type_emb
        out = self.dropout(self.layer_norm(out))

        return out, batch_mention_type_emb


class EmbPlusEncoder(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(EmbPlusEncoder, self).__init__()

        self.layer_norm = transformer.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, rep_emb1, rep_emb2):
        out = rep_emb1 + rep_emb2
        out = self.dropout(self.layer_norm(out))

        return out


class GatedFusion(nn.Module):
    r"""
    Reference:
        - ACL2020, Document-Level Event Role Filler Extraction using Multi-Granularity Contextualized Encoding
    """

    def __init__(self, n_in):
        super().__init__()
        self.n_in = n_in
        self.hidden2scalar1 = nn.Linear(self.n_in, 1)
        self.hidden2scalar2 = nn.Linear(self.n_in, 1)

    def forward(self, hidden1, hidden2):
        gate_alpha = torch.sigmoid(
            self.hidden2scalar1(hidden1) + self.hidden2scalar2(hidden2)
        )
        out = gate_alpha * hidden1 + (1 - gate_alpha) * hidden2
        return out
