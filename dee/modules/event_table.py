import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dee.modules.mlp import MLP
from dee.modules.transformer import attention


class EventTable(nn.Module):
    def __init__(self, event_type, field_types, hidden_size):
        super(EventTable, self).__init__()

        self.event_type = event_type
        self.field_types = field_types
        self.num_fields = len(field_types)
        self.hidden_size = hidden_size

        self.event_cls = nn.Linear(hidden_size, 2)  # 0: NA, 1: trigger this event
        self.field_cls_list = nn.ModuleList(
            # 0: NA, 1: trigger this field
            [nn.Linear(hidden_size, 2) for _ in range(self.num_fields)]
        )

        # used to aggregate sentence and span embedding
        self.event_query = nn.Parameter(torch.Tensor(1, self.hidden_size))
        # used for fields that do not contain any valid span
        # self.none_span_emb = nn.Parameter(torch.Tensor(1, self.hidden_size))
        # used for aggregating history filled span info
        self.field_queries = nn.ParameterList(
            [
                nn.Parameter(torch.Tensor(1, self.hidden_size))
                for _ in range(self.num_fields)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        self.event_query.data.uniform_(-stdv, stdv)
        # self.none_span_emb.data.uniform_(-stdv, stdv)
        for fq in self.field_queries:
            fq.data.uniform_(-stdv, stdv)

    def forward(self, sent_context_emb=None, batch_span_emb=None, field_idx=None):
        assert (sent_context_emb is None) ^ (batch_span_emb is None)

        if sent_context_emb is not None:  # [num_spans+num_sents, hidden_size]
            # tzhu: every sentence has an attention score, and document embedding is a attention-based weighted sentence representation
            # doc_emb.size = [1, hidden_size]
            doc_emb, _ = attention(self.event_query, sent_context_emb, sent_context_emb)
            doc_pred_logits = self.event_cls(doc_emb)
            doc_pred_logp = F.log_softmax(doc_pred_logits, dim=-1)

            return doc_pred_logp

        if batch_span_emb is not None:
            assert field_idx is not None
            # span_context_emb: [batch_size, hidden_size] or [hidden_size]
            if batch_span_emb.dim() == 1:
                batch_span_emb = batch_span_emb.unsqueeze(0)
            span_pred_logits = self.field_cls_list[field_idx](batch_span_emb)
            span_pred_logp = F.log_softmax(span_pred_logits, dim=-1)

            return span_pred_logp

    def extra_repr(self):
        return "event_type={}, num_fields={}, hidden_size={}".format(
            self.event_type, self.num_fields, self.hidden_size
        )


class EventTableWithRNNCell(EventTable):
    def __init__(self, event_type, field_types, hidden_size):
        super().__init__(event_type, field_types, hidden_size)

        self.rnn_cell = nn.LSTMCell(
            input_size=self.hidden_size, hidden_size=self.hidden_size
        )


class EventTableForArgRel(nn.Module):
    """build for event and corresponding roles classification
    simpler and fewer parameters than original EventTable in Doc2EDAG
    """

    def __init__(self, event_type, field_types, hidden_size, min_field_num):
        super(EventTableForArgRel, self).__init__()

        self.event_type = event_type
        self.field_types = field_types
        self.num_fields = len(field_types)
        self.min_field_num = min_field_num
        self.hidden_size = hidden_size

        # for event classification: 0: NA, 1: trigger this event
        self.event_cls = nn.Linear(hidden_size, 2)
        # for event attention with sentences
        self.event_query = nn.Parameter(torch.Tensor(1, self.hidden_size))
        # used for field classification (+1: None field situation in pred_span mode)
        self.field_cls = nn.Linear(hidden_size, self.num_fields)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        self.event_query.data.uniform_(-stdv, stdv)
        self.field_cls.reset_parameters()

    def forward(self, sent_context_emb=None, batch_span_emb=None):
        assert (sent_context_emb is None) ^ (batch_span_emb is None)

        if sent_context_emb is not None:  # [num_spans+num_sents, hidden_size]
            # tzhu: every sentence has an attention score,
            # and document embedding is a attention-based
            # weighted sentence representation
            # doc_emb.size = [1, hidden_size]
            doc_emb, _ = attention(self.event_query, sent_context_emb, sent_context_emb)
            doc_pred_logits = self.event_cls(doc_emb)
            doc_pred_logp = F.log_softmax(doc_pred_logits, dim=-1)

            return doc_pred_logp

        if batch_span_emb is not None:
            # span_context_emb: [batch_size, hidden_size] or [hidden_size]
            if batch_span_emb.dim() == 1:
                batch_span_emb = batch_span_emb.unsqueeze(0)
            span_pred_logits = self.field_cls(batch_span_emb)
            span_pred_logp = F.log_softmax(span_pred_logits, dim=-1)

            return span_pred_logp

    def extra_repr(self):
        return "event_type={}, num_fields={}, hidden_size={}".format(
            self.event_type, self.num_fields, self.hidden_size
        )


class EventTableForSigmoidMultiArgRel(nn.Module):
    """build for event and corresponding roles classification
    simpler and fewer parameters than original EventTable in Doc2EDAG

    Compared with `EventTableForArgRel`, this module is multi-class multi-label
    classification, which supports one entity to have multiple fields.
    """

    def __init__(
        self,
        event_type,
        field_types,
        dim_event_query,
        hidden_size,
        min_field_num,
        threshold=0.5,
        use_field_cls_mlp=False,
        dropout=0.1,
    ):
        super(EventTableForSigmoidMultiArgRel, self).__init__()

        self.event_type = event_type
        self.field_types = field_types
        self.num_fields = len(field_types)
        self.min_field_num = min_field_num
        self.dim_event_query = dim_event_query
        self.hidden_size = hidden_size

        # for event classification: 0: NA, 1: trigger this event
        self.event_cls = nn.Linear(dim_event_query, 2)
        # for event attention with sentences
        self.event_query = nn.Parameter(torch.Tensor(1, dim_event_query))
        # used for field classification (+1: None field situation in pred_span mode)

        if use_field_cls_mlp:
            self.field_cls = MLP(hidden_size, self.num_fields, dropout=dropout)
        else:
            self.field_cls = nn.Linear(hidden_size, self.num_fields, bias=True)
        # threshold for determing roles
        self.threshold = threshold

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim_event_query)
        self.event_query.data.uniform_(-stdv, stdv)
        self.field_cls.reset_parameters()

    def forward(self, sent_context_emb=None, batch_span_emb=None):
        assert (sent_context_emb is None) ^ (batch_span_emb is None)

        if sent_context_emb is not None:  # [num_spans+num_sents, hidden_size]
            # tzhu: every sentence has an attention score,
            # and document embedding is a attention-based
            # weighted sentence representation
            # doc_emb.size = [1, hidden_size]
            doc_emb, _ = attention(self.event_query, sent_context_emb, sent_context_emb)
            doc_pred_logits = self.event_cls(doc_emb)
            doc_pred_logp = F.log_softmax(doc_pred_logits, dim=-1)

            return doc_pred_logp

        if batch_span_emb is not None:
            # span_context_emb: [batch_size, hidden_size] or [hidden_size]
            if batch_span_emb.dim() == 1:
                batch_span_emb = batch_span_emb.unsqueeze(0)
            # (B, C)
            span_pred_logits = torch.sigmoid(self.field_cls(batch_span_emb))
            return span_pred_logits

    def predict_span_role(self, batch_span_emb, unique_role=True):
        num_ents = batch_span_emb.shape[0]
        if batch_span_emb.dim() == 1:
            batch_span_emb = batch_span_emb.unsqueeze(0)
        # (B, C)
        span_pred_logits = torch.sigmoid(self.field_cls(batch_span_emb))
        span_pred = span_pred_logits.ge(self.threshold).long()
        ent_results = [[] for _ in range(num_ents)]
        if unique_role:
            role_pred_stats = span_pred.sum(0).detach().cpu().tolist()
            pred_results = []
            for role_index, rps in enumerate(role_pred_stats):
                if rps == 0:
                    pred_results.append(None)
                else:
                    pred_results.append(span_pred_logits[:, role_index].argmax())
            # # FIXED: one ent should be able to map multiple roles, other than overlapping by new roles
            # ent_results = [None for _ in range(num_ents)]
            # for role_index, pr in enumerate(pred_results):
            #     if pr is not None:
            #         ent_results[pr] = role_index
            # return ent_results
            for role_index, pr in enumerate(pred_results):
                if pr is not None:
                    ent_results[pr].append(role_index)
        else:
            for span in range(num_ents):
                for role_index, role_result in enumerate(span_pred[span]):
                    if role_result == 1:
                        ent_results[span].append(role_index)
        new_ent_results = []
        for r in ent_results:
            if isinstance(r, list) and len(r) <= 0:
                r = None
            new_ent_results.append(r)
        return new_ent_results

    def extra_repr(self):
        return "event_type={}, num_fields={}, hidden_size={}".format(
            self.event_type, self.num_fields, self.hidden_size
        )
