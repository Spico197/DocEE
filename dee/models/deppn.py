import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy.optimize import linear_sum_assignment
from transformers.models.bert.modeling_bert import (
    BertAttention,
    BertIntermediate,
    BertOutput,
)

from dee.modules import (
    AttentiveReducer,
    MentionTypeEncoder,
    SentencePosEncoder,
    transformer,
)
from dee.modules.doc_info import get_deppn_doc_span_info_list
from dee.modules.event_table import EventTable
from dee.modules.ner_model import NERModel


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_weight, matcher, num_event_type):
        super().__init__()
        self.cost_event_type = cost_weight["event_type"]
        self.cost_role = cost_weight["role"]
        self.matcher = matcher
        self.num_event_type = num_event_type

    def forward(self, outputs, targets):
        num_sets, num_roles, num_entities = outputs["pred_role_logits"].size()
        pred_event = outputs["pred_doc_event_logps"].softmax(
            -1
        )  # [num_sets, num_event_types]
        gold_event = targets["doc_event_label"]
        # gold_event_list = [gold_event_type for gold_event_type in gold_event if gold_event_type != self.num_event_type]
        gold_event_tensor = torch.tensor(gold_event).cuda()
        if self.num_event_type == 2:
            gold_event_tensor = torch.zeros(gold_event_tensor.size()).long().cuda()

        pred_role = outputs["pred_role_logits"].softmax(
            -1
        )  # [num_sets,num_roles,num_etities]
        gold_role = targets["role_label"]

        gold_role_lists = [
            role_list for role_list in gold_role if role_list is not None
        ]
        # gold_roles_list = [role for role in gold_role_lists]
        gold_role = torch.tensor(gold_role_lists).cuda()

        pred_role_list = pred_role.split(1, 1)
        gold_role_list = gold_role.split(1, 1)

        if self.matcher == "avg":
            # cost = self.cost_role * pred_role[:, gold_role_tensor]
            cost_list = []
            for pred_role_tensor, gold_role_tensor in zip(
                pred_role_list, gold_role_list
            ):
                pred_role_tensor = pred_role_tensor.squeeze(1)
                cost_list.append(
                    -self.cost_role * pred_role_tensor[:, gold_role_tensor.squeeze(1)]
                )

            event_type_cost = -self.cost_event_type * pred_event[:, gold_event_tensor]
            # cost_list.append(event_type_cost)
            # cost = torch.index_select(pred_role, 1, gold_role)

        role_cost_tensor = torch.stack(cost_list)
        role_cost_tensor = role_cost_tensor.transpose(1, 0)
        role_cost_tensor = role_cost_tensor.view(num_sets, num_roles, -1)
        role_cost = torch.sum(role_cost_tensor, dim=1)
        all_cost = role_cost + event_type_cost
        # all_cost = role_cost

        indices = linear_sum_assignment(all_cost.cpu().detach().numpy())
        # indices_list =  [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        indices_tensor = torch.as_tensor(indices, dtype=torch.int64)
        return indices_tensor


class SetCriterion(nn.Module):
    """This class computes the loss for Set_RE.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class, subject position and object position)
    """

    def __init__(
        self,
        config,
        num_classes,
        event_type_weight=False,
        cost_weight=False,
        na_coef=0.1,
        losses=["event", "role"],
        matcher="avg",
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of relation categories
            matcher: module able to compute a matching between targets and proposals
            loss_weight: dict containing as key the names of the losses and as values their relative weight.
            na_coef: list containg the relative classification weight applied to the NA category and positional classification weight applied to the [SEP]
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(cost_weight, matcher, self.num_classes)
        self.losses = losses
        self.cost_weight = cost_weight
        if event_type_weight:
            self.type_weight = torch.tensor(event_type_weight).cuda()
        else:
            self.type_weight = torch.ones(self.num_classes).cuda()
            self.type_weight[-1] = na_coef
        self.register_buffer("rel_weight", self.type_weight)
        self.cross_entropy = nn.CrossEntropyLoss(reduction="sum")
        self.config = config

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        indices_tensor = self.matcher(outputs, targets)
        losses = self.get_role_loss(outputs, targets, indices_tensor)
        return losses

    def get_role_loss(self, outputs, targets, indices_tensor):
        num_sets, num_roles, num_entities = outputs["pred_role_logits"].size()

        pred_event = outputs["pred_doc_event_logps"].softmax(-1)
        gold_event = targets["doc_event_label"]
        gold_event_tensor = torch.tensor(gold_event).cuda()
        # selected_pred_event_tensor = pred_event[indices_tensor[0][0]]

        pred_role = outputs["pred_role_logits"].softmax(
            -1
        )  # [num_sets,num_roles,num_etities]
        gold_role = targets["role_label"]
        gold_role_tensor = torch.tensor(gold_role).cuda()
        # gold_role_lists = [role_list for role_list in gold_role if role_list != None]
        if self.num_classes == 2:
            gold_event_tensor = torch.zeros(gold_event_tensor.size()).long().cuda()

        selected_pred_role_tensor = pred_role[indices_tensor[0]]
        selected_gold_role_tensor = gold_role_tensor[indices_tensor[1]]
        # role_loss = self.cross_entropy(selected_pred_role_tensor.flatten(0, 1), selected_gold_role_tensor.flatten(0, 1))

        key_role_weight = torch.ones(num_entities).cuda()
        key_role_weight[-1] = 1
        key_role_loss = F.cross_entropy(
            selected_pred_role_tensor[:, :-2].flatten(0, 1),
            selected_gold_role_tensor[:, :-2].flatten(0, 1),
            weight=key_role_weight,
        )

        rest_role_weight = torch.ones(num_entities).cuda()
        rest_role_weight[-1] = 0.2
        rest_role_loss = F.cross_entropy(
            selected_pred_role_tensor[:, -2:].flatten(0, 1),
            selected_gold_role_tensor[:, -2:].flatten(0, 1),
            weight=rest_role_weight,
        )

        role_loss = key_role_loss + rest_role_loss

        gold_event_label = torch.full(
            pred_event.shape[:1], self.num_classes - 1, dtype=torch.int64
        ).cuda()
        gold_event_label[indices_tensor[0]] = gold_event_tensor
        event_type_loss = F.cross_entropy(
            pred_event, gold_event_label, weight=self.type_weight
        )

        if self.config.deppn_train_nopair_sets:
            other_indices_tensor = torch.tensor(
                [i for i in range(num_sets) if i not in indices_tensor[0]],
                dtype=torch.int64,
            )
            other_pred_role_tensor = pred_role[other_indices_tensor]
            other_gold_role_tensor = torch.full(
                other_pred_role_tensor.shape[:2], num_entities - 1, dtype=torch.int64
            ).cuda()
            other_role_loss = F.cross_entropy(
                other_pred_role_tensor.flatten(0, 1),
                other_gold_role_tensor.flatten(0, 1),
            )
            losses = event_type_loss + role_loss + 0.2 * other_role_loss
        else:
            losses = event_type_loss + role_loss

        return losses


class SetPred4DEE(nn.Module):
    def __init__(self, config, event_type2role_index_list, return_intermediate=True):
        super(SetPred4DEE, self).__init__()
        self.config = config
        event_type_classes = config.deppn_event_type_classes
        self.num_generated_sets = config.deppn_num_generated_sets
        num_set_layers = config.deppn_num_set_decoder_layers
        num_role_layers = config.deppn_num_role_decoder_layers
        self.cost_weight = config.deppn_cost_weight
        self.event_cls = nn.Linear(config.hidden_size, event_type_classes)
        self.query_embed = nn.Embedding(self.num_generated_sets, config.hidden_size)
        self.event_type2role_index_list = event_type2role_index_list
        self.role_index_list = [
            role_index
            for role_index_list in event_type2role_index_list
            for role_index in role_index_list
        ]
        self.event_type2role_index_list.append(None)
        self.role_index_num = len(set(self.role_index_list))
        self.role_embed = nn.Embedding(self.role_index_num, config.hidden_size)
        self.role_embed4None = nn.Embedding(1, config.hidden_size)

        if config.deppn_use_event_type_enc:
            self.event_type_embed = nn.Embedding(
                len(event_type2role_index_list), config.hidden_size
            )

        self.return_intermediate = return_intermediate
        self.dropout = nn.Dropout(config.deppn_hidden_dropout)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.deppn_layer_norm_eps
        )
        if num_set_layers > 0:
            self.set_layers = nn.ModuleList(
                [DecoderLayer(config) for _ in range(num_set_layers)]
            )
        else:
            self.set_layers = False

        if num_role_layers > 0:
            self.role_layers = nn.ModuleList(
                [DecoderLayer(config) for _ in range(num_role_layers)]
            )
        else:
            self.role_layers = False

        self.metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.metric_3 = nn.Linear(config.hidden_size, config.hidden_size)
        self.metric_4 = nn.Linear(config.hidden_size, 1, bias=False)
        self.event_type_weight = config.deppn_event_type_weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cuda")
        self.criterion = SetCriterion(
            config, event_type_classes, self.event_type_weight, self.cost_weight
        )
        self.cross_entropy = nn.CrossEntropyLoss(reduction="sum")

        if self.config.deppn_use_sent_span_encoder:
            self.span_sent_encodr = transformer.make_transformer_encoder(
                config.num_tf_layers,
                config.hidden_size,
                ff_size=config.ff_size,
                dropout=config.dropout,
            )

        if self.config.deppn_use_role_decoder:
            self.event2role_decoder = transformer.make_transformer_decoder(
                config.deppn_num_event2role_decoder_layer,
                config.hidden_size,
                ff_size=config.ff_size,
                dropout=config.dropout,
            )

    def forward(
        self,
        doc_sent_context,
        batch_span_context,
        doc_span_info,
        event_type_pred=None,
        train_flag=True,
    ):
        """
        :param doc_sent_context:   num_sents, hidden_size
        :param batch_span_context:   num_candidate_args, hidden_size
        :param doc_span_info:       event_infor of a doc
        :return:
        """
        role4None_embed = self.role_embed4None.weight.unsqueeze(0)
        batch_span_context = batch_span_context.unsqueeze(0)
        batch_span_context = torch.cat((batch_span_context, role4None_embed), 1)
        num_pred_entities = batch_span_context.size()[1]
        doc_sent_context = doc_sent_context.unsqueeze(0)
        if self.config.deppn_use_sent_span_encoder:
            doc_span_sent_context = torch.cat((batch_span_context, doc_sent_context), 1)
            doc_span_sent_context = self.span_sent_encodr(doc_span_sent_context, None)
        else:
            doc_span_sent_context = torch.cat((batch_span_context, doc_sent_context), 1)

        bsz = doc_span_sent_context.size()[0]
        hidden_states = self.query_embed.weight.unsqueeze(0).repeat(bsz, 1, 1)
        hidden_states = self.dropout(self.LayerNorm(hidden_states))

        if self.config.deppn_use_event_type_enc:
            event_type_tensor = torch.tensor(
                event_type_pred, dtype=torch.long, requires_grad=False
            ).to(self.device)
            event_type_embed = self.event_type_embed(event_type_tensor)
            hidden_states = hidden_states + event_type_embed

        all_hidden_states = ()
        if self.set_layers:
            for i, layer_module in enumerate(self.set_layers):
                if self.return_intermediate:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                layer_outputs = layer_module(
                    # hidden_states, batch_span_context
                    hidden_states,
                    doc_sent_context
                    # hidden_states, doc_span_sent_context
                )
                hidden_states = layer_outputs[0]
        else:
            hidden_states = hidden_states

        # event type classification (no-None or None)
        pred_doc_event_logps = self.event_cls(hidden_states).squeeze(0)
        if train_flag:
            event_type_idxs_list = doc_span_info.pred_event_type_idxs_list[
                event_type_pred
            ][: self.num_generated_sets]
            event_arg_idxs_objs_list = doc_span_info.pred_event_arg_idxs_objs_list[
                event_type_pred
            ][: self.num_generated_sets]

        event_index2role_list = [self.event_type2role_index_list[event_type_pred]]
        event_index2role_index_tensor = torch.tensor(
            event_index2role_list, dtype=torch.long, requires_grad=False
        ).to(self.device)
        num_roles = len(event_index2role_list[0])
        event_role_embed = self.role_embed(event_index2role_index_tensor)
        event_role_embed = self.dropout(self.LayerNorm(event_role_embed))
        all_hidden_states = ()

        if self.role_layers:
            for i, layer_module in enumerate(self.role_layers):
                if self.return_intermediate:
                    all_hidden_states = all_hidden_states + (event_role_embed,)
                layer_outputs = layer_module(
                    # event_role_embed, doc_sent_context
                    event_role_embed,
                    batch_span_context
                    # event_role_embed, doc_span_sent_context
                )
                event_role_embed = layer_outputs[0]
        event_role_hidden_states = event_role_embed

        if self.config.deppn_use_role_decoder:
            pred_role_enc = torch.repeat_interleave(
                event_role_hidden_states.unsqueeze(1),
                repeats=self.num_generated_sets,
                dim=1,
            )
            pred_set_role_enc = self.event2role_decoder(
                pred_role_enc.squeeze(0),
                hidden_states.unsqueeze(2).squeeze(0),
                None,
                None,
            )
            pred_set_role_tensor = self.metric_1(
                pred_set_role_enc.unsqueeze(2)
            ) + self.metric_2(doc_span_sent_context).unsqueeze(1)
        else:
            pred_set_tensor = self.metric_1(hidden_states).unsqueeze(2) + self.metric_2(
                doc_span_sent_context
            ).unsqueeze(1)
            pred_set_role_tensor = pred_set_tensor.unsqueeze(2) + self.metric_3(
                event_role_hidden_states
            ).unsqueeze(1).unsqueeze(3)

        pred_role_logits = self.metric_4(torch.tanh(pred_set_role_tensor)).squeeze()
        pred_role_logits = pred_role_logits.view(
            self.num_generated_sets, num_roles, -1
        )  # [num_sets, num_roles, num_entities]
        pred_role_logits = pred_role_logits[:, :, :num_pred_entities]
        outputs = {
            "pred_doc_event_logps": pred_doc_event_logps,
            "pred_role_logits": pred_role_logits,
        }

        if train_flag:
            targets = {
                "doc_event_label": event_type_idxs_list,
                "role_label": event_arg_idxs_objs_list,
            }
            loss = self.criterion(outputs, targets)
            return loss, outputs
        else:
            return outputs


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self, hidden_states, encoder_hidden_states, encoder_attention_mask=None
    ):
        self_attention_outputs = self.attention(hidden_states)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        _ = (encoder_batch_size, encoder_sequence_length)

        cross_attention_outputs = self.crossattention(
            hidden_states=attention_output, encoder_hidden_states=encoder_hidden_states
        )
        attention_output = cross_attention_outputs[0]
        outputs = (
            outputs + cross_attention_outputs[1:]
        )  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class DEPPNModel(nn.Module):
    """Document-level Event Extraction Model"""

    def __init__(self, config, event_type_fields_pairs, ner_model=None):
        super().__init__()
        # Note that for distributed training, you must ensure that
        # for any batch, all parameters need to be used
        logger.warning(
            "This is a simple fork from https://github.com/HangYang-NLP/DE-PPN/"
        )
        logger.warning(
            "Some details may not be implemented to reproduce the results reported in the paper."
        )
        logger.warning(
            "More info could be found at https://github.com/HangYang-NLP/DE-PPN/issues/14"
        )

        self.config = config
        self.event_type_fields_pairs = event_type_fields_pairs
        role2index = {}
        index = 0
        self.event_type2role_index_list = []

        if config.deppn_train_on_multi_roles is True:
            for _, role_type_list, _, _ in self.event_type_fields_pairs:
                event_type2role_index = []
                for role_type in role_type_list:
                    role2index[role_type] = index
                    event_type2role_index.append(index)
                    index += 1
                self.event_type2role_index_list.append(event_type2role_index)
        else:
            for _, role_type_list, *_ in self.event_type_fields_pairs:
                event_type2role_index = []
                for role_type in role_type_list:
                    if role_type not in role2index.keys():
                        role2index[role_type] = index
                        event_type2role_index.append(index)
                        index += 1
                    else:
                        event_type2role_index.append(role2index[role_type])
                self.event_type2role_index_list.append(event_type2role_index)

        if ner_model is None:
            self.ner_model = NERModel(config)
        else:
            self.ner_model = ner_model

        # all event tables
        self.event_tables = nn.ModuleList(
            [
                EventTable(event_type, field_types, config.hidden_size)
                for event_type, field_types, *_ in self.event_type_fields_pairs
            ]
        )

        # sentence position indicator
        self.sent_pos_encoder = SentencePosEncoder(
            config.hidden_size, max_sent_num=config.max_sent_num, dropout=config.dropout
        )

        if self.config.use_token_role:
            self.ment_type_encoder = MentionTypeEncoder(
                config.hidden_size, config.num_entity_labels, dropout=config.dropout
            )

        # various attentive reducer
        if self.config.seq_reduce_type == "AWA":
            self.doc_token_reducer = AttentiveReducer(
                config.hidden_size, dropout=config.dropout
            )
            self.span_token_reducer = AttentiveReducer(
                config.hidden_size, dropout=config.dropout
            )
            self.span_mention_reducer = AttentiveReducer(
                config.hidden_size, dropout=config.dropout
            )
        else:
            assert self.config.seq_reduce_type in {"MaxPooling", "MeanPooling"}

        if self.config.use_doc_enc:
            # get doc-level context information for every mention and sentence
            self.doc_context_encoder = transformer.make_transformer_encoder(
                config.num_tf_layers,
                config.hidden_size,
                ff_size=config.ff_size,
                dropout=config.dropout,
            )

        self.Setpred4DEE = SetPred4DEE(config, self.event_type2role_index_list)

    def get_doc_span_mention_emb(self, doc_token_emb, doc_span_info):
        if len(doc_span_info.mention_drange_list) == 0:
            doc_mention_emb = None
        else:
            # get mention context embeding
            mention_emb_list = []
            for sent_idx, char_s, char_e in doc_span_info.mention_drange_list:
                mention_token_emb = doc_token_emb[
                    sent_idx, char_s:char_e, :
                ]  # [num_mention_tokens, hidden_size]
                if self.config.seq_reduce_type == "AWA":
                    mention_emb = self.span_token_reducer(
                        mention_token_emb
                    )  # [hidden_size]
                elif self.config.seq_reduce_type == "MaxPooling":
                    mention_emb = mention_token_emb.max(dim=0)[0]
                elif self.config.seq_reduce_type == "MeanPooling":
                    mention_emb = mention_token_emb.mean(dim=0)
                else:
                    raise Exception(
                        "Unknown seq_reduce_type {}".format(self.config.seq_reduce_type)
                    )
                mention_emb_list.append(mention_emb)
            doc_mention_emb = torch.stack(mention_emb_list, dim=0)

            # add sentence position embedding
            mention_sent_id_list = [
                drange[0] for drange in doc_span_info.mention_drange_list
            ]
            doc_mention_emb = self.sent_pos_encoder(
                doc_mention_emb, sent_pos_ids=mention_sent_id_list
            )

            if self.config.use_token_role:
                # get mention type embedding
                doc_mention_emb = self.ment_type_encoder(
                    doc_mention_emb, doc_span_info.mention_type_list
                )

        return doc_mention_emb

    def get_batch_sent_emb(self, ner_token_emb, ner_token_masks, valid_sent_num_list):
        # From [ner_batch_size, sent_len, hidden_size] to [ner_batch_size, hidden_size]
        if self.config.seq_reduce_type == "AWA":
            total_sent_emb = self.doc_token_reducer(
                ner_token_emb, masks=ner_token_masks
            )
        elif self.config.seq_reduce_type == "MaxPooling":
            total_sent_emb = ner_token_emb.max(dim=1)[0]
        elif self.config.seq_reduce_type == "MeanPooling":
            total_sent_emb = ner_token_emb.mean(dim=1)
        else:
            raise Exception(
                "Unknown seq_reduce_type {}".format(self.config.seq_reduce_type)
            )

        total_sent_pos_ids = []
        for valid_sent_num in valid_sent_num_list:
            total_sent_pos_ids += list(range(valid_sent_num))
        total_sent_emb = self.sent_pos_encoder(
            total_sent_emb, sent_pos_ids=total_sent_pos_ids
        )

        return total_sent_emb

    def get_doc_span_sent_context(
        self, doc_token_emb, doc_sent_emb, doc_fea, doc_span_info
    ):
        doc_mention_emb = self.get_doc_span_mention_emb(doc_token_emb, doc_span_info)
        # print('num_mentions', doc_mention_emb.size())
        # only consider actual sentences
        if doc_sent_emb.size(0) > doc_fea.valid_sent_num:
            doc_sent_emb = doc_sent_emb[: doc_fea.valid_sent_num, :]
        span_context_list = []

        if doc_mention_emb is None:
            if self.config.use_doc_enc:
                doc_sent_context = self.doc_context_encoder(
                    doc_sent_emb.unsqueeze(0), None
                ).squeeze(0)
            else:
                doc_sent_context = doc_sent_emb
        else:
            num_mentions = doc_mention_emb.size(0)

            if self.config.use_doc_enc:
                # Size([1, num_mentions + num_valid_sents, hidden_size])
                total_ment_sent_emb = torch.cat(
                    [doc_mention_emb, doc_sent_emb], dim=0
                ).unsqueeze(0)

                # size = [num_mentions+num_valid_sents, hidden_size]
                # here we do not need mask
                total_ment_sent_context = self.doc_context_encoder(
                    total_ment_sent_emb, None
                ).squeeze(0)

                # collect span context
                for mid_s, mid_e in doc_span_info.span_mention_range_list:
                    assert mid_e <= num_mentions
                    multi_ment_context = total_ment_sent_context[
                        mid_s:mid_e
                    ]  # [num_mentions, hidden_size]

                    # span_context.size [1, hidden_size]
                    if self.config.seq_reduce_type == "AWA":
                        span_context = self.span_mention_reducer(
                            multi_ment_context, keepdim=True
                        )
                    elif self.config.seq_reduce_type == "MaxPooling":
                        span_context = multi_ment_context.max(dim=0, keepdim=True)[0]
                    elif self.config.seq_reduce_type == "MeanPooling":
                        span_context = multi_ment_context.mean(dim=0, keepdim=True)
                    else:
                        raise Exception(
                            "Unknown seq_reduce_type {}".format(
                                self.config.seq_reduce_type
                            )
                        )

                    span_context_list.append(span_context)

                # collect sent context
                doc_sent_context = total_ment_sent_context[num_mentions:, :]
            else:
                # collect span context
                for mid_s, mid_e in doc_span_info.span_mention_range_list:
                    assert mid_e <= num_mentions
                    multi_ment_emb = doc_mention_emb[
                        mid_s:mid_e
                    ]  # [num_mentions, hidden_size]

                    # span_context.size is [1, hidden_size]
                    if self.config.seq_reduce_type == "AWA":
                        span_context = self.span_mention_reducer(
                            multi_ment_emb, keepdim=True
                        )
                    elif self.config.seq_reduce_type == "MaxPooling":
                        span_context = multi_ment_emb.max(dim=0, keepdim=True)[0]
                    elif self.config.seq_reduce_type == "MeanPooling":
                        span_context = multi_ment_emb.mean(dim=0, keepdim=True)
                    else:
                        raise Exception(
                            "Unknown seq_reduce_type {}".format(
                                self.config.seq_reduce_type
                            )
                        )
                    span_context_list.append(span_context)

                # collect sent context
                doc_sent_context = doc_sent_emb

        return span_context_list, doc_sent_context

    def get_event_cls_info(self, sent_context_emb, doc_fea, train_flag=True):
        doc_event_logps = []
        for event_idx, event_label in enumerate(doc_fea.event_type_labels):
            event_table = self.event_tables[event_idx]
            cur_event_logp = event_table(
                sent_context_emb=sent_context_emb
            )  # [1, hidden_size]
            doc_event_logps.append(cur_event_logp)
        doc_event_logps = torch.cat(doc_event_logps, dim=0)  # [num_event_types, 2]
        if train_flag:
            device = doc_event_logps.device
            doc_event_labels = torch.tensor(
                doc_fea.event_type_labels,
                device=device,
                dtype=torch.long,
                requires_grad=False,
            )  # [num_event_types]
            doc_event_cls_loss = F.nll_loss(
                doc_event_logps, doc_event_labels, reduction="sum"
            )
            doc_event_pred_list = doc_event_logps.argmax(dim=-1).tolist()
            return doc_event_cls_loss, doc_event_pred_list
        else:
            doc_event_pred_list = doc_event_logps.argmax(dim=-1).tolist()
            return doc_event_pred_list

    def get_none_span_context(self, init_tensor):
        none_span_context = torch.zeros(
            1,
            self.config.hidden_size,
            device=init_tensor.device,
            dtype=init_tensor.dtype,
            requires_grad=False,
        )
        return none_span_context

    def get_loss_on_doc(self, doc_token_emb, doc_sent_emb, doc_fea, doc_span_info):
        span_context_list, doc_sent_context = self.get_doc_span_sent_context(
            doc_token_emb,
            doc_sent_emb,
            doc_fea,
            doc_span_info,
        )
        if len(span_context_list) == 0:
            raise Exception(
                "Error: doc_fea.ex_idx {} does not have valid span".format(
                    doc_fea.ex_idx
                )
            )
        event_set_pred_loss = 0
        batch_span_context = torch.cat(span_context_list, dim=0)
        event_cls_loss, doc_event_pred_list = self.get_event_cls_info(
            doc_sent_context, doc_fea, train_flag=True
        )
        event_type_list = doc_fea.event_type_labels
        for event_idx, event_type in enumerate(event_type_list):
            if event_type != 0:
                set_pred_loss, _ = self.Setpred4DEE(
                    doc_sent_context,
                    batch_span_context,
                    doc_span_info,
                    event_idx,
                    train_flag=True,
                )
                event_set_pred_loss += set_pred_loss
        return event_set_pred_loss, event_cls_loss

    def get_mix_loss(self, doc_sent_loss_list, doc_event_loss_list, doc_span_info_list):
        batch_size = len(doc_span_info_list)
        loss_batch_avg = 1.0 / batch_size
        lambda_1 = self.config.deppn_ner_loss_weight
        lambda_2 = self.config.deppn_type_loss_weight
        lambda_3 = self.config.deppn_event_generation_loss_weight

        doc_ner_loss_list = []
        doc_event_type_loss_list = []
        doc_event_generate_loss_list = []
        for doc_sent_loss, doc_span_info in zip(doc_sent_loss_list, doc_span_info_list):
            # doc_sent_loss: Size([num_valid_sents])
            doc_ner_loss_list.append(doc_sent_loss.sum())

        for doc_event_loss in doc_event_loss_list:
            set_pred_loss, event_cls_loss = doc_event_loss
            doc_event_type_loss_list.append(event_cls_loss)
            doc_event_generate_loss_list.append(set_pred_loss)

        return loss_batch_avg * (
            lambda_1 * sum(doc_ner_loss_list)
            + lambda_2 * sum(doc_event_type_loss_list)
            + lambda_3 * sum(doc_event_generate_loss_list)
        )

    def get_eval_on_doc(self, doc_token_emb, doc_sent_emb, doc_fea, doc_span_info):
        span_context_list, doc_sent_context = self.get_doc_span_sent_context(
            doc_token_emb, doc_sent_emb, doc_fea, doc_span_info
        )
        if len(span_context_list) == 0:
            event_pred_list = []
            event_idx2obj_idx2field_idx2token_tup = []
            event_idx2event_decode_paths = []
            for event_idx in range(len(self.event_type_fields_pairs)):
                event_pred_list.append(0)
                event_idx2obj_idx2field_idx2token_tup.append(None)
                event_idx2event_decode_paths.append(None)

            return (
                doc_fea.ex_idx,
                event_pred_list,
                event_idx2obj_idx2field_idx2token_tup,
                doc_span_info,
                event_idx2event_decode_paths,
            )

        batch_span_context = torch.cat(span_context_list, dim=0)
        event_pred_list = self.get_event_cls_info(
            doc_sent_context, doc_fea, train_flag=False
        )
        num_entities = len(span_context_list)

        event_idx2obj_idx2field_idx2token_tup = []
        event_idx2event_decode_paths = []
        for event_idx, event_pred in enumerate(event_pred_list):
            if event_pred == 0:
                event_idx2obj_idx2field_idx2token_tup.append(None)
                event_idx2event_decode_paths.append(None)
            else:
                outputs = self.Setpred4DEE(
                    doc_sent_context,
                    batch_span_context,
                    doc_span_info,
                    event_idx,
                    train_flag=False,
                )
                pred_event = (
                    outputs["pred_doc_event_logps"].softmax(-1).argmax(-1)
                )  # [num_sets,event_types]
                pred_role = (
                    outputs["pred_role_logits"].softmax(-1).argmax(-1)
                )  # [num_sets,num_roles,num_etities]
                obj_idx2field_idx2token_tup = self.pred2standard(
                    pred_event, pred_role, doc_span_info, num_entities
                )
                if len(obj_idx2field_idx2token_tup) < 1:
                    event_idx2obj_idx2field_idx2token_tup.append(None)
                    event_idx2event_decode_paths.append(None)
                else:
                    event_idx2obj_idx2field_idx2token_tup.append(
                        obj_idx2field_idx2token_tup
                    )
                    event_idx2event_decode_paths.append(None)
        return (
            doc_fea.ex_idx,
            event_pred_list,
            event_idx2obj_idx2field_idx2token_tup,
            doc_span_info,
            event_idx2event_decode_paths,
        )

    def pred2standard(
        self, pred_event_list, pred_role_list, doc_span_info, num_entities
    ):
        obj_idx2field_idx2token_tup = []
        for pred_event, pred_role in zip(pred_event_list, pred_role_list):
            if int(pred_event) == 0:
                field_idx2token_tup = []
                for pred_role_index in pred_role:
                    if pred_role_index == num_entities:
                        field_idx2token_tup.append(None)
                    else:
                        field_idx2token_tup.append(
                            doc_span_info.span_token_tup_list[pred_role_index]
                        )
                if field_idx2token_tup not in obj_idx2field_idx2token_tup:
                    obj_idx2field_idx2token_tup.append(field_idx2token_tup)
        return obj_idx2field_idx2token_tup

    def adjust_token_label(self, doc_token_labels_list):
        if self.config.use_token_role:  # do not use detailed token
            return doc_token_labels_list
        else:
            adj_doc_token_labels_list = []
            for doc_token_labels in doc_token_labels_list:
                entity_begin_mask = doc_token_labels % 2 == 1
                entity_inside_mask = (doc_token_labels != 0) & (
                    doc_token_labels % 2 == 0
                )
                adj_doc_token_labels = doc_token_labels.masked_fill(
                    entity_begin_mask, 1
                )
                adj_doc_token_labels = adj_doc_token_labels.masked_fill(
                    entity_inside_mask, 2
                )

                adj_doc_token_labels_list.append(adj_doc_token_labels)
            return adj_doc_token_labels_list

    def get_local_context_info(
        self, doc_batch_dict, train_flag=False, use_gold_span=False
    ):
        label_key = "doc_token_labels"
        if train_flag or use_gold_span:
            assert label_key in doc_batch_dict
            need_label_flag = True
        else:
            need_label_flag = False

        if need_label_flag:
            doc_token_labels_list = self.adjust_token_label(doc_batch_dict[label_key])
        else:
            doc_token_labels_list = None

        batch_size = len(doc_batch_dict["ex_idx"])
        doc_token_ids_list = doc_batch_dict["doc_token_ids"]
        doc_token_masks_list = doc_batch_dict["doc_token_masks"]
        valid_sent_num_list = doc_batch_dict["valid_sent_num"]

        # transform doc_batch into sent_batch
        ner_batch_idx_start_list = [0]
        ner_token_ids = []
        ner_token_masks = []
        ner_token_labels = [] if need_label_flag else None
        for batch_idx, valid_sent_num in enumerate(valid_sent_num_list):
            idx_start = ner_batch_idx_start_list[-1]
            idx_end = idx_start + valid_sent_num
            ner_batch_idx_start_list.append(idx_end)

            ner_token_ids.append(doc_token_ids_list[batch_idx])
            ner_token_masks.append(doc_token_masks_list[batch_idx])
            if need_label_flag:
                ner_token_labels.append(doc_token_labels_list[batch_idx])

        # [batch_size, norm_sent_len]
        ner_token_ids = torch.cat(ner_token_ids, dim=0)
        ner_token_masks = torch.cat(ner_token_masks, dim=0)
        if need_label_flag:
            ner_token_labels = torch.cat(ner_token_labels, dim=0)
        # get ner output
        ner_token_emb, ner_loss, ner_token_preds = self.ner_model(
            ner_token_ids,
            ner_token_masks,
            label_ids=ner_token_labels,
            train_flag=train_flag,
            decode_flag=not use_gold_span,
        )

        if use_gold_span:  # definitely use gold span info
            ner_token_types = ner_token_labels
        else:
            ner_token_types = ner_token_preds
        # get sentence embedding
        ner_sent_emb = self.get_batch_sent_emb(
            ner_token_emb, ner_token_masks, valid_sent_num_list
        )
        assert sum(valid_sent_num_list) == ner_token_emb.size(0) == ner_sent_emb.size(0)

        # followings are all lists of tensors
        doc_token_emb_list = []
        doc_token_masks_list = []
        doc_token_types_list = []
        doc_sent_emb_list = []
        doc_sent_loss_list = []
        for batch_idx in range(batch_size):
            idx_start = ner_batch_idx_start_list[batch_idx]
            idx_end = ner_batch_idx_start_list[batch_idx + 1]
            doc_token_emb_list.append(ner_token_emb[idx_start:idx_end, :, :])
            doc_token_masks_list.append(ner_token_masks[idx_start:idx_end, :])
            doc_token_types_list.append(ner_token_types[idx_start:idx_end, :])
            doc_sent_emb_list.append(ner_sent_emb[idx_start:idx_end, :])
            if ner_loss is not None:
                doc_sent_loss_list.append(ner_loss[idx_start:idx_end])

        return (
            doc_token_emb_list,
            doc_token_masks_list,
            doc_token_types_list,
            doc_sent_emb_list,
            doc_sent_loss_list,
        )

    def forward(
        self,
        doc_batch_dict,
        doc_features,
        train_flag=True,
        use_gold_span=False,
        teacher_prob=1,
        event_idx2entity_idx2field_idx=None,
        heuristic_type=None,
    ):
        # Using scheduled sampling to gradually transit to predicted entity spans
        if train_flag and self.config.use_scheduled_sampling:
            # teacher_prob will gradually decrease outside
            if random.random() < teacher_prob:
                use_gold_span = True
            else:
                use_gold_span = False
        #

        # get doc token-level local context
        (
            doc_token_emb_list,
            doc_token_masks_list,
            doc_token_types_list,
            doc_sent_emb_list,
            doc_sent_loss_list,
        ) = self.get_local_context_info(
            doc_batch_dict,
            train_flag=train_flag,
            use_gold_span=use_gold_span,
        )

        # get doc feature objects
        ex_idx_list = doc_batch_dict["ex_idx"]
        doc_fea_list = [doc_features[ex_idx] for ex_idx in ex_idx_list]

        # get doc span-level info for event extraction
        doc_span_info_list = get_deppn_doc_span_info_list(
            doc_token_types_list, doc_fea_list, use_gold_span=use_gold_span
        )

        if train_flag:
            doc_event_loss_list = []
            for batch_idx, ex_idx in enumerate(ex_idx_list):
                doc_event_loss_list.append(
                    self.get_loss_on_doc(
                        doc_token_emb_list[batch_idx],
                        doc_sent_emb_list[batch_idx],
                        doc_fea_list[batch_idx],
                        doc_span_info_list[batch_idx],
                    )
                )
            mix_loss = self.get_mix_loss(
                doc_sent_loss_list, doc_event_loss_list, doc_span_info_list
            )
            return mix_loss
        else:
            # return a list object may not be supported by torch.nn.parallel.DataParallel
            # ensure to run it under the single-gpu mode
            eval_results = []
            for batch_idx, ex_idx in enumerate(ex_idx_list):
                eval_results.append(
                    self.get_eval_on_doc(
                        doc_token_emb_list[batch_idx],
                        doc_sent_emb_list[batch_idx],
                        doc_fea_list[batch_idx],
                        doc_span_info_list[batch_idx],
                    )
                )
            return eval_results
