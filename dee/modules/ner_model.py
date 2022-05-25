import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mcrf.modules import MaskedCRF, allowed_transitions
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel

from dee.helper.dee import DEEExample
from dee.modules.biaffine import Biaffine

from . import transformer


class BertForBasicNER(BertPreTrainedModel):
    """BERT model for basic NER functionality.
    This module is composed of the BERT model with a linear layer on top of
    the output sequences.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_entity_labels`: the number of entity classes for the classifier.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary.
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `label_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with label indices selected in [0, ..., num_labels-1].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits sequence.
    """

    def __init__(self, config, num_entity_labels):
        super(BertForBasicNER, self).__init__(config)
        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, num_entity_labels)
        self.init_weights()

        self.num_entity_labels = num_entity_labels

    def forward(
        self,
        input_ids,
        input_masks,
        label_ids=None,
        train_flag=True,
        decode_flag=True,
        return_sent_rep=False,
    ):
        """Assume input size [batch_size, seq_len]"""
        if input_masks.dtype != torch.uint8:
            input_masks = input_masks == 1

        batch_seq_enc, batch_cls_output = self.bert(
            input_ids, attention_mask=input_masks
        )
        # [batch_size, seq_len, hidden_size]
        batch_seq_enc = self.dropout(batch_seq_enc)
        # [batch_size, seq_len, num_entity_labels]
        batch_seq_logits = self.classifier(batch_seq_enc)

        batch_seq_logp = F.log_softmax(batch_seq_logits, dim=-1)

        if train_flag:
            batch_logp = batch_seq_logp.view(-1, batch_seq_logp.size(-1))
            batch_label = label_ids.view(-1)
            # ner_loss = F.nll_loss(batch_logp, batch_label, reduction='sum')
            ner_loss = F.nll_loss(batch_logp, batch_label, reduction="none")
            ner_loss = ner_loss.view(label_ids.size()).sum(dim=-1)  # [batch_size]
        else:
            ner_loss = None

        if decode_flag:
            batch_seq_preds = batch_seq_logp.argmax(dim=-1)
        else:
            batch_seq_preds = None

        if return_sent_rep:
            return batch_seq_enc, ner_loss, batch_seq_preds, batch_cls_output
        return batch_seq_enc, ner_loss, batch_seq_preds


class BERTCRFNERModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.bert = BertModel.from_pretrained(config.bert_model)

        if self.config.use_crf_layer:
            self.crf_layer = CRFLayer(config.hidden_size, self.config.num_entity_labels)
        else:
            # Token Label Classification
            self.classifier = nn.Linear(
                config.hidden_size, self.config.num_entity_labels
            )

    def forward(
        self, input_ids, input_masks, label_ids=None, train_flag=True, decode_flag=True
    ):
        """Assume input size [batch_size, seq_len]"""
        if input_masks.dtype != torch.uint8:
            input_masks = input_masks == 1
        if train_flag:
            assert label_ids is not None

        # batch_seq_enc, sent_reps = self.bert(input_ids, attention_mask=input_masks)
        outputs = self.bert(input_ids, attention_mask=input_masks)
        batch_seq_enc = outputs.last_hidden_state
        sent_reps = outputs.pooler_output

        if self.config.use_crf_layer:
            ner_loss, batch_seq_preds = self.crf_layer(
                batch_seq_enc,
                seq_token_label=label_ids,
                batch_first=True,
                train_flag=train_flag,
                decode_flag=decode_flag,
            )
        else:
            # [batch_size, seq_len, num_entity_labels]
            batch_seq_logits = self.classifier(batch_seq_enc)
            batch_seq_logp = F.log_softmax(batch_seq_logits, dim=-1)

            if train_flag:
                batch_logp = batch_seq_logp.view(-1, batch_seq_logp.size(-1))
                batch_label = label_ids.view(-1)
                # ner_loss = F.nll_loss(batch_logp, batch_label, reduction='sum')
                ner_loss = F.nll_loss(batch_logp, batch_label, reduction="none")
                ner_loss = ner_loss.view(label_ids.size()).sum(dim=-1)  # [batch_size]
            else:
                ner_loss = None

            if decode_flag:
                batch_seq_preds = batch_seq_logp.argmax(dim=-1)
            else:
                batch_seq_preds = None

        return batch_seq_enc, ner_loss, batch_seq_preds, sent_reps


class NERModel(nn.Module):
    def __init__(self, config):
        super(NERModel, self).__init__()

        self.config = config
        # Word Embedding, Word Local Position Embedding
        self.token_embedding = NERTokenEmbedding(
            config.vocab_size,
            config.hidden_size,
            max_sent_len=config.max_sent_len,
            dropout=config.dropout,
        )
        # Multi-layer Transformer Layers to Incorporate Contextual Information
        # self.token_encoder = transformer.make_transformer_encoder(
        #     config.num_tf_layers, config.hidden_size, ff_size=config.ff_size, dropout=config.dropout
        # )
        self.token_encoder = transformer.make_transformer_encoder(
            config.num_ner_tf_layers,
            config.hidden_size,
            ff_size=config.ff_size,
            dropout=config.dropout,
        )
        if self.config.use_crf_layer:
            self.crf_layer = CRFLayer(config.hidden_size, self.config.num_entity_labels)
        else:
            # Token Label Classification
            self.classifier = nn.Linear(
                config.hidden_size, self.config.num_entity_labels
            )

    def forward(
        self, input_ids, input_masks, label_ids=None, train_flag=True, decode_flag=True
    ):
        """Assume input size [batch_size, seq_len]"""
        if input_masks.dtype != torch.uint8:
            input_masks = input_masks == 1
        if train_flag:
            assert label_ids is not None

        # get contextual info
        input_emb = self.token_embedding(input_ids)
        input_masks = input_masks.unsqueeze(-2)  # to fit for the transformer code
        batch_seq_enc = self.token_encoder(input_emb, input_masks)

        if self.config.use_crf_layer:
            ner_loss, batch_seq_preds = self.crf_layer(
                batch_seq_enc,
                seq_token_label=label_ids,
                batch_first=True,
                train_flag=train_flag,
                decode_flag=decode_flag,
            )
        else:
            # [batch_size, seq_len, num_entity_labels]
            batch_seq_logits = self.classifier(batch_seq_enc)
            batch_seq_logp = F.log_softmax(batch_seq_logits, dim=-1)

            if train_flag:
                batch_logp = batch_seq_logp.view(-1, batch_seq_logp.size(-1))
                batch_label = label_ids.view(-1)
                # ner_loss = F.nll_loss(batch_logp, batch_label, reduction='sum')
                ner_loss = F.nll_loss(batch_logp, batch_label, reduction="none")
                ner_loss = ner_loss.view(label_ids.size()).sum(dim=-1)  # [batch_size]
            else:
                ner_loss = None

            if decode_flag:
                batch_seq_preds = batch_seq_logp.argmax(dim=-1)
            else:
                batch_seq_preds = None

        return batch_seq_enc, ner_loss, batch_seq_preds


class LSTMCRFNERModel(nn.Module):
    def __init__(self, config):
        super(LSTMCRFNERModel, self).__init__()

        self.config = config
        # Word Embedding, Word Local Position Embedding
        self.token_embedding = CommonNERTokenEmbedding(
            config.vocab_size,
            config.hidden_size,
            max_sent_len=config.max_sent_len,
            dropout=config.dropout,
        )
        self.token_encoder = nn.LSTM(
            config.hidden_size,
            config.hidden_size // 2,
            num_layers=config.num_lstm_layers,
            bias=True,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=True,
        )
        if self.config.use_crf_layer:
            self.crf_layer = CRFLayer(config.hidden_size, self.config.num_entity_labels)
        else:
            # Token Label Classification
            self.classifier = nn.Linear(
                config.hidden_size, self.config.num_entity_labels
            )

    def lstm_encode(self, input_states, mask):
        lens = mask.sum(dim=1)
        total_length = mask.shape[1]
        batch_size, seq_len, hidden_size = input_states.shape

        x = pack_padded_sequence(
            input_states, lens.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        output, (h_n, _) = self.token_encoder(
            x
        )  # h_n: #(direction, #sent, hidden_size)
        output, _ = pad_packed_sequence(
            output, batch_first=True, padding_value=0.0, total_length=total_length
        )
        h_n = h_n.view(self.config.num_lstm_layers, 2, batch_size, hidden_size // 2)[-1]
        sent_reps = h_n.transpose(0, 1).reshape(mask.shape[0], -1)
        return output, sent_reps  # output: B*L*2H

    def forward(
        self, input_ids, input_masks, label_ids=None, train_flag=True, decode_flag=True
    ):
        """Assume input size [batch_size, seq_len]"""
        if input_masks.dtype != torch.uint8:
            input_masks = input_masks == 1
        if train_flag:
            assert label_ids is not None

        # get contextual info
        input_emb = self.token_embedding(input_ids)
        batch_seq_enc, sent_reps = self.lstm_encode(input_emb, input_masks)

        if self.config.use_crf_layer:
            ner_loss, batch_seq_preds = self.crf_layer(
                batch_seq_enc,
                seq_token_label=label_ids,
                batch_first=True,
                train_flag=train_flag,
                decode_flag=decode_flag,
            )
        else:
            # [batch_size, seq_len, num_entity_labels]
            batch_seq_logits = self.classifier(batch_seq_enc)
            batch_seq_logp = F.log_softmax(batch_seq_logits, dim=-1)

            if train_flag:
                batch_logp = batch_seq_logp.view(-1, batch_seq_logp.size(-1))
                batch_label = label_ids.view(-1)
                # ner_loss = F.nll_loss(batch_logp, batch_label, reduction='sum')
                ner_loss = F.nll_loss(batch_logp, batch_label, reduction="none")
                ner_loss = ner_loss.view(label_ids.size()).sum(dim=-1)  # [batch_size]
            else:
                ner_loss = None

            if decode_flag:
                batch_seq_preds = batch_seq_logp.argmax(dim=-1)
            else:
                batch_seq_preds = None

        return batch_seq_enc, ner_loss, batch_seq_preds, sent_reps


class LSTMMaskedCRFNERModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        # Word Embedding, Word Local Position Embedding
        self.token_embedding = CommonNERTokenEmbedding(
            config.vocab_size,
            config.hidden_size,
            max_sent_len=config.max_sent_len,
            dropout=config.dropout,
        )
        self.token_encoder = nn.LSTM(
            config.hidden_size,
            config.hidden_size // 2,
            num_layers=config.num_lstm_layers,
            bias=True,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=True,
        )
        if self.config.use_crf_layer:
            self.crf_layer = MaskedCRF(
                self.config.num_entity_labels,
                allowed_transitions("BIO", self.config.tag_id2tag_name),
            )
        self.hidden2tag = nn.Linear(config.hidden_size, self.config.num_entity_labels)

    def lstm_encode(self, input_states, mask):
        lens = mask.sum(dim=1)
        total_length = mask.shape[1]
        batch_size, seq_len, hidden_size = input_states.shape

        x = pack_padded_sequence(
            input_states, lens, batch_first=True, enforce_sorted=False
        )
        output, (h_n, _) = self.token_encoder(
            x
        )  # h_n: #(direction, #sent, hidden_size)
        output, _ = pad_packed_sequence(
            output, batch_first=True, padding_value=0.0, total_length=total_length
        )
        h_n = h_n.view(self.config.num_lstm_layers, 2, batch_size, hidden_size // 2)[-1]
        sent_reps = h_n.transpose(0, 1).reshape(mask.shape[0], -1)
        return output, sent_reps  # output: B*L*2H

    def forward(
        self, input_ids, input_masks, label_ids=None, train_flag=True, decode_flag=True
    ):
        """Assume input size [batch_size, seq_len]"""
        if input_masks.dtype != torch.uint8:
            input_masks = input_masks == 1
        if train_flag:
            assert label_ids is not None

        # get contextual info
        input_emb = self.token_embedding(input_ids)
        batch_seq_enc, sent_reps = self.lstm_encode(input_emb, input_masks)

        batch_seq_logits = self.hidden2tag(batch_seq_enc)
        if self.config.use_crf_layer:
            if train_flag:
                ner_loss = -self.crf_layer(
                    batch_seq_logits, label_ids, input_masks.bool(), reduction="none"
                )
            else:
                ner_loss = None
            batch_seq_preds = self.crf_layer.decode(batch_seq_logits)
            batch_seq_preds = torch.tensor(
                batch_seq_preds, dtype=torch.long, device=batch_seq_logits.device
            )
        else:
            # [batch_size, seq_len, num_entity_labels]
            batch_seq_logp = F.log_softmax(batch_seq_logits, dim=-1)

            if train_flag:
                batch_logp = batch_seq_logp.view(-1, batch_seq_logp.size(-1))
                batch_label = label_ids.view(-1)
                # ner_loss = F.nll_loss(batch_logp, batch_label, reduction='sum')
                ner_loss = F.nll_loss(batch_logp, batch_label, reduction="none")
                ner_loss = ner_loss.view(label_ids.size()).sum(dim=-1)  # [batch_size]
            else:
                ner_loss = None

            if decode_flag:
                batch_seq_preds = batch_seq_logp.argmax(dim=-1)
            else:
                batch_seq_preds = None

        return batch_seq_enc, ner_loss, batch_seq_preds, sent_reps


class LSTMCRFAttNERModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        # Word Embedding, Word Local Position Embedding
        self.token_embedding = CommonNERTokenEmbedding(
            config.vocab_size,
            config.hidden_size,
            max_sent_len=config.max_sent_len,
            dropout=config.dropout,
        )
        self.token_encoder = nn.LSTM(
            config.hidden_size,
            config.hidden_size // 2,
            num_layers=config.num_lstm_layers,
            bias=True,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=True,
        )
        self.self_att = transformer.SelfAttention(self.config.hidden_size)
        self.att_cls = nn.Linear(config.hidden_size, self.config.num_entity_labels)
        if self.config.use_crf_layer:
            self.crf_layer = CRFLayer(config.hidden_size, self.config.num_entity_labels)
        else:
            # Token Label Classification
            self.classifier = nn.Linear(
                config.hidden_size, self.config.num_entity_labels
            )

    def lstm_encode(self, input_states, mask):
        lens = mask.sum(dim=1)
        total_length = mask.shape[1]
        batch_size, seq_len, hidden_size = input_states.shape

        x = pack_padded_sequence(
            input_states, lens, batch_first=True, enforce_sorted=False
        )
        output, (h_n, _) = self.token_encoder(
            x
        )  # h_n: #(direction, #sent, hidden_size)
        output, _ = pad_packed_sequence(
            output, batch_first=True, padding_value=0.0, total_length=total_length
        )
        h_n = h_n.view(self.config.num_lstm_layers, 2, batch_size, hidden_size // 2)[-1]
        sent_reps = h_n.transpose(0, 1).reshape(mask.shape[0], -1)
        return output, sent_reps  # output: B*L*2H

    def forward(
        self, input_ids, input_masks, label_ids=None, train_flag=True, decode_flag=True
    ):
        """Assume input size [batch_size, seq_len]"""
        if input_masks.dtype != torch.uint8:
            input_masks = input_masks == 1
        if train_flag:
            assert label_ids is not None

        # get contextual info
        input_emb = self.token_embedding(input_ids)

        att_emb, _ = self.self_att(input_emb, mask=input_masks)
        att_logits = self.att_cls(att_emb)

        batch_seq_enc, sent_reps = self.lstm_encode(input_emb, input_masks)

        if self.config.use_crf_layer:
            ner_loss, batch_seq_preds = self.crf_layer(
                batch_seq_enc,
                seq_token_label=label_ids,
                batch_first=True,
                train_flag=train_flag,
                decode_flag=decode_flag,
            )
        else:
            # [batch_size, seq_len, num_entity_labels]
            batch_seq_logits = self.classifier(batch_seq_enc)
            batch_seq_logp = F.log_softmax(batch_seq_logits, dim=-1)

            if train_flag:
                batch_logp = batch_seq_logp.view(-1, batch_seq_logp.size(-1))
                batch_label = label_ids.view(-1)
                # ner_loss = F.nll_loss(batch_logp, batch_label, reduction='sum')
                ner_loss = F.nll_loss(batch_logp, batch_label, reduction="none")
                ner_loss = ner_loss.view(label_ids.size()).sum(dim=-1)  # [batch_size]
            else:
                ner_loss = None

            if decode_flag:
                batch_seq_preds = batch_seq_logp.argmax(dim=-1)
            else:
                batch_seq_preds = None

        if train_flag:
            att_loss = F.cross_entropy(
                att_logits.reshape(-1, att_logits.size(-1)), label_ids.reshape(-1)
            )
            if ner_loss is not None:
                ner_loss += att_loss

        return batch_seq_enc, ner_loss, batch_seq_preds, sent_reps, att_emb


class LSTMCRFSentAttNERModel(nn.Module):
    r"""
    attended sentence representation
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        # Word Embedding, Word Local Position Embedding
        self.token_embedding = CommonNERTokenEmbedding(
            config.vocab_size,
            config.hidden_size,
            max_sent_len=config.max_sent_len,
            dropout=config.dropout,
        )
        self.token_encoder = nn.LSTM(
            config.hidden_size,
            config.hidden_size // 2,
            num_layers=config.num_lstm_layers,
            bias=True,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=True,
        )
        self.self_att = transformer.SelfAttention(self.config.hidden_size)
        self.att_cls = nn.Linear(config.hidden_size, self.config.num_entity_labels)
        if self.config.use_crf_layer:
            self.crf_layer = CRFLayer(config.hidden_size, self.config.num_entity_labels)
        else:
            # Token Label Classification
            self.classifier = nn.Linear(
                config.hidden_size, self.config.num_entity_labels
            )

    def lstm_encode(self, input_states, mask):
        lens = mask.sum(dim=1)
        total_length = mask.shape[1]
        batch_size, seq_len, hidden_size = input_states.shape

        x = pack_padded_sequence(
            input_states, lens, batch_first=True, enforce_sorted=False
        )
        output, (h_n, _) = self.token_encoder(
            x
        )  # h_n: #(direction, #sent, hidden_size)
        output, _ = pad_packed_sequence(
            output, batch_first=True, padding_value=0.0, total_length=total_length
        )
        h_n = h_n.view(self.config.num_lstm_layers, 2, batch_size, hidden_size // 2)[-1]
        sent_reps = h_n.transpose(0, 1).reshape(mask.shape[0], -1)
        return output, sent_reps  # output: B*L*2H

    def forward(
        self, input_ids, input_masks, label_ids=None, train_flag=True, decode_flag=True
    ):
        """Assume input size [batch_size, seq_len]"""
        if input_masks.dtype != torch.uint8:
            input_masks = input_masks == 1
        if train_flag:
            assert label_ids is not None

        # get contextual info
        input_emb = self.token_embedding(input_ids)

        _, p_attn = self.self_att(input_emb, mask=input_masks)
        # (batch size, 1, seq_len)
        p_attn = p_attn.max(dim=-1)[1].unsqueeze(1)
        sent_reps = torch.matmul(p_attn, input_emb).squeeze(1)

        batch_seq_enc, _ = self.lstm_encode(input_emb, input_masks)

        if self.config.use_crf_layer:
            ner_loss, batch_seq_preds = self.crf_layer(
                batch_seq_enc,
                seq_token_label=label_ids,
                batch_first=True,
                train_flag=train_flag,
                decode_flag=decode_flag,
            )
        else:
            # [batch_size, seq_len, num_entity_labels]
            batch_seq_logits = self.classifier(batch_seq_enc)
            batch_seq_logp = F.log_softmax(batch_seq_logits, dim=-1)

            if train_flag:
                batch_logp = batch_seq_logp.view(-1, batch_seq_logp.size(-1))
                batch_label = label_ids.view(-1)
                # ner_loss = F.nll_loss(batch_logp, batch_label, reduction='sum')
                ner_loss = F.nll_loss(batch_logp, batch_label, reduction="none")
                ner_loss = ner_loss.view(label_ids.size()).sum(dim=-1)  # [batch_size]
            else:
                ner_loss = None

            if decode_flag:
                batch_seq_preds = batch_seq_logp.argmax(dim=-1)
            else:
                batch_seq_preds = None

        return batch_seq_enc, ner_loss, batch_seq_preds, sent_reps


class StackLSTMNERModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        # Word Embedding, Word Local Position Embedding
        self.token_embedding = CommonNERTokenEmbedding(
            config.vocab_size,
            config.hidden_size,
            max_sent_len=config.max_sent_len,
            dropout=config.dropout,
        )
        self.token_encoder = nn.LSTM(
            config.hidden_size,
            config.hidden_size // 2,
            num_layers=config.num_lstm_layers,
            bias=True,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.sent_encoder = nn.LSTM(
            config.hidden_size,
            config.hidden_size // 2,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )
        if self.config.use_crf_layer:
            self.crf_layer = CRFLayer(config.hidden_size, self.config.num_entity_labels)
        else:
            # Token Label Classification
            self.classifier = nn.Linear(
                config.hidden_size, self.config.num_entity_labels
            )

    def lstm_encode(self, input_states, mask):
        lens = mask.sum(dim=1)
        total_length = mask.shape[1]
        batch_size, seq_len, hidden_size = input_states.shape

        x = pack_padded_sequence(
            input_states, lens, batch_first=True, enforce_sorted=False
        )
        output, (h_n, _) = self.token_encoder(
            x
        )  # h_n: #(direction, #sent, hidden_size)
        ner_output, _ = pad_packed_sequence(
            output, batch_first=True, padding_value=0.0, total_length=total_length
        )
        sent_input = self.dropout(ner_output)
        sent_input = pack_padded_sequence(
            sent_input, lens, batch_first=True, enforce_sorted=False
        )
        sent_output, _ = self.sent_encoder(sent_input)
        sent_output, _ = pad_packed_sequence(
            sent_output, batch_first=True, padding_value=0.0, total_length=total_length
        )
        sent_output = sent_output.max(dim=1)[0]
        return ner_output, sent_output  # output: B*L*2H

    def forward(
        self, input_ids, input_masks, label_ids=None, train_flag=True, decode_flag=True
    ):
        """Assume input size [batch_size, seq_len]"""
        if input_masks.dtype != torch.uint8:
            input_masks = input_masks == 1
        if train_flag:
            assert label_ids is not None

        # get contextual info
        input_emb = self.token_embedding(input_ids)
        batch_seq_enc, sent_reps = self.lstm_encode(input_emb, input_masks)

        if self.config.use_crf_layer:
            ner_loss, batch_seq_preds = self.crf_layer(
                batch_seq_enc,
                seq_token_label=label_ids,
                batch_first=True,
                train_flag=train_flag,
                decode_flag=decode_flag,
            )
        else:
            # [batch_size, seq_len, num_entity_labels]
            batch_seq_logits = self.classifier(batch_seq_enc)
            batch_seq_logp = F.log_softmax(batch_seq_logits, dim=-1)

            if train_flag:
                batch_logp = batch_seq_logp.view(-1, batch_seq_logp.size(-1))
                batch_label = label_ids.view(-1)
                # ner_loss = F.nll_loss(batch_logp, batch_label, reduction='sum')
                ner_loss = F.nll_loss(batch_logp, batch_label, reduction="none")
                ner_loss = ner_loss.view(label_ids.size()).sum(dim=-1)  # [batch_size]
            else:
                ner_loss = None

            if decode_flag:
                batch_seq_preds = batch_seq_logp.argmax(dim=-1)
            else:
                batch_seq_preds = None

        return batch_seq_enc, ner_loss, batch_seq_preds, sent_reps


class LSTMBiaffineNERModel(nn.Module):
    r"""
    Biaffine based NER model
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        # Word Embedding, Word Local Position Embedding
        self.token_embedding = CommonNERTokenEmbedding(
            config.vocab_size,
            config.hidden_size,
            max_sent_len=config.max_sent_len,
            dropout=config.dropout,
        )
        self.token_encoder = nn.LSTM(
            config.hidden_size,
            config.hidden_size // 2,
            num_layers=config.num_lstm_layers,
            bias=True,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=True,
        )
        self.start_mlp = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.biaffine_hidden_size,
            bias=True,
        )
        self.end_mlp = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.biaffine_hidden_size,
            bias=True,
        )

        self.tril_matrix = torch.ones(
            size=[config.max_sent_len, config.max_sent_len],
            requires_grad=False,
            device=self.start_mlp.weight.device,
            dtype=torch.bool,
        ).tril()

        self.biaffine = Biaffine(config.biaffine_hidden_size, config.num_entity_labels)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")
        # with B-, I- and O
        self.index2entity_label = dict()
        self.entity_label2index = dict()
        # without B-, I- and O
        self.ent2idx = dict()
        self.idx2ent = dict()
        for idx, entity_label in enumerate(DEEExample.get_entity_label_list()):
            self.index2entity_label[idx] = entity_label
            self.entity_label2index[entity_label] = idx
            if entity_label == "O":
                self.ent2idx[entity_label] = len(self.ent2idx)
            else:
                self.ent2idx[entity_label[2:]] = len(self.ent2idx)

    def lstm_encode(self, input_states, mask):
        lens = mask.sum(dim=1)
        total_length = mask.shape[1]
        batch_size, seq_len, hidden_size = input_states.shape

        x = pack_padded_sequence(
            input_states, lens, batch_first=True, enforce_sorted=False
        )
        output, (h_n, _) = self.token_encoder(
            x
        )  # h_n: #(direction, #sent, hidden_size)
        output, _ = pad_packed_sequence(
            output, batch_first=True, padding_value=0.0, total_length=total_length
        )
        h_n = h_n.view(self.config.num_lstm_layers, 2, batch_size, hidden_size // 2)[-1]
        sent_reps = h_n.transpose(0, 1).reshape(mask.shape[0], -1)
        return output, sent_reps  # output: B*L*2H

    def get_range_type(self, label_ids):
        if isinstance(label_ids, torch.Tensor):
            label_ids = label_ids.cpu().detach().tolist()
        ranges = []
        for idx, label_id in enumerate(label_ids):
            ent_label = self.index2entity_label[label_id]
            if ent_label.startswith("B-"):
                ranges.append([idx, idx + 1, ent_label[2:]])
            elif ent_label.startswith("I-") and ent_label[2:] == ranges[-1][2]:
                ranges[-1][1] += 1
        return ranges

    def convert_label_ids2label_mat(self, label_ids, mask=None):
        seq_len = label_ids.shape[0]
        target = -1 * torch.ones(
            seq_len,
            seq_len,
            requires_grad=False,
            device=label_ids.device,
            dtype=torch.long,
        )
        target = target.tril()
        ranges = self.get_range_type(label_ids)
        for start_idx, end_idx, ent_type in ranges:
            target[start_idx][end_idx - 1] = self.ent2idx[ent_type]
        if mask is not None:
            sent_len = mask.sum(-1)
            target[:, sent_len:] = -1
        return target

    def decode(self, biaffine_logitses, seq_lens):
        pred_idses = []
        for biaffine_logits, seq_len in zip(biaffine_logitses, seq_lens):
            scores, pred = biaffine_logits.max(dim=-1)
            pred = (
                pred.cpu()
                .detach()
                .numpy()
                .reshape(-1, self.config.max_sent_len, self.config.max_sent_len)
                .tolist()
            )
            scores = (
                scores.cpu()
                .detach()
                .numpy()
                .reshape(-1, self.config.max_sent_len, self.config.max_sent_len)
                .tolist()
            )

            top_spans = []
            for i in range(seq_len):
                for j in range(seq_len):
                    if pred[i][j] != self.ent2idx["O"]:
                        top_spans.append((i, j, pred[i][j], scores[i][j]))
            top_spans = sorted(top_spans, key=lambda x: x[3], reverse=True)
            sent_pred = []
            for ns, ne, t, _ in top_spans:
                for ts, te, _ in sent_pred:
                    if ns < ts <= ne < te or ts < ns <= te < ne:
                        # for both nested and flat ner no clash is allowed
                        break
                    if ns <= ts <= te <= ne or ts <= ns <= ne <= te:
                        # for flat ner nested mentions are not allowed
                        break
                # Tip: The else clause executes after the loop completes normally
                else:
                    sent_pred.append((ns, ne, t))
            pred_ids = [self.entity_label2index["O"]] * self.config.max_sent_len
            for ns, ne, t in sent_pred:
                t = self.idx2ent[t]
                pred_ids[ns] = self.entity_label2index[f"B-{t}"]
                for i in range(ns + 1, ne + 1):
                    pred_ids[i] = self.entity_label2index[f"I-{t}"]
            pred_ids = torch.tensor(
                pred_ids, dtype=torch.long, device=seq_lens.device, requires_grad=False
            )
            pred_idses.append(pred_ids)
        pred_results = torch.cat(pred_idses, dim=0)
        return pred_results

    def forward(
        self, input_ids, input_masks, label_ids=None, train_flag=True, decode_flag=True
    ):
        """Assume input size [batch_size, seq_len]"""
        if input_masks.dtype != torch.uint8:
            input_masks = input_masks == 1
        if train_flag:
            assert label_ids is not None

        # get contextual info
        input_emb = self.token_embedding(input_ids)
        batch_seq_enc, sent_reps = self.lstm_encode(input_emb, input_masks)
        hidden_start = self.start_mlp(batch_seq_enc)
        hidden_end = self.end_mlp(batch_seq_enc)
        biaffine_logits = self.biaffine(hidden_start, hidden_end)

        if train_flag:
            ner_losses = []
            for i in range(input_ids.shape[0]):
                target = self.convert_label_ids2label_mat(label_ids[i], input_masks[i])
                loss = self.criterion(
                    biaffine_logits[i]
                    .permute(1, 2, 0)
                    .reshape(-1, self.config.num_entity_labels),
                    target.reshape(-1),
                )
                ner_losses.append(loss)
            ner_loss = torch.stack(ner_losses)
        else:
            ner_loss = None

        if decode_flag:
            return self.decode(biaffine_logits.permute(0, 2, 3, 1), input_masks.sum(-1))
        else:
            batch_seq_preds = None

        return batch_seq_enc, ner_loss, batch_seq_preds, sent_reps


class NERTokenEmbedding(nn.Module):
    """Add token position information"""

    def __init__(self, vocab_size, hidden_size, max_sent_len=256, dropout=0.1):
        super(NERTokenEmbedding, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_sent_len, hidden_size)

        self.layer_norm = transformer.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_token_ids):
        batch_size, sent_len = batch_token_ids.size()
        device = batch_token_ids.device

        batch_pos_ids = torch.arange(
            sent_len, dtype=torch.long, device=device, requires_grad=False
        )
        batch_pos_ids = batch_pos_ids.unsqueeze(0).expand_as(batch_token_ids)

        batch_token_emb = self.token_embedding(batch_token_ids)
        batch_pos_emb = self.pos_embedding(batch_pos_ids)

        batch_token_emb = batch_token_emb + batch_pos_emb

        batch_token_out = self.layer_norm(batch_token_emb)
        batch_token_out = self.dropout(batch_token_out)

        return batch_token_out


class CommonNERTokenEmbedding(nn.Module):
    """Add token position information"""

    def __init__(self, vocab_size, hidden_size, max_sent_len=256, dropout=0.1):
        super(CommonNERTokenEmbedding, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_token_ids):
        batch_token_emb = self.token_embedding(batch_token_ids)
        batch_token_out = self.dropout(batch_token_emb)

        return batch_token_out


class CRFLayer(nn.Module):
    NEG_LOGIT = -100000.0
    """
    Conditional Random Field Layer
    Reference:
        https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#sphx-glr-beginner-nlp-advanced-tutorial-py
    The original example codes operate on one sequence, while this version operates on one batch
    """

    def __init__(self, hidden_size, num_entity_labels):
        super(CRFLayer, self).__init__()

        self.tag_size = num_entity_labels + 2  # add start tag and end tag
        self.start_tag = self.tag_size - 2
        self.end_tag = self.tag_size - 1

        # Map token-level hidden state into tag scores
        self.hidden2tag = nn.Linear(hidden_size, self.tag_size)
        # Transition Matrix
        # [i, j] denotes transitioning from j to i
        self.trans_mat = nn.Parameter(torch.randn(self.tag_size, self.tag_size))
        self.reset_trans_mat()

    def reset_trans_mat(self):
        nn.init.kaiming_uniform_(
            self.trans_mat, a=math.sqrt(5)
        )  # copy from Linear init
        # set parameters that will not be updated during training, but is important
        self.trans_mat.data[self.start_tag, :] = self.NEG_LOGIT
        self.trans_mat.data[:, self.end_tag] = self.NEG_LOGIT

    def get_log_parition(self, seq_emit_score):
        """
        Calculate the log of the partition function
        :param seq_emit_score: [seq_len, batch_size, tag_size]
        :return: Tensor with Size([batch_size])
        """
        seq_len, batch_size, tag_size = seq_emit_score.size()
        # dynamic programming table to store previously summarized tag logits
        dp_table = seq_emit_score.new_full(
            (batch_size, tag_size), self.NEG_LOGIT, requires_grad=False
        )
        dp_table[:, self.start_tag] = 0.0

        batch_trans_mat = self.trans_mat.unsqueeze(0).expand(
            batch_size, tag_size, tag_size
        )

        for token_idx in range(seq_len):
            prev_logit = dp_table.unsqueeze(1)  # [batch_size, 1, tag_size]
            batch_emit_score = seq_emit_score[token_idx].unsqueeze(
                -1
            )  # [batch_size, tag_size, 1]
            cur_logit = (
                batch_trans_mat + batch_emit_score + prev_logit
            )  # [batch_size, tag_size, tag_size]
            dp_table = log_sum_exp(cur_logit)  # [batch_size, tag_size]
        batch_logit = dp_table + self.trans_mat[self.end_tag, :].unsqueeze(0)
        log_partition = log_sum_exp(batch_logit)  # [batch_size]

        return log_partition

    def get_gold_score(self, seq_emit_score, seq_token_label):
        """
        Calculate the score of the given sequence label
        :param seq_emit_score: [seq_len, batch_size, tag_size]
        :param seq_token_label: [seq_len, batch_size]
        :return: Tensor with Size([batch_size])
        """
        seq_len, batch_size, tag_size = seq_emit_score.size()

        end_token_label = seq_token_label.new_full(
            (1, batch_size), self.end_tag, requires_grad=False
        )
        seq_cur_label = (
            torch.cat([seq_token_label, end_token_label], dim=0)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(seq_len + 1, batch_size, 1, tag_size)
        )

        start_token_label = seq_token_label.new_full(
            (1, batch_size), self.start_tag, requires_grad=False
        )
        seq_prev_label = (
            torch.cat([start_token_label, seq_token_label], dim=0)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )  # [seq_len+1, batch_size, 1, 1]

        seq_trans_score = (
            self.trans_mat.unsqueeze(0)
            .unsqueeze(0)
            .expand(seq_len + 1, batch_size, tag_size, tag_size)
        )
        # gather according to token label at the current token
        gold_trans_score = torch.gather(
            seq_trans_score, 2, seq_cur_label
        )  # [seq_len+1, batch_size, 1, tag_size]
        # gather according to token label at the previous token
        gold_trans_score = torch.gather(
            gold_trans_score, 3, seq_prev_label
        )  # [seq_len+1, batch_size, 1, 1]
        batch_trans_score = (
            gold_trans_score.sum(dim=0).squeeze(-1).squeeze(-1)
        )  # [batch_size]

        gold_emit_score = torch.gather(
            seq_emit_score, 2, seq_token_label.unsqueeze(-1)
        )  # [seq_len, batch_size, 1]
        batch_emit_score = gold_emit_score.sum(dim=0).squeeze(-1)  # [batch_size]

        gold_score = batch_trans_score + batch_emit_score  # [batch_size]

        return gold_score

    def viterbi_decode(self, seq_emit_score):
        """
        Use viterbi decoding to get prediction
        :param seq_emit_score: [seq_len, batch_size, tag_size]
        :return:
            batch_best_path: [batch_size, seq_len], the best tag for each token
            batch_best_score: [batch_size], the corresponding score for each path
        """
        seq_len, batch_size, tag_size = seq_emit_score.size()

        dp_table = seq_emit_score.new_full(
            (batch_size, tag_size), self.NEG_LOGIT, requires_grad=False
        )
        dp_table[:, self.start_tag] = 0
        backpointers = []

        for token_idx in range(seq_len):
            last_tag_score = dp_table.unsqueeze(-2)  # [batch_size, 1, tag_size]
            batch_trans_mat = self.trans_mat.unsqueeze(0).expand(
                batch_size, tag_size, tag_size
            )
            cur_emit_score = seq_emit_score[token_idx].unsqueeze(
                -1
            )  # [batch_size, tag_size, 1]
            cur_trans_score = (
                batch_trans_mat + last_tag_score + cur_emit_score
            )  # [batch_size, tag_size, tag_size]
            dp_table, cur_tag_bp = cur_trans_score.max(dim=-1)  # [batch_size, tag_size]
            backpointers.append(cur_tag_bp)
        # transition to the end tag
        last_trans_arr = (
            self.trans_mat[self.end_tag].unsqueeze(0).expand(batch_size, tag_size)
        )
        dp_table = dp_table + last_trans_arr

        # get the best path score and the best tag of the last token
        batch_best_score, best_tag = dp_table.max(dim=-1)  # [batch_size]
        best_tag = best_tag.unsqueeze(-1)  # [batch_size, 1]
        best_tag_list = [best_tag]
        # reversely traverse back pointers to recover the best path
        for last_tag_bp in reversed(backpointers):
            # best_tag Size([batch_size, 1]) records the current tag that can own the highest score
            # last_tag_bp Size([batch_size, tag_size]) records the last best tag that the current tag is based on
            best_tag = torch.gather(last_tag_bp, 1, best_tag)  # [batch_size, 1]
            best_tag_list.append(best_tag)
        batch_start = best_tag_list.pop()
        assert (batch_start == self.start_tag).sum().item() == batch_size
        best_tag_list.reverse()
        batch_best_path = torch.cat(best_tag_list, dim=-1)  # [batch_size, seq_len]

        return batch_best_path, batch_best_score

    def forward(
        self,
        seq_token_emb,
        seq_token_label=None,
        batch_first=False,
        train_flag=True,
        decode_flag=True,
    ):
        """
        Get loss and prediction with CRF support.
        :param seq_token_emb: assume size [seq_len, batch_size, hidden_size] if not batch_first
        :param seq_token_label: assume size [seq_len, batch_size] if not batch_first
        :param batch_first: Flag to denote the meaning of the first dimension
        :param train_flag: whether to calculate the loss
        :param decode_flag: whether to decode the path based on current parameters
        :return:
            nll_loss: negative log-likelihood loss
            seq_token_pred: seqeunce predictions
        """
        if batch_first:
            # CRF assumes the input size of [seq_len, batch_size, hidden_size]
            seq_token_emb = seq_token_emb.transpose(0, 1).contiguous()
            if seq_token_label is not None:
                seq_token_label = seq_token_label.transpose(0, 1).contiguous()

        seq_emit_score = self.hidden2tag(
            seq_token_emb
        )  # [seq_len, batch_size, tag_size]
        if train_flag:
            gold_score = self.get_gold_score(
                seq_emit_score, seq_token_label
            )  # [batch_size]
            log_partition = self.get_log_parition(seq_emit_score)  # [batch_size]
            nll_loss = log_partition - gold_score
        else:
            nll_loss = None

        if decode_flag:
            # Use viterbi decoding to get the current prediction
            # no matter what batch_first is, return size is [batch_size, seq_len]
            batch_best_path, batch_best_score = self.viterbi_decode(seq_emit_score)
        else:
            batch_best_path = None

        return nll_loss, batch_best_path


# Compute log sum exp in a numerically stable way
def log_sum_exp(batch_logit):
    """
    Caculate the log-sum-exp operation for the last dimension.
    :param batch_logit: Size([*, logit_size]), * should at least be 1
    :return: Size([*])
    """
    batch_max, _ = batch_logit.max(dim=-1)
    batch_broadcast = batch_max.unsqueeze(-1)
    return batch_max + torch.log(
        torch.sum(torch.exp(batch_logit - batch_broadcast), dim=-1)
    )


def produce_ner_batch_metrics(seq_logits, gold_labels, masks):
    # seq_logits: [batch_size, seq_len, num_entity_labels]
    # gold_labels: [batch_size, seq_len]
    # masks: [batch_size, seq_len]
    batch_size, seq_len, num_entities = seq_logits.size()

    # [batch_size, seq_len, num_entity_labels]
    seq_logp = F.log_softmax(seq_logits, dim=-1)
    # [batch_size, seq_len]
    pred_labels = seq_logp.argmax(dim=-1)
    # [batch_size*seq_len, num_entity_labels]
    token_logp = seq_logp.view(-1, num_entities)
    # [batch_size*seq_len]
    token_labels = gold_labels.view(-1)
    # [batch_size, seq_len]
    seq_token_loss = F.nll_loss(token_logp, token_labels, reduction="none").view(
        batch_size, seq_len
    )

    batch_metrics = []
    for bid in range(batch_size):
        ex_loss = seq_token_loss[bid, masks[bid]].mean().item()
        ex_acc = (
            (pred_labels[bid, masks[bid]] == gold_labels[bid, masks[bid]])
            .float()
            .mean()
            .item()
        )
        ex_pred_lids = pred_labels[bid, masks[bid]].tolist()
        ex_gold_lids = gold_labels[bid, masks[bid]].tolist()
        ner_tp_set, ner_fp_set, ner_fn_set = judge_ner_prediction(
            ex_pred_lids, ex_gold_lids
        )
        batch_metrics.append(
            [ex_loss, ex_acc, len(ner_tp_set), len(ner_fp_set), len(ner_fn_set)]
        )

    return torch.tensor(batch_metrics, dtype=torch.float, device=seq_logits.device)


def judge_ner_prediction(pred_label_ids, gold_label_ids):
    """Very strong assumption on label_id, 0: others, odd: ner_start, even: ner_mid"""
    if isinstance(pred_label_ids, torch.Tensor):
        pred_label_ids = pred_label_ids.tolist()
    if isinstance(gold_label_ids, torch.Tensor):
        gold_label_ids = gold_label_ids.tolist()
    # element: (ner_start_index, ner_end_index, ner_type_id)
    pred_ner_set = set()
    gold_ner_set = set()

    pred_ner_sid = None
    for idx, ner in enumerate(pred_label_ids):
        if pred_ner_sid is None:
            if ner % 2 == 1:
                pred_ner_sid = idx
                continue
        else:
            prev_ner = pred_label_ids[pred_ner_sid]
            if ner == 0:
                pred_ner_set.add((pred_ner_sid, idx, prev_ner))
                pred_ner_sid = None
                continue
            elif ner == prev_ner + 1:  # same entity
                continue
            elif ner % 2 == 1:
                pred_ner_set.add((pred_ner_sid, idx, prev_ner))
                pred_ner_sid = idx
                continue
            else:  # ignore invalid subsequence ners
                pred_ner_set.add((pred_ner_sid, idx, prev_ner))
                pred_ner_sid = None
                pass
    if pred_ner_sid is not None:
        prev_ner = pred_label_ids[pred_ner_sid]
        pred_ner_set.add((pred_ner_sid, len(pred_label_ids), prev_ner))

    gold_ner_sid = None
    for idx, ner in enumerate(gold_label_ids):
        if gold_ner_sid is None:
            if ner % 2 == 1:
                gold_ner_sid = idx
                continue
        else:
            prev_ner = gold_label_ids[gold_ner_sid]
            if ner == 0:
                gold_ner_set.add((gold_ner_sid, idx, prev_ner))
                gold_ner_sid = None
                continue
            elif ner == prev_ner + 1:  # same entity
                continue
            elif ner % 2 == 1:
                gold_ner_set.add((gold_ner_sid, idx, prev_ner))
                gold_ner_sid = idx
                continue
            else:  # ignore invalid subsequence ners
                gold_ner_set.add((gold_ner_sid, idx, prev_ner))
                gold_ner_sid = None
                pass
    if gold_ner_sid is not None:
        prev_ner = gold_label_ids[gold_ner_sid]
        gold_ner_set.add((gold_ner_sid, len(gold_label_ids), prev_ner))

    ner_tp_set = pred_ner_set.intersection(gold_ner_set)
    ner_fp_set = pred_ner_set - gold_ner_set
    ner_fn_set = gold_ner_set - pred_ner_set

    return ner_tp_set, ner_fp_set, ner_fn_set
