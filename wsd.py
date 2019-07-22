import torch
from allennlp.modules.elmo import Elmo
from pytorch_transformers import BertModel, BertConfig
from torch import nn
from torch.nn import CrossEntropyLoss

from models import Attention, WSDTransformerEncoder
from utils.config import TransformerConfig


class BaselineWSD(nn.Module):

    def __init__(self, num_senses: int, max_len: int):
        super().__init__()
        self.tagset_size = num_senses
        self.win_size = max_len
        self.pad_tag_index = 0
        self.ce_loss = CrossEntropyLoss(ignore_index=0)

    def forward(self, *inputs):
        pass

    def _loss(self, scores, tags, device):
        y_true = torch.tensor(tags).view(-1).to(device)
        scores = scores.view(-1, self.tagset_size)
        mask = (y_true != self.pad_tag_index).float()
        num_tokens = int(torch.sum(mask).item())

        y_l = scores[range(scores.shape[0]), y_true] * mask
        ce_loss = - torch.sum(y_l) / (num_tokens + 1)
        # Negative LogLikelihood
        return ce_loss

    def loss(self, scores, tags, device):
        y_true = torch.tensor(tags).view(-1).to(device)
        scores = scores.view(-1, self.tagset_size)
        return self.ce_loss(scores, y_true)


class SimpleWSD(BaselineWSD):

    _ELMO_OPTIONS = 'res/elmo/elmo_2x1024_128_2048cnn_1xhighway_options.json'
    _ELMO_WEIGHTS = 'res/elmo/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'
    _ELMO_SIZE = 128
    _HIDDEN_SIZE = 1024
    _NUM_LAYERS = 2
    _BATCH_SIZE = 32

    def __init__(self,
                 num_senses,
                 max_len,
                 elmo_weights=_ELMO_WEIGHTS,
                 elmo_options=_ELMO_OPTIONS,
                 elmo_size=_ELMO_SIZE,
                 hidden_size=_HIDDEN_SIZE,
                 num_layers=_NUM_LAYERS,
                 batch_size=_BATCH_SIZE):
        super().__init__(num_senses, max_len)
        self.elmo_weights = elmo_weights
        self.elmo_options = elmo_options
        self.elmo_size = elmo_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.elmo = Elmo(self.elmo_options,
                         self.elmo_weights,
                         2, dropout=0)
        self.embedding_size = 2 * self.elmo_size
        self.lstm = nn.LSTM(self.embedding_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bidirectional=True,
                            batch_first=True)
        self.attention = Attention(self.hidden_size)
        self.output_dense = nn.Linear(self.hidden_size * 4, self.tagset_size)  # 2 directions * (state + attn)
        self.batch_size = batch_size
        self.h, self.cell = self.init_hidden(self.batch_size)

    def init_hidden(self, batch_size):
        self.batch_size = batch_size
        return (torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size),  # hidden state
                torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size))  # cell state

    def forward(self, char_ids, lengths):

        embeddings = self.elmo(char_ids)
        x = embeddings['elmo_representations'][1]
        hidden_states, (self.h, self.cell) = self.lstm(x, (self.h, self.cell))
        out = hidden_states

        y = self.attention(out)
        y = y.contiguous().view(-1, y.shape[2])
        y = self.output_dense(y)
        y = nn.LogSoftmax(dim=1)(y)
        tag_scores = y.view(self.batch_size, self.win_size, self.tagset_size)
        return tag_scores


class ElmoTransformerWSD(nn.Module):

    _ELMO_OPTIONS = ''
    _ELMO_WEIGHTS = ''

    def __init__(self):
        super().__init__()
        pass

    def forward(self, *inputs):
        pass


class BertTransformerWSD(BaselineWSD):

    def __init__(self, device, num_senses, max_len, config: TransformerConfig):
        super().__init__(num_senses, max_len)
        self.bert_config = BertConfig.from_pretrained('bert-base-uncased')
        self.bert_embedding = BertModel(self.bert_config)
        self.config = config
        self.transformer_layer = WSDTransformerEncoder(self.config)
        self.output_dense = nn.Linear(self.config.encoder_embed_dim, self.tagset_size)
        self.device = device

    def _aggregate_bert(self, x, starts):
        # how to do efficiently?
        pass

    def forward(self, token_ids, lengths, slices):
        """

        :param slices: List[Slice]
        :param lengths: List[int] shape = `(batch)`
        :param token_ids: (Tensor) shape `(batch, seq_len)`
        :return:
        """
        max_len = token_ids.shape[1]  # self.bert_config.max_position_embeddings
        attention_mask = torch.arange(max_len)\
                              .expand(len(lengths), max_len)\
                              .to(self.device) < lengths.unsqueeze(1)
        x, _ = self.bert_embedding(token_ids, attention_mask=attention_mask)
        x = x.transpose(1, 0)  # make batch second dim for fairseq transformer.
        for _ in range(self.config.num_layers):
            x = self.transformer_layer(x, 1 - attention_mask)

        x = x.transpose(1, 0)  # restore batch first

        batch = []
        for i in range(x.shape[0]):  # different slices across batch
            b = torch.cat([torch.mean(x[i, sl, :], dim=-2) for sl in slices[i]])\
                     .reshape(-1, self.config.encoder_embed_dim)
            batch.append(b)
        x = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
        x = self.output_dense(x)

        return x

    def loss(self, scores, tags, device=None):
        y_true = tags.view(-1)
        scores = scores.view(-1, self.tagset_size)
        return self.ce_loss(scores, y_true)


class WSDNet(nn.Module):
    """
    Multi-Task network for WSD
    """

    def __init__(self):
        super().__init__()
        pass

    def forward(self, *inputs):
        pass
