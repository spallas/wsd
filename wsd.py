import torch
from allennlp.modules.elmo import Elmo
from torch import nn
from torch.nn import CrossEntropyLoss

from models import Attention, ElmoEmbeddings, WSDTransformerEncoder, \
    RobertaEmbeddings, get_transformer_mask, BertEmbeddings
from utils.util import NOT_AMB_SYMBOL


class BaseWSD(nn.Module):

    def __init__(self, device, num_senses: int, max_len: int,
                 batch_size: int = None):
        super().__init__()
        self.device = device
        self.tagset_size = num_senses
        self.win_size = max_len
        self.batch_size = batch_size
        self.ce_loss = CrossEntropyLoss(ignore_index=NOT_AMB_SYMBOL)

    def forward(self, *inputs):
        raise NotImplementedError("Do not use base class, use concrete classes instead.")

    def init_hidden(self, batch_size):
        return None, None

    def loss(self, scores, tags):
        y_true = tags.view(-1)
        scores = scores.view(-1, self.tagset_size)
        return self.ce_loss(scores, y_true)


class BaselineWSD(BaseWSD):

    def __init__(self,
                 num_senses,
                 max_len,
                 elmo_weights,
                 elmo_options,
                 elmo_size,
                 hidden_size,
                 num_layers,
                 batch_size):
        super().__init__(num_senses, max_len, batch_size)
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
        tag_scores = y.view(-1, self.win_size, self.tagset_size)
        return tag_scores


class ElmoTransformerWSD(BaseWSD):

    def __init__(self,
                 device,
                 num_senses,
                 max_len,
                 elmo_weights,
                 elmo_options,
                 elmo_size,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 4):
        super().__init__(num_senses, max_len)
        self.device = device
        self.elmo_weights = elmo_weights
        self.elmo_options = elmo_options
        self.elmo_size = elmo_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.elmo_embedding = ElmoEmbeddings(self.elmo_options,
                                             self.elmo_weights,
                                             self.elmo_size)
        self.transformer = WSDTransformerEncoder(self.elmo_size * 2,
                                                 self.d_model,
                                                 self.tagset_size,
                                                 self.num_layers,
                                                 self.num_heads)

    def forward(self, char_ids, lengths=None):
        x = self.elmo_embedding(char_ids)
        mask = get_transformer_mask(lengths, self.win_size, self.device)
        x = self.transformer(x, mask)
        return x


class RobertaTransformerWSD(BaseWSD):

    def __init__(self,
                 device,
                 num_senses,
                 max_len,
                 model_path,
                 d_embedding: int = 1024,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 4):
        super().__init__(device, num_senses, max_len)
        self.d_embedding = d_embedding
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.embedding = RobertaEmbeddings(device, model_path)
        self.transformer = WSDTransformerEncoder(self.d_embedding, self.d_model,
                                                 self.tagset_size, self.num_layers,
                                                 self.num_heads)

    def forward(self, seq_list, lengths=None):
        x = self.embedding(seq_list)
        mask = get_transformer_mask(lengths, self.win_size, self.device)
        x = self.transformer(x, mask)
        return x


class RobertaDenseWSD(BaseWSD):

    def __init__(self,
                 device,
                 num_senses,
                 max_len,
                 model_path,
                 d_embedding: int = 1024,
                 d_model: int = 512,
                 num_layers: int = 4):
        super().__init__(device, num_senses, max_len)
        pass

    def forward(self, *inputs):
        pass


class BertTransformerWSD(BaseWSD):

    def __init__(self,
                 device,
                 num_senses,
                 max_len,
                 d_model: int = 512,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 # pos_embed_dim: int = 32,
                 bert_model='bert-large-cased'):
        super().__init__(device, num_senses, max_len)
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.bert_model = bert_model
        # self.pos_embed_dim = pos_embed_dim
        # self.pos_embed = nn.Embedding(len(pos2id), self.pos_embed_dim, padding_idx=0)
        self.d_embedding = 768 if 'base' in bert_model else 1024
        self.bert_embedding = BertEmbeddings(device, bert_model)
        self.transformer = WSDTransformerEncoder(self.d_embedding, self.d_model,
                                                 self.tagset_size, self.num_layers,
                                                 self.num_heads)

    def forward(self, seq_list, lengths=None):
        # x_p = self.pos_embed(pos_tags)
        # x = torch.cat([x, x_p], dim=-1)
        x = self.bert_embedding(seq_list, lengths)
        mask = get_transformer_mask(lengths, self.win_size, self.device)
        x = self.transformer(x, mask)
        return x
