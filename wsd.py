from torch import nn

from models import ElmoEmbeddings, WSDTransformerEncoder, \
    RobertaEmbeddings, get_transformer_mask, BertEmbeddings, LSTMEncoder
from utils.util import NOT_AMB_SYMBOL


class BaseWSD(nn.Module):

    def __init__(self, device, num_senses: int, max_len: int,
                 batch_size: int = None):
        super().__init__()
        self.device = device
        self.tagset_size = num_senses
        self.win_size = max_len
        self.batch_size = batch_size
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=NOT_AMB_SYMBOL)

    def forward(self, *inputs):
        raise NotImplementedError("Do not use base class, use concrete classes instead.")

    def loss(self, scores, tags):
        y_true = tags.view(-1)
        scores = scores.view(-1, self.tagset_size)
        return self.ce_loss(scores, y_true)


class BaselineWSD(BaseWSD):

    def __init__(self,
                 device,
                 num_senses,
                 max_len,
                 elmo_weights,
                 elmo_options,
                 elmo_size,
                 hidden_size,
                 num_layers):
        super().__init__(device, num_senses, max_len)
        self.elmo_weights = elmo_weights
        self.elmo_options = elmo_options
        self.elmo_size = elmo_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.elmo_embedding = ElmoEmbeddings(device, elmo_options,
                                             elmo_weights, elmo_size)
        self.embedding_size = 2 * self.elmo_size
        self.lstm_encoder = LSTMEncoder(self.embedding_size, self.tagset_size,
                                        self.num_layers, self.hidden_size, self.batch_size)

    def forward(self, seq_list, lengths=None):
        x = self.elmo_embedding(seq_list)
        x = self.lstm_encoder(x)
        return x


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
        super().__init__(device, num_senses, max_len)
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

    def forward(self, seq_list, lengths=None):
        x = self.elmo_embedding(seq_list)
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


class RobertaTransformerLM(RobertaTransformerWSD):

    def __init__(self,
                 device,
                 num_senses,
                 max_len,
                 model_path,
                 d_embedding: int = 1024,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 4):
        super().__init__(device, num_senses, max_len, model_path,
                         d_embedding, d_model, num_heads, num_layers)
        lm = None

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
