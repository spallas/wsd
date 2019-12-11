import logging
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

try:
    import torch_sparse
    SPARSE = True
except ImportError:
    SPARSE = False

from models import ElmoEmbeddings, WSDTransformerEncoder, \
    RobertaAlignedEmbed, get_transformer_mask, BertEmbeddings, LSTMEncoder, DenseEncoder
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

    def loss(self, scores, tags, pre_training=False):
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


class RobertaDenseWSD(BaseWSD):

    def __init__(self,
                 device,
                 num_senses,
                 max_len,
                 model_path,
                 d_embedding: int = 1024,
                 hidden_dim: int = 512,
                 cached_embeddings: bool = False):
        super().__init__(device, num_senses, max_len)
        self.d_embedding = d_embedding
        self.hidden_dim = hidden_dim
        self.embedding = RobertaAlignedEmbed(device, model_path) if not cached_embeddings else None
        self.dense = DenseEncoder(self.d_embedding, self.tagset_size, self.hidden_dim)
        self.batch_norm = nn.BatchNorm1d(self.win_size)

    def forward(self, seq_list, lengths=None, cached_embeddings=None):
        x = self.embedding(seq_list) if cached_embeddings is None else cached_embeddings
        x = self.batch_norm(x)
        y, h = self.dense(x)
        return y


class RobertaTransformerWSD(BaseWSD):

    def __init__(self,
                 device,
                 num_senses,
                 max_len,
                 model_path,
                 d_embedding: int = 1024,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 cached_embeddings: bool = False):
        super().__init__(device, num_senses, max_len)
        self.d_embedding = d_embedding
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.embedding = RobertaAlignedEmbed(device, model_path) if not cached_embeddings else None
        self.transformer = WSDTransformerEncoder(self.d_embedding, self.d_model,
                                                 self.tagset_size, self.num_layers,
                                                 self.num_heads)

    def forward(self, seq_list, lengths=None, cached_embeddings=None):
        x = self.embedding(seq_list) if cached_embeddings is None else cached_embeddings
        mask = get_transformer_mask(lengths, self.win_size, self.device)
        x, h = self.transformer(x, mask)
        return x


class WSDNet(RobertaTransformerWSD):

    SLM_SCALE = 0.0001
    FINAL_HIDDEN_SIZE = 512

    def __init__(self,
                 device,
                 num_senses,
                 max_len,
                 model_path,
                 d_embedding: int = 1024,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 output_vocab: str = 'res/dictionaries/syn_lemma_vocab.txt',
                 sense_lemmas: str = 'res/dictionaries/sense_lemmas.txt',
                 cached_embeddings: bool = False):
        super().__init__(device, num_senses, max_len, model_path, d_embedding,
                         d_model, num_heads, num_layers, cached_embeddings)
        self.out_vocab = OrderedDict()
        with open(output_vocab) as f:
            for i, line in enumerate(f):
                self.out_vocab[line.strip()] = i
        self.sense_lemmas = OrderedDict()
        with open(sense_lemmas) as f:
            for line in f:
                sid = int(line.strip().split('\t')[0])
                lemma_list = eval(line.strip().split('\t')[1])
                self.sense_lemmas[sid] = lemma_list
        logging.info('WSDNet dictionaries loaded.')
        self.slm_output_size = len(self.out_vocab)
        # self.reduce_project = nn.Linear(self.transformer.small_dim, self.FINAL_HIDDEN_SIZE)
        self.output_slm = nn.Linear(self.transformer.d_model, len(self.out_vocab))
        self.double_loss = False
        self.v = None

    def forward(self, seq_list, lengths=None, cached_embeddings=None):
        x = self.embedding(seq_list) if cached_embeddings is None else cached_embeddings
        mask = get_transformer_mask(lengths, self.win_size, self.device)
        y, h = self.transformer(x, mask)
        # h = self.reduce_project(h)
        if self.double_loss:
            self.v = self.output_slm(h)
        return y

    def loss(self, scores, tags, opt1=False):
        y_true = tags.view(-1)
        scores = scores.view(-1, self.tagset_size)
        wsd_loss = self.ce_loss(scores, y_true)
        if self.double_loss:
            slm_scores = self.v.view(-1, self.slm_output_size)
            y_slm = torch.zeros_like(slm_scores).to(self.device)
            mask_weights = torch.zeros_like(slm_scores).to(self.device)
            assert y_true.size(0) == y_slm.size(0)
            for y_i, y in enumerate(y_true):
                if y != NOT_AMB_SYMBOL:
                    y_slm[y_i][self.sense_lemmas[y.item()], ] = 1
                    mask_weights[y_i] = 1
                else:
                    mask_weights[y_i] = 0
            slm_loss = F.binary_cross_entropy_with_logits(slm_scores, y_slm, mask_weights, reduction='sum')
            # self.bce_loss(slm_scores, y_slm)
            wsd_loss += slm_loss * self.SLM_SCALE
        return wsd_loss


class WSDNetX(WSDNet):

    SLM_LOGITS_SCALE = 0.1

    def __init__(self,
                 device,
                 num_senses,
                 max_len,
                 model_path,
                 d_embedding: int = 1024,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 output_vocab: str = 'res/dictionaries/syn_lemma_vocab.txt',
                 sense_lemmas: str = 'res/dictionaries/sense_lemmas.txt',
                 cached_embeddings: bool = False):
        super().__init__(device, num_senses, max_len, model_path,
                         d_embedding, d_model, num_heads, num_layers,
                         output_vocab, sense_lemmas, cached_embeddings)
        # build |S| x |V| matrix
        self.double_loss = True
        self.sv_size = torch.Size((len(self.sense_lemmas) + 1, len(self.out_vocab)))
        sparse_coord, values = [], []
        k = 32
        for syn in self.sense_lemmas:
            for j, i in enumerate(self.sense_lemmas[syn]):
                if j > k:
                    break
                sparse_coord.append([syn, i])
                values.append(1 / min(len(self.sense_lemmas[syn]), k))
        keys = torch.LongTensor(sparse_coord)
        vals = torch.FloatTensor(values)
        self.sv_matrix = torch.sparse.FloatTensor(keys.t(), vals, self.sv_size).to(self.device)

    def forward(self, seq_list, lengths=None, cached_embeddings=None):
        x = self.embedding(seq_list) if cached_embeddings is None else cached_embeddings
        mask = get_transformer_mask(lengths, self.win_size, self.device)
        y, h = self.transformer(x, mask)
        # h = self.reduce_project(h)
        self.v = self.output_slm(h)  # shape: |B| x Time steps x |V|
        slm_logits = torch.sparse.mm(self.sv_matrix, self.v.view(-1, self.v.size(-1)).t())   # shape: |S| x T * |B|
        slm_logits = slm_logits.t().view(self.v.size(0), self.v.size(1), -1)
        return y + slm_logits * self.SLM_LOGITS_SCALE


class WSDNetDense(RobertaDenseWSD):

    SLM_SCALE = 0.0001
    SLM_LOGITS_SCALE = 0.1
    FINAL_HIDDEN_SIZE = 64

    def __init__(self,
                 device,
                 num_senses,
                 max_len,
                 model_path,
                 d_embedding: int = 1024,
                 hidden_dim: int = 512,
                 cached_embeddings: bool = False,
                 output_vocab: str = 'res/dictionaries/syn_lemma_vocab.txt',
                 sense_lemmas: str = 'res/dictionaries/sense_lemmas.txt'):
        super().__init__(device, num_senses, max_len, model_path,
                         d_embedding, hidden_dim, cached_embeddings)
        self.out_vocab = OrderedDict()
        with open(output_vocab) as f:
            for i, line in enumerate(f):
                self.out_vocab[line.strip()] = i
        self.sense_lemmas = OrderedDict()
        with open(sense_lemmas) as f:
            for line in f:
                sid = int(line.strip().split('\t')[0])
                lemma_list = eval(line.strip().split('\t')[1])
                self.sense_lemmas[sid] = lemma_list
        logging.info('WSDNetDense: dictionaries loaded.')
        self.slm_output_size = len(self.out_vocab)
        self.output_slm = nn.Linear(self.dense.hidden_dim, len(self.out_vocab)).half()
        self.double_loss = True
        self.v = None
        # build |S| x |V| matrix
        self.sv_size = torch.Size((len(self.sense_lemmas) + 1, len(self.out_vocab)))
        sparse_coord, values = [], []
        k = 32
        for syn in self.sense_lemmas:
            for j, i in enumerate(self.sense_lemmas[syn]):
                if j > k:
                    break
                sparse_coord.append([syn, i])
                values.append(1 / min(len(self.sense_lemmas[syn]), k))
        keys = torch.LongTensor(sparse_coord)
        vals = torch.FloatTensor(values)
        self.sv_matrix = torch.sparse.FloatTensor(keys.t(), vals, self.sv_size).to(self.device)

    def forward(self, seq_list, lengths=None, cached_embeddings=None):
        x = self.embedding(seq_list) if cached_embeddings is None else cached_embeddings
        x = self.batch_norm(x)
        y, h = self.dense(x)
        self.v = self.output_slm(h)  # shape: |B| x Time steps x |V|
        slm_logits = torch.sparse.mm(self.sv_matrix, self.v.view(-1, self.v.size(-1)).t())  # shape: |S| x T * |B|
        slm_logits = slm_logits.t().view(self.v.size(0), self.v.size(1), -1)
        return y + slm_logits * self.SLM_LOGITS_SCALE

    def loss(self, scores, tags, opt1=False):
        y_true = tags.view(-1)
        scores = scores.view(-1, self.tagset_size)
        wsd_loss = self.ce_loss(scores, y_true)
        slm_scores = self.v.view(-1, self.slm_output_size)
        y_slm = torch.zeros_like(slm_scores).to(self.device)
        mask_weights = torch.zeros_like(slm_scores).to(self.device)
        assert y_true.size(0) == y_slm.size(0)
        for y_i, y in enumerate(y_true):
            if y != NOT_AMB_SYMBOL:
                y_slm[y_i][self.sense_lemmas[y.item()], ] = 1
                mask_weights[y_i] = 1
            else:
                mask_weights[y_i] = 0
        slm_loss = F.binary_cross_entropy_with_logits(slm_scores, y_slm, mask_weights, reduction='sum')
        wsd_loss += slm_loss * self.SLM_SCALE
        return wsd_loss


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
        x, h = self.transformer(x, mask)
        return x
