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
    RobertaAlignedEmbed, get_transformer_mask, BertEmbeddings, LSTMEncoder, DenseEncoder, \
    label_smoothing_loss
from utils.util import NOT_AMB_SYMBOL


def torch_gmean(t1, t2):
    return torch.exp((torch.log(t1) + torch.log(t2))/2)


class BaseWSD(nn.Module):

    def __init__(self, device, num_senses: int, max_len: int,
                 batch_size: int = None):
        super().__init__()
        self.device = device
        self.tagset_size = num_senses
        self.win_size = max_len
        self.batch_size = batch_size
        # self.batch_norm = nn.BatchNorm1d(self.win_size)

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError("Do not use base class, use concrete classes instead.")

    def loss(self, scores, tags, pre_training=False):
        y_true = tags.view(-1)
        scores = scores.view(-1, self.tagset_size)
        return F.cross_entropy(scores, y_true, ignore_index=NOT_AMB_SYMBOL)


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
        self.project = DenseEncoder(self.d_embedding, self.tagset_size, self.hidden_dim)

    def forward(self, seq_list, lengths=None, cached_embeddings=None, tags=None):
        x = self.embedding(seq_list) if cached_embeddings is None else cached_embeddings
        y, h = self.project(x)
        if tags is None:
            return y
        else:
            return y, self.loss(y, tags.to(y.get_device()))

    def loss(self, scores, tags, pre_training=False):
        y_true = tags.view(-1)
        scores = scores.view(-1, self.tagset_size)
        return label_smoothing_loss(scores, y_true, NOT_AMB_SYMBOL)


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

    def forward(self, seq_list, lengths=None, cached_embeddings=None, tags=None):
        x = self.embedding(seq_list) if cached_embeddings is None else cached_embeddings
        mask = get_transformer_mask(lengths, self.win_size, self.device)
        x, h = self.transformer(x, mask)
        if tags is None:
            return x
        else:
            return x, self.loss(x, tags.to(x.get_device()))

    def loss(self, scores, tags, pre_training=False):
        y_true = tags.view(-1)
        scores = scores.view(-1, self.tagset_size)
        return label_smoothing_loss(scores, y_true, NOT_AMB_SYMBOL)


class WSDNetX(RobertaTransformerWSD):

    SLM_SCALE = 0.01
    FINAL_HIDDEN_SIZE = 512
    SLM_LOGITS_SCALE = 1

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
                 cached_embeddings: bool = False,
                 sv_trainable: bool = False):
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
        self.output_slm = nn.Linear(self.transformer.d_model, len(self.out_vocab))
        self.v = None
        self.sv_trainable = sv_trainable
        # build |S| x |V| matrix
        self.sv_size = torch.Size((len(self.sense_lemmas) + 1, len(self.out_vocab)))
        sparse_coord, values = [], []
        for syn in self.sense_lemmas:
            for j, i in enumerate(self.sense_lemmas[syn]):
                sparse_coord.append([syn, i])
                values.append(1 / len(self.sense_lemmas[syn]))
        self.keys = torch.LongTensor(sparse_coord)
        self.vals = nn.Parameter(torch.FloatTensor(values)) if self.sv_trainable else torch.FloatTensor(values)

    def forward(self, seq_list, lengths=None, cached_embeddings=None, tags=None):
        scores = self._get_scores(seq_list, lengths, cached_embeddings)
        if tags is None:
            return scores
        else:
            return scores, self.loss(scores, tags.to(scores.get_device()))

    def _get_scores(self, seq_list, lengths=None, cached_embeddings=None):
        x = self.embedding(seq_list) if cached_embeddings is None else cached_embeddings
        mask = get_transformer_mask(lengths, self.win_size, self.device)
        y, h = self.transformer(x, mask)
        self.v = self.output_slm(h)  # shape: |B| x Time steps x |V|
        if self.sv_trainable:
            slm_logits = torch_sparse.spmm(self.keys.t().to(self.v.get_device()),
                                           self.vals, self.sv_size[0], self.sv_size[1],
                                           self.v.view(-1, self.v.size(-1)).t())
        else:
            sv_matrix = torch.sparse.FloatTensor(self.keys.t(), self.vals, self.sv_size).to(self.v.get_device())
            slm_logits = torch.sparse.mm(sv_matrix, self.v.t())  # |S| x T * |B|
        slm_logits = slm_logits.t().view(self.v.size(0), self.v.size(1), -1)
        scores = y + slm_logits * self.SLM_LOGITS_SCALE
        return scores

    def loss(self, scores, tags, opt1=False):
        y_true = tags.view(-1)
        scores = scores.view(-1, self.tagset_size)
        wsd_loss = label_smoothing_loss(scores, y_true, ignore_index=NOT_AMB_SYMBOL)
        wsd_loss += self._get_slm_loss(scores.get_device(), y_true) * self.SLM_SCALE
        return wsd_loss

    def _get_slm_loss(self, device, y_true):
        k = 500
        slm_loss = 0
        num_predictions = 0
        for i in range(0, self.v.size(0), k):
            y_slm = torch.zeros_like(self.v[i:i+k]).to(device)
            mask_weights = torch.zeros_like(self.v[i:i+k]).to(device)
            for y_i, y in enumerate(y_true[i:i+k]):
                if y != NOT_AMB_SYMBOL:
                    y_slm[y_i][self.sense_lemmas[y.item()], ] = 1
                    mask_weights[y_i] = 1
                    num_predictions += 1
                else:
                    mask_weights[y_i] = 0
            slm_loss += F.binary_cross_entropy_with_logits(self.v[i:i+k], y_slm,
                                                           mask_weights, reduction='sum')
        return slm_loss / num_predictions


class WSDNetDense(RobertaDenseWSD):

    SLM_SCALE = 0.01
    SLM_LOGITS_SCALE = 1
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
                 sense_lemmas: str = 'res/dictionaries/sense_lemmas.txt',
                 sv_trainable: bool = False):
        assert not sv_trainable or SPARSE
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
        self.project = nn.Linear(self.d_embedding, self.hidden_dim)
        self.h1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.relu1 = nn.ReLU()
        self.h2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.tagset_size)
        self.output_slm = nn.Linear(self.hidden_dim,  len(self.out_vocab))
        self.v = None
        self.wsd_loss = None
        # build |S| x |V| matrix
        self.sv_size = torch.Size((len(self.sense_lemmas) + 1, len(self.out_vocab)))
        self.sv_trainable = sv_trainable
        sparse_coord, values = [], []
        for syn in self.sense_lemmas:
            for j, i in enumerate(self.sense_lemmas[syn]):
                sparse_coord.append([syn, i])
                values.append(1 / len(self.sense_lemmas[syn]))
        logging.info(f"Number of elements in SV matrix: {len(values)}")
        self.keys = torch.LongTensor(sparse_coord)
        self.vals = nn.Parameter(torch.FloatTensor(values)) if self.sv_trainable else torch.FloatTensor(values)

    def forward(self, seq_list, lengths=None, cached_embeddings=None, tags=None):
        scores = self._get_scores(seq_list, cached_embeddings)
        if tags is None:
            return scores
        else:
            return scores, self.loss(scores, tags.to(scores.get_device()))

    def _get_scores(self, seq_list, cached_embeddings=None):
        x = self.embedding(seq_list) if cached_embeddings is None else cached_embeddings
        x = self.project(x)
        x = self.h1(x)
        x = self.relu1(x)
        x = self.h2(x)  # |B| x T x hidden_dim
        h = x.view(-1, x.size(-1))  # |B| * T x hidden_dim
        self.v = self.output_slm(h)  # |B| * T x |V|
        if self.sv_trainable:
            slm_logits = torch_sparse.spmm(self.keys.t().to(self.v.get_device()),
                                           self.vals, self.sv_size[0], self.sv_size[1], self.v.t())
        else:
            sv_matrix = torch.sparse.FloatTensor(self.keys.t(), self.vals, self.sv_size).to(self.v.get_device())
            slm_logits = torch.sparse.mm(sv_matrix, self.v.t())  # |S| x T * |B|
        slm_logits = slm_logits.t()  # |B| * T x |S|
        y = self.output_layer(h)
        scores = y + slm_logits * self.SLM_LOGITS_SCALE
        scores = scores.view(x.size(0), x.size(1), -1)
        return scores

    def loss(self, scores, tags, opt1=False):
        y_true = tags.view(-1)
        scores = scores.view(-1, self.tagset_size)
        wsd_loss = label_smoothing_loss(scores, y_true, ignore_index=NOT_AMB_SYMBOL)
        slm_loss = self._get_slm_loss(scores.get_device(), y_true)
        loss = wsd_loss + slm_loss * self.SLM_SCALE
        return loss

    def _get_slm_loss(self, device, y_true):
        k = 500
        slm_loss = 0
        for i in range(0, self.v.size(0), k):
            y_slm = torch.zeros_like(self.v[i:i+k]).to(device)
            mask_weights = torch.zeros_like(self.v[i:i+k]).to(device)
            num_predictions = 0
            for y_i, y in enumerate(y_true[i:i+k]):
                if y != NOT_AMB_SYMBOL:
                    y_slm[y_i][self.sense_lemmas[y.item()], ] = 1
                    mask_weights[y_i] = 1
                    num_predictions += 1
                else:
                    mask_weights[y_i] = 0
            slm_loss += F.binary_cross_entropy_with_logits(self.v[i:i+k], y_slm,
                                                           mask_weights, reduction='sum') / num_predictions
        return slm_loss


class WSDNetDenseAdasoft(RobertaDenseWSD):

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
        self.dense = nn.Linear(self.d_embedding, self.hidden_dim)
        self.h1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.relu1 = nn.ReLU()
        self.h2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_layer = nn.AdaptiveLogSoftmaxWithLoss(self.hidden_dim, self.tagset_size, [1000, 3000, 10_000])
        self.output_slm = nn.AdaptiveLogSoftmaxWithLoss(self.hidden_dim,  len(self.out_vocab), [1000, 3000, 10_000])
        # nn.Linear(self.dense.hidden_dim, len(self.out_vocab))
        self.v = None
        self.wsd_loss = None
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
        x = self.dense(x)
        x = self.h1(x)
        x = self.relu1(x)
        x = self.h2(x)  # |B| x T x hidden_dim
        h = x.view(-1, x.size(-1))  # |B| * T x hidden_dim
        self.v = self.output_slm.log_prob(h)  # |B| * T x |V|
        slm_logits = torch.sparse.mm(self.sv_matrix, self.v.t())  # |S| x T * |B|
        slm_logits = slm_logits.t()  # |B| * T x |S|
        y = self.output_layer.log_prob(h)  # |B| * T x |S|
        return y + slm_logits * self.SLM_LOGITS_SCALE

    def loss(self, scores, tags, opt1=False):
        y_true = tags.view(-1)
        scores = scores.view(-1, self.tagset_size)
        wsd_loss = F.nll_loss(scores, y_true, ignore_index=NOT_AMB_SYMBOL)
        slm_loss = self._get_slm_loss(y_true)
        loss = torch_gmean(wsd_loss, slm_loss)  # slm_loss * self.SLM_SCALE
        return loss

    def _get_slm_loss(self, y_true):
        k = 10
        slm_loss = 0
        for i in range(0, self.v.size(0), k):
            y_slm = torch.zeros_like(self.v[i:i+k]).to(self.device)
            mask_weights = torch.zeros_like(self.v[i:i+k]).to(self.device)
            for y_i, y in enumerate(y_true[i:i+k]):
                if y != NOT_AMB_SYMBOL:
                    y_slm[y_i][self.sense_lemmas[y.item()], ] = 1
                    mask_weights[y_i] = 1
                else:
                    mask_weights[y_i] = 0
            slm_loss += F.binary_cross_entropy_with_logits(self.v[i:i+k], y_slm, mask_weights, reduction='sum')
        return slm_loss


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
