import math
from collections import Counter
from typing import List

import torch
from allennlp.modules import Elmo
from allennlp.modules.elmo import batch_to_ids
from fairseq.models.roberta import RobertaModel, alignment_utils
from transformers import BertConfig, BertModel, BertTokenizer
from torch import nn
import torch.nn.functional as F


def str_to_token_ids(batch: List[List[str]], tokenizer):
    slices_b = []
    tokens_b = []
    for seq in batch:
        bert_tok_ids = []
        slices = []
        j = 0
        seq = ['[CLS]'] + seq + ['[SEP]']
        for w in seq:
            bert_tok_ids += tokenizer.encode(w)
            slices.append(slice(j, len(bert_tok_ids)))
            j = len(bert_tok_ids)
        tokens_b.append(torch.tensor(bert_tok_ids))
        slices_b.append(slices)
    tokens_b = nn.utils.rnn.pad_sequence(tokens_b, batch_first=True, padding_value=0)
    return tokens_b, slices_b


def get_transformer_mask(lengths: torch.Tensor, max_len, device):
    if not lengths:
        return None
    # mask is True for values to be masked
    mask_range = torch.arange(max_len) \
        .expand(len(lengths), max_len) \
        .to(device)
    transformer_mask = (mask_range >= lengths.unsqueeze(1))
    return transformer_mask


def align_features_to_words(roberta, features, alignment):
    """
    Align given features to words. Without assert.

    Args:
        roberta (RobertaHubInterface): RoBERTa instance
        features (torch.Tensor): features to align of shape `(T_bpe x C)`
        alignment: alignment between BPE tokens and words returned by
            func:`align_bpe_to_words`.
    """
    assert features.dim() == 2

    bpe_counts = Counter(j for bpe_indices in alignment for j in bpe_indices)
    assert bpe_counts[0] == 0  # <s> shouldn't be aligned
    denom = features.new([bpe_counts.get(j, 1) for j in range(len(features))])
    weighted_features = features / denom.unsqueeze(-1)

    output = [weighted_features[0]]
    largest_j = -1
    for bpe_indices in alignment:
        output.append(weighted_features[bpe_indices].sum(dim=0))
        largest_j = max(largest_j, *bpe_indices)
    for j in range(largest_j + 1, len(features)):
        output.append(weighted_features[j])
    output = torch.stack(output)
    # assert torch.all(torch.abs(output.sum(dim=0) - features.sum(dim=0)) < 1e-4)
    return output


class Attention(nn.Module):
    """
    As described in Raganato et al. https://www.aclweb.org/anthology/D17-1120
    """

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.u_weight = nn.Linear(hidden_size * 2, 1)

    def forward(self, hidden_states):
        """
        :param hidden_states: shape=( B x T x 2*H )
        :return:
        """
        u = self.u_weight(nn.Tanh()(hidden_states))  # shape: B x T x 1
        a = nn.Softmax(dim=1)(u)  # shape: B x T x 1
        c = hidden_states.transpose(1, 2) @ a  # shape: B x 2*H x 1
        c = c.expand(-1, -1, hidden_states.shape[1])  # replicate for each time step, shape: B x 2*H x T
        return torch.cat((hidden_states, c.transpose(1, 2)), dim=-1)


class WSDTransformerEncoder(nn.Module):

    def __init__(self,
                 d_input,
                 d_model,
                 d_output,
                 num_layers,
                 num_heads,
                 small_dim: int = 512):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.d_output = d_output
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.small_dim = small_dim
        self.project_layer = nn.Linear(self.d_input, self.d_model)
        self.layer = nn.TransformerEncoderLayer(self.d_model, self.num_heads)
        self.encoder = nn.TransformerEncoder(self.layer, self.num_layers)
        # self.h_small = nn.Linear(self.d_model, self.small_dim)
        self.output_dense = nn.Linear(self.d_model, self.d_output)
        self.scale = math.sqrt(self.d_input)

    def forward(self, x: torch.Tensor, mask=None):
        """
        """
        seq_len = x.shape[1]
        x = self.project_layer(x)
        x = x * self.scale  # embedding scale
        x = x.transpose(1, 0)  # make batch second dim for transformer layer
        x = self.encoder(x, src_key_padding_mask=mask)
        x = x.transpose(1, 0)  # restore batch first
        # x = self.h_small(x)
        h = x.contiguous().view(-1, x.shape[1], x.shape[2])
        y = self.output_dense(h)
        scores = y.view(-1, seq_len, self.d_output)
        return scores, h


class LSTMEncoder(nn.Module):

    def __init__(self,
                 d_input,
                 d_output,
                 num_layers,
                 hidden_size,
                 batch_size):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.lstm = nn.LSTM(self.d_input,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bidirectional=True,
                            batch_first=True)
        self.attention = Attention(self.hidden_size)
        self.output_dense = nn.Linear(self.hidden_size * 4, self.tagset_size)  # 2 directions * (state + attn)
        self.h = torch.zeros(self.num_layers * 2, 1, self.hidden_size)
        self.cell = torch.zeros(self.num_layers * 2, 1, self.hidden_size)

    def forward(self, x, mask):
        self.h = torch.zeros(self.num_layers * 2, len(x), self.hidden_size)
        self.cell = torch.zeros(self.num_layers * 2, len(x), self.hidden_size)
        hidden_states, (self.h, self.cell) = self.lstm(x, (self.h, self.cell))
        x = self.attention(hidden_states)
        x = x.contiguous().view(-1, x.shape[2])
        x = self.output_dense(x)
        return x


class DenseEncoder(nn.Module):

    def __init__(self,
                 d_input,
                 d_output,
                 hidden_dim: int = 512,
                 small_dim: int = 512):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.hidden_dim = hidden_dim
        self.small_dim = small_dim
        self.project_layer = nn.Linear(self.d_input, self.hidden_dim)
        self.h1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.h2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_dense = nn.Linear(self.hidden_dim, self.d_output)

    def forward(self, x, mask=None):
        x = self.project_layer(x)
        x = self.h1(x)
        # x = F.relu(x)
        x = self.h2(x)
        y = self.output_dense(x)
        return y, x


class BertEmbeddings(nn.Module):

    def __init__(self,
                 device,
                 bert_model):
        super().__init__()
        with torch.no_grad():
            self.bert_config = BertConfig.from_pretrained(bert_model)
            self.bert_embed = BertModel(self.bert_config)
            is_uncased = bert_model.endswith('-uncased')
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=is_uncased)
            self.device = device

    def forward(self, b_x, lengths=None):
        with torch.no_grad():
            token_ids, slices = str_to_token_ids(b_x, self.bert_tokenizer)
            bert_lengths = torch.tensor([sl[-1].stop for sl in slices]).to(self.device)
            max_len = token_ids.shape[1]
            bert_mask = torch.arange(max_len) \
                             .expand(token_ids.shape[0], max_len) \
                             .to(self.device) < bert_lengths.unsqueeze(1)
            x = self.bert_embed(token_ids.to(self.device), attention_mask=bert_mask)[0]
            batch_x = []
            for i in range(x.shape[0]):
                s = x[i]
                m = [torch.mean(s[sl, :], dim=-2) for sl in slices[i]]
                mt = torch.stack(m[1:-1], dim=0)  # remove [CLS] and [SEP] vectors
                batch_x.append(mt)
            x = torch.stack(batch_x, dim=0)
            return x


class BertTrainableEmbeddings(nn.Module):

    def __init__(self,
                 bert_model):
        super().__init__()
        self.bert_config = BertConfig.from_pretrained(bert_model)
        self.bert_model = BertModel(self.bert_config)

    def forward(self, b_x, lengths=None):
        pass


class ElmoEmbeddings(nn.Module):

    def __init__(self,
                 device,
                 elmo_options,
                 elmo_weights,
                 elmo_size=None):
        super().__init__()
        self.device = device
        self.elmo_options = elmo_options
        self.elmo_weights = elmo_weights
        self.elmo_size = elmo_size
        self.elmo = Elmo(self.elmo_options,
                         self.elmo_weights,
                         2, dropout=0)

    def forward(self, b_x, mask=None):
        char_ids = batch_to_ids(b_x)
        char_ids.to(self.device)
        embeddings = self.elmo(char_ids)
        x = embeddings['elmo_representations'][1]
        return x


class RobertaAlignedEmbed(nn.Module):

    def __init__(self,
                 device,
                 model_path='res/roberta.large'):
        super().__init__()
        self.device = device
        with torch.no_grad():
            self.roberta = RobertaModel.from_pretrained(model_path, checkpoint_file='model.pt')
            self.roberta.eval()

    def forward(self, seq_list):
        with torch.no_grad():
            seq_embeddings = []
            for seq in seq_list:
                sent = ' '.join(seq)
                encoded = self.roberta.encode(sent)
                alignment = alignment_utils.align_bpe_to_words(self.roberta, encoded, seq)
                features = self.roberta.extract_features(encoded, return_all_hiddens=False)
                features = features.squeeze(0)
                aligned = align_features_to_words(self.roberta, features, alignment)
                seq_embeddings.append(aligned[1:-1])  # skip <s>,</s> embeddings
            return torch.stack(seq_embeddings, dim=0).to(self.device)
