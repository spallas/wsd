import math

import torch
from allennlp.modules import Elmo
from torch import nn


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
                 ):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.d_output = d_output
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.project_layer = nn.Linear(self.d_input, self.d_model)
        self.layer = nn.TransformerEncoderLayer(self.d_model, self.num_heads)
        self.encoder = nn.TransformerEncoder(self.layer, self.num_layers)
        self.output_dense = nn.Linear(self.d_model, self.d_output)
        self.scale = math.sqrt(self.d_input)

    def forward(self, x: torch.Tensor, mask):
        """
        """
        seq_len = x.shape[1]
        x = self.project_layer(x)
        x = x * self.scale  # embedding scale
        x = x.transpose(1, 0)  # make batch second dim for transformer layer
        x = self.encoder(x, src_key_padding_mask=mask)
        x = x.transpose(1, 0)  # restore batch first
        y = x.contiguous().view(-1, x.shape[2])
        y = self.output_dense(y)
        y = nn.LogSoftmax(dim=1)(y)
        scores = y.view(-1, seq_len, self.d_output)
        return scores

    @staticmethod
    def get_transformer_mask(lengths: torch.Tensor, max_len, device):
        # mask is True for values to be masked
        mask_range = torch.arange(max_len) \
            .expand(len(lengths), max_len) \
            .to(device)
        transformer_mask = (mask_range >= lengths.unsqueeze(1))
        return transformer_mask


class LSTMEncoder(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        pass


class BertEmbeddings(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, token_ids, slices):
        pass


class ElmoEmbeddings(nn.Module):

    def __init__(self,
                 elmo_options,
                 elmo_weights,
                 elmo_size=None):
        super().__init__()
        self.elmo_options = elmo_options
        self.elmo_weights = elmo_weights
        self.elmo_size = elmo_size
        self.elmo = Elmo(self.elmo_options,
                         self.elmo_weights,
                         2, dropout=0)

    def forward(self, char_ids, mask=None):
        embeddings = self.elmo(char_ids)
        x = embeddings['elmo_representations'][1]
        return x

