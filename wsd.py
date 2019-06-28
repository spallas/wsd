import torch
from allennlp.modules.elmo import Elmo
from torch import nn

from data_preprocessing import SemCorDataLoader
from models import Attention


class WSDNet(nn.Module):

    _ELMO_OPTIONS = ''
    _ELMO_WEIGHTS = ''

    def __init__(self):
        super().__init__()
        pass

    def forward(self, *inputs):
        pass


class SimpleWSD(nn.Module):

    _ELMO_OPTIONS = 'res/elmo/elmo_2x1024_128_2048cnn_1xhighway_options.json'
    _ELMO_WEIGHTS = 'res/elmo/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'
    _ELMO_SIZE = 128
    _HIDDEN_SIZE = 128
    _NUM_LAYERS = 1

    def __init__(self, loader: SemCorDataLoader):
        super().__init__()
        self.tagset_size = len(loader.dataset.senses_count)
        self.pad_tag_index = 0
        self.elmo = Elmo(self._ELMO_OPTIONS,
                         self._ELMO_WEIGHTS,
                         2, dropout=0)
        self.embedding_size = 2 * self._ELMO_SIZE
        self.lstm = nn.LSTM(self.embedding_size,
                            hidden_size=self._HIDDEN_SIZE,
                            num_layers=self._NUM_LAYERS,
                            bidirectional=True,
                            batch_first=True)
        self.attention = Attention(self._HIDDEN_SIZE)
        self.output_dense = nn.Linear(self._HIDDEN_SIZE * 4, self.tagset_size)  # 2 directions * (state + attn)
        self.batch_size = loader.batch_size
        self.h, self.cell = self.init_hidden(self.batch_size)

    def init_hidden(self, batch_size):
        self.batch_size = batch_size
        return (torch.zeros(self._NUM_LAYERS * 2, self.batch_size, self._HIDDEN_SIZE),  # hidden state
                torch.zeros(self._NUM_LAYERS * 2, self.batch_size, self._HIDDEN_SIZE))  # cell state

    def forward(self, char_ids, lengths):

        embeddings = self.elmo(char_ids)
        x = embeddings
        hidden_states, (self.h, self.cell) = self.lstm(x, (self.h, self.cell))
        out = hidden_states

        y = self.attention(out)
        y = y.contiguous().view(-1, y.shape[2])
        y = self.output_dense(y)
        y = nn.LogSoftmax(dim=1)(y)
        tag_scores = y.view(self.batch_size, max(lengths), self.tagset_size)
        return tag_scores

    def loss(self, y, tags, device):
        y_true = torch.tensor(tags).view(-1).to(device)
        y = y.view(-1, self.tagset_size)
        mask = (y_true != self.pad_tag_index).float()
        num_tokens = int(torch.sum(mask).item())

        y_l = y[range(y.shape[0]), y_true] * mask
        ce_loss = - torch.sum(y_l) / num_tokens  # Negative LogLikelihood
        return ce_loss


class BaselineWSD(nn.Module):

    def __init__(self):
        super().__init__()
        pass

    def forward(self, *inputs):
        pass
