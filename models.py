import torch
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
