import torch
from fairseq.models.transformer import TransformerEncoderLayer
from torch import nn

from utils.config import TransformerConfig


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

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.transformer_encoder = TransformerEncoderLayer(config)

    def forward(self, x: torch.Tensor, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        return self.transformer_encoder(x, encoder_padding_mask)
