import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import ConvNorm, LinearNorm

class LocationLayer(nn.Module):
    def __init__(self, n_filtters, kernel_size, dim):
        super().__init__()

        self.conv = ConvNorm(in_channels=2,
                             out_channels=n_filtters,
                             kernel_size=kernel_size,
                             bias=False)
        self.linear = LinearNorm(in_dim=n_filtters,
                                 out_dim=dim,
                                 bias=False,
                                 w_init_gain='tanh')

    def forward(self, x):
        attn = self.conv(x)
        attn = attn.transpose(1, 2)
        attn = self.linear(attn)
        return attn

class Attention(nn.Module):
    def __init__(self, rnn_dim, embed_dim, attn_dim, n_filters, kernel_size):
        super().__init__()

        self.q = LinearNorm(in_dim=rnn_dim,
                            out_dim=attn_dim,
                            bias=False,
                            w_init_gain='tanh')
        self.mem = LinearNorm(in_dim=embed_dim,
                              out_dim=attn_dim,
                              bias=False,
                              w_init_gain='tanh')
        self.v = LinearNorm(in_dim=attn_dim,
                            out_dim=1,
                            bias=False)
        self.location = LocationLayer(n_filters,
                                      kernel_size,
                                      attn_dim)
        self.score_mask_value = -float('inf')

    def get_alignment_energies(self, query, processed_memory, attn_weights_cat):
        processed_query = self.q(query.unsqueeze(1))
        processed_attn_weights = self.location(attn_weights_cat)
        energies = processed_query + processed_attn_weights + processed_memory
        energies = torch.tanh(energies)
        energies = self.v(energies).squeeze(-1)
        return energies

    def forward(self, attn_hidden_state, memory, processed_memory,
                attn_weights_cat, mask):
        alignment = self.get_alignment_energies(attn_hidden_state,
                                                processed_memory,
                                                attn_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attn_weights = F.softmax(alignment, dim=1)
        attn_context = torch.bmm(attn_weights.unsqueeze(1), memory)
        attn_context = attn_context.squeeze(1)
        return attn_context, attn_weights