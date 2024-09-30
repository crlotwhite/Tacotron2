import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod

class CommonNorm(nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__()

        self.layer = self.get_layer(**kwargs)

        gain = nn.init.calculate_gain(kwargs.get('w_init_gain', 'linear'))
        nn.init.xavier_uniform_(self.layer.weight, gain=gain)

    def forward(self, x):
        return self.layer(x)

    @abstractmethod
    def get_layer(self, **kwargs):
        raise NotImplementedError

class LinearNorm(CommonNorm):
    def get_layer(self, **kwargs):
        in_dim = kwargs['in_dim']
        out_dim = kwargs['out_dim']
        bias = kwargs.get('bias', True)
        return nn.Linear(in_dim, out_dim, bias=bias)

class ConvNorm(CommonNorm):
    def get_layer(self, **kwargs):
        in_channels = kwargs['in_channels']
        out_channels = kwargs['out_channels']
        kernel_size = kwargs.get('kernel_size', 1)
        stride = kwargs.get('stride', 1)
        padding = kwargs.get('padding', None)
        dilation = kwargs.get('dilation', 1)
        bias = kwargs.get('bias', True)

        if padding is None:
            assert kernel_size % 2 == 1, "kernel size must be odd for padding"
            padding = dilation * (kernel_size - 1) // 2
        return nn.Conv1d(in_channels,
                         out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         bias=bias)

class CNBN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 w_init_gain='linear'):
        super().__init__()
        self.conv = ConvNorm(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             w_init_gain=w_init_gain)

        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super().__init__()

        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList([
            LinearNorm(in_dim=in_size, out_dim=out_size, bias=False)
            for (in_size, out_size) in zip(in_sizes, sizes)
        ])

    def forward(self, x):
        for linear in self.layers:
            x = linear(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=True)

        return x

class Postnet(nn.Module):
    def __init__(self, n_mel_channels, embed_dim, kernel_size, n_layers):
        super().__init__()

        self.conv_layers = nn.ModuleList(
            [CNBN(n_mel_channels, embed_dim, kernel_size, w_init_gain='tanh'), ]
            + [
                CNBN(embed_dim, embed_dim, kernel_size, w_init_gain='tanh')
                for _ in range(n_layers-2)
            ]
        )
        self.last_conv = CNBN(embed_dim, n_mel_channels,
                              kernel_size, w_init_gain='linear')

        assert len(self.conv_layers) == n_layers - 1

    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)
            x = torch.tanh(x)
            x = F.dropout(x, p=0.5, training=self.training)

        x = self.last_conv(x)
        x = F.dropout(x, p=0.5, training=self.training)

        return x