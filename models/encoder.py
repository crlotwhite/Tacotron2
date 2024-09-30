import torch.nn as nn
import torch.nn.functional as F

from models.common import CNBN

class Encoder(nn.Module):
    def __init__(self, embed_dim, n_layers, kernel_size):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            CNBN(embed_dim, embed_dim, kernel_size, w_init_gain='relu')
            for _ in range(n_layers)
        ])
        self.lstm = nn.LSTM(embed_dim,
                            embed_dim // 2,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.conv_layers:
            x = conv(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        x = x.transpose(1, 2)

        input_lengths = input_lengths.detach().cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(x,
                                              input_lengths,
                                              batch_first=True)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return x

    def inference(self, x):
        for conv in self.conv_layers:
            x = conv(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x