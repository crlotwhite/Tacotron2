import math

import torch
import torch.nn as nn

from models.common import Postnet
from models.decoder import Decoder
from models.encoder import Encoder
from models.utils import to_gpu, get_mask_from_lengths

class Tacotron2(nn.Module):
    def __init__(self,
                 cfg,
                 vocab_size,
                 mask_padding,
                 n_mel_channels,
                 n_frames_per_step,
                 symbols_embedding_dim):
        super().__init__()

        self.mask_padding = mask_padding
        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step
        self.embedding = nn.Embedding(vocab_size, symbols_embedding_dim)
        std = math.sqrt(2.0 / (vocab_size + symbols_embedding_dim))
        val = math.sqrt(3.0) * std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(embed_dim=cfg.encoder.embed_dim,
                               n_layers=cfg.encoder.n_layers,
                               kernel_size=cfg.encoder.kernel_size)
        self.decoder = Decoder(n_mel_channels=n_mel_channels,
                               n_frames_per_step=n_frames_per_step,
                               encoder_embed_dim=cfg.encoder.embed_dim,
                               attn_rnn_dim=cfg.attention.rnn_dim,
                               attn_dim=cfg.attention.attn_dim,
                               attn_location_n_filters=cfg.location.n_filters,
                               attn_location_kernel_size=cfg.location.kernel_size,
                               decoder_rnn_dim=cfg.decoder.rnn_dim,
                               prenet_dim=cfg.decoder.prenet_dim,
                               max_decoder_steps=cfg.decoder.max_steps,
                               gate_threshold=cfg.decoder.gate_threshold,
                               p_attn_dropout=cfg.attention.p_dropout,
                               p_decoder_dropout=cfg.decoder.p_dropout)
        self.postnet = Postnet(n_mel_channels=n_mel_channels,
                               embed_dim=cfg.postnet.embed_dim,
                               kernel_size=cfg.postnet.kernel_size,
                               n_layers=cfg.postnet.n_layers)

    @classmethod
    def parse_batch(cls, batch):
        (
            text_padded,
            input_lengths,
            mel_padded,
            gate_padded,
            output_lengths
        ) = batch

        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded)
        )

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)

        return outputs

    def forward(self, text_inputs, text_lengths, mels, max_len, output_lengths):
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs,
                                                             mels,
                                                             text_lengths)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            (mel_outputs, mel_outputs_postnet, gate_outputs, alignments),
            output_lengths
        )

    def inference(self, inputs):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        (
            mel_outputs,
            gate_outputs,
            alignments
        ) = self.decoder.inference(encoder_outputs)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            (mel_outputs, mel_outputs_postnet, gate_outputs, alignments)
        )
