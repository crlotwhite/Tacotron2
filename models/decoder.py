import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import Attention
from models.common import Prenet, LinearNorm
from models.utils import get_mask_from_lengths

class Decoder(nn.Module):
    def __init__(self,
                 n_mel_channels,
                 n_frames_per_step,
                 encoder_embed_dim,
                 attn_rnn_dim,
                 attn_dim,
                 attn_location_n_filters,
                 attn_location_kernel_size,
                 decoder_rnn_dim,
                 prenet_dim,
                 max_decoder_steps,
                 gate_threshold,
                 p_attn_dropout,
                 p_decoder_dropout):
        super().__init__()

        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step
        self.encoder_embed_dim = encoder_embed_dim
        self.attn_rnn_dim = attn_rnn_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dim = prenet_dim
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        self.p_attn_dropout = p_attn_dropout
        self.p_decoder_dropout = p_decoder_dropout
        self.register_buffer('attention_map', None)

        self.prenet = Prenet(n_mel_channels * n_frames_per_step,
                             [prenet_dim, prenet_dim])

        self.attn_rnn = nn.LSTMCell(prenet_dim + encoder_embed_dim,
                                    attn_rnn_dim)

        self.attention = Attention(attn_rnn_dim,
                                   encoder_embed_dim,
                                   attn_dim,
                                   attn_location_n_filters,
                                   attn_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(attn_rnn_dim + encoder_embed_dim,
                                       decoder_rnn_dim)

        self.linear_projection = LinearNorm(in_dim=decoder_rnn_dim + encoder_embed_dim,
                                            out_dim=n_mel_channels * n_frames_per_step)

        self.gate = LinearNorm(in_dim=decoder_rnn_dim + encoder_embed_dim,
                               out_dim=1,
                               w_init_gain='sigmoid')

    def init_decoder_state(self, memory, mask):
        batch_size = memory.size(0)
        max_time = memory.size(1)

        attn_hidden = memory.new_zeros(batch_size, self.attn_rnn_dim)
        attn_cell = memory.new_zeros(batch_size, self.attn_rnn_dim)
        decoder_hidden = memory.new_zeros(batch_size, self.decoder_rnn_dim)
        decoder_cell = memory.new_zeros(batch_size, self.decoder_rnn_dim)
        attn_weights = memory.new_zeros(batch_size, max_time)
        attn_weights_cum = memory.new_zeros(batch_size, max_time)
        attn_context = memory.new_zeros(batch_size, self.encoder_embed_dim)
        processed_memory = self.attention.mem(memory)

        return (
            attn_hidden,
            attn_cell,
            decoder_hidden,
            decoder_cell,
            attn_weights,
            attn_weights_cum,
            attn_context,
            processed_memory
        )

    def get_go_frame(self, memory):
        batch_size = memory.size(0)
        decoder_input = memory.new_zeros(
            batch_size, self.n_mel_channels * self.n_frames_per_step)
        return decoder_input

    def parse_decoder_inputs(self, decoder_inputs):
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(decoder_inputs.size(0),
                                            decoder_inputs.size(1) // self.n_frames_per_step,
                                            -1)

        return decoder_inputs.transpose(0, 1)

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decoder(self,
                decoder_input,
                memory,
                mask,
                attn_hidden,
                attn_cell,
                decoder_hidden,
                decoder_cell,
                attn_weights,
                attn_weights_cum,
                attn_context,
                processed_memory):

        cell_input = torch.cat((decoder_input, attn_context), -1)
        attn_hidden, attn_cell = self.attn_rnn(cell_input,
                                               (attn_hidden, attn_cell))
        attn_hidden = F.dropout(attn_hidden,
                                self.p_attn_dropout,
                                self.training)

        attn_weights_cat = torch.cat(
            (attn_weights.unsqueeze(1), attn_weights_cum.unsqueeze(1)), dim=1)
        attn_context, attn_weights = self.attention(attn_hidden,
                                                    memory,
                                                    processed_memory,
                                                    attn_weights_cat,
                                                    mask)
        attn_weights_cum += attn_weights

        decoder_input = torch.cat((attn_hidden, attn_context), -1)
        decoder_hidden, decoder_cell = self.decoder_rnn(decoder_input,
                                                        (decoder_hidden, decoder_cell))
        decoder_hidden = F.dropout(decoder_hidden,
                                   self.p_decoder_dropout,
                                   self.training)

        decoder_hidden_attn_context = torch.cat((decoder_hidden, attn_context),
                                                dim=1)
        decoder_output = self.linear_projection(decoder_hidden_attn_context)
        gate_prediction = self.gate(decoder_hidden_attn_context)

        return (
            decoder_output,
            gate_prediction,
            attn_hidden,
            attn_cell,
            decoder_hidden,
            decoder_cell,
            attn_weights,
            attn_weights_cum,
            attn_context,
        )

    def forward(self, memory, decoder_inputs, memory_lengths):
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)
        mask = ~get_mask_from_lengths(memory_lengths)
        attn_weights_stack = []

        (
            attn_hidden,
            attn_cell,
            decoder_hidden,
            decoder_cell,
            attn_weights,
            attn_weights_cum,
            attn_context,
            processed_memory
        ) = self.init_decoder_state(memory, mask)

        mel_outputs = []
        gate_outputs = []
        alignments = []

        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            (
                mel_output,
                gate_output,
                attn_hidden,
                attn_cell,
                decoder_hidden,
                decoder_cell,
                attn_weights,
                attn_weights_cum,
                attn_context,
            ) = self.decoder(
                decoder_input,
                memory,
                mask,
                attn_hidden,
                attn_cell,
                decoder_hidden,
                decoder_cell,
                attn_weights,
                attn_weights_cum,
                attn_context,
                processed_memory
            )

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attn_weights]
            attn_weights_stack += [attn_weights]

        attn_weights_stack = torch.stack(attn_weights_stack, dim=1)
        self.attention_map = attn_weights_stack
        return self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)

    def inference(self, memory):
        decoder_input = self.get_go_frame(memory)

        (
            attn_hidden,
            attn_cell,
            decoder_hidden,
            decoder_cell,
            attn_weights,
            attn_weights_cum,
            attn_context,
            processed_memory
        ) = self.init_decoder_state(memory, mask=None)

        mel_outputs = []
        gate_outputs = []
        alignments = []

        while True:
            decoder_input = self.prenet(decoder_input)
            (
                mel_output,
                gate_output,
                attn_hidden,
                attn_cell,
                decoder_hidden,
                decoder_cell,
                attn_weights,
                attn_weights_cum,
                attn_context,
            ) = self.decoder(
                decoder_input,
                memory,
                None,
                attn_hidden,
                attn_cell,
                decoder_hidden,
                decoder_cell,
                attn_weights,
                attn_weights_cum,
                attn_context,
                processed_memory
            )

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [attn_weights]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print('위험! 최대 스텝 수 초과')
                break

            decoder_input = mel_output

        return self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)
