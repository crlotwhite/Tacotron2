import h5py
import os
import torch
import torch.nn as nn
import torch.optim as optim

from bases.base_trainer import BaseTrainer
from models.tacotron import Tacotron2
from models.loss import Tacotron2Loss
from preprocessors.ljspeech import LJSpeechPreprocessor
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchaudio.prototype.pipelines import HIFIGAN_VOCODER_V3_LJSPEECH as bundle
from tqdm import tqdm


def collate_fn(batch):
    mel_list = [data[0].squeeze() for data in batch]

    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([len(data[1]) for data in batch]), dim=0, descending=True
    )
    mel_lengths, ids_sorted_mel = torch.sort(
        torch.LongTensor([mel.shape[1] for mel in mel_list]), dim=0, descending=True
    )

    max_input_len = input_lengths[0]
    max_target_len = mel_lengths[0]

    text_padded = torch.LongTensor(len(batch), max_input_len).zero_()
    mel_padded = torch.FloatTensor(len(batch), 80, max_target_len).zero_()
    gate_padded = torch.FloatTensor(len(batch), max_target_len).zero_()
    output_lengths = torch.LongTensor(len(batch))

    # 배치 처리
    for i in range(len(ids_sorted_decreasing)):
        text = batch[ids_sorted_decreasing[i]][1]
        mel = mel_list[ids_sorted_decreasing[i]]
        mel = dynamic_range_compression(mel)
        mel_padded[i, :, :mel.size(1)] = mel
        gate_padded[i, mel.size(1):] = 1
        output_lengths[i] = mel.size(1)
        text_padded[i, :len(text)] = text

    return text_padded, input_lengths, mel_padded, gate_padded, output_lengths

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

class LJSpeechDataset(Dataset):
    def __init__(self, cfg, mode='train'):
        self.hdf_path = f'{cfg.datapath}/data.hdf5'
        self.mode = mode
        self.hdf = None

        with h5py.File(self.hdf_path, 'r') as hdf:
            self._length = len(hdf[f'{mode}/texts'])

    def _lazy_init_hdf(self):
        if self.hdf is None:
            self.hdf = h5py.File(self.hdf_path, 'r')

    def __getitem__(self, idx):
        self._lazy_init_hdf()

        mels_group = self.hdf[f'{self.mode}/mel_spectrograms']
        texts_group = self.hdf[f'{self.mode}/texts']

        mel_spectrogram = mels_group[str(idx)][:]
        text = texts_group[str(idx)][:]

        mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32)
        text = torch.tensor(text, dtype=torch.int)

        return mel_spectrogram, text

    def __len__(self):
        return self._length

    def __del__(self):
        if self.hdf is not None:
            self.hdf.close()


class LJSpeechTrainer(BaseTrainer):
    def __init__(self, cfg):
        self._epoch_start = 1
        self._total_epochs = cfg.train.total_epochs
        self.cfg = cfg
        self.criterion = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.experiment_name = cfg.train.experiment_name
        self.learning_rate = cfg.train.learning_rate
        self.log_interval = cfg.train.interval_tensorboard
        self.model = None
        self.optimizer = None
        self.resume_from = cfg.train.resume_from
        self.test_loader = None
        self.train_loader = None
        self.vocoder = bundle.get_vocoder().to(self.device)
        self.weight_decay = cfg.train.weight_decay
        self.writer = SummaryWriter(log_dir=f'logs/{self.experiment_name}')

        os.makedirs(f'checkpoints/{self.experiment_name}', exist_ok=True)

    def __del__(self):
        self.writer.close()

    @property
    def total_epochs(self):
        return self._total_epochs

    @property
    def epoch_start(self):
        return self._epoch_start

    def get_dataloader(self):
        train_dataset = LJSpeechDataset(self.cfg, 'train')
        test_dataset = LJSpeechDataset(self.cfg, 'test')

        self.train_loader = DataLoader(train_dataset, batch_size=self.cfg.train.batch_size, shuffle=True, collate_fn=collate_fn,
                                  pin_memory=True, pin_memory_device='cuda')
        self.test_loader = DataLoader(test_dataset, batch_size=self.cfg.train.batch_size, shuffle=False, collate_fn=collate_fn,
                                 pin_memory=True, pin_memory_device='cuda')

    def get_model(self):
        model = Tacotron2(cfg=self.cfg,
                          vocab_size=self.cfg.common.vocab_size,
                          mask_padding=self.cfg.common.mask_padding,
                          n_mel_channels=self.cfg.common.n_mel_channels,
                          n_frames_per_step=self.cfg.common.n_frames_per_step,
                          symbols_embedding_dim=self.cfg.common.symbols_embedding_dim)
        model.to(self.device)

        self.model = model
        self.optimizer = optim.Adam(model.parameters(),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)
        self.criterion = Tacotron2Loss()

        if self.cfg.train.resume_from > 0:
            self.load(self.cfg.train.resume_from)

    def load(self, epoch):
        checkpoint = torch.load(f'checkpoints/{self.experiment_name}/{self.experiment_name}_{epoch}.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._epoch_start = checkpoint['epoch']

    def save(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, f'checkpoints/{self.experiment_name}/{self.experiment_name}_{epoch}.pt')

    def tensorboard_log(self, epoch, train_loss, val_loss):
        train_loss, train_grad_norm = train_loss
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/valid', val_loss, epoch)

        if epoch % self.log_interval == 0:
            for i, text in enumerate(self.cfg.test.cases):
                sequence = LJSpeechPreprocessor.text_processing(text)
                sequence = torch.IntTensor(sequence).squeeze().to(self.device)
                mel_outputs, mel_outputs_postnet, _, alignments = self.model.inference(sequence)

                # Mel-spectrograms
                self.writer.add_image(f'Mel-spectrogram/melout_{i}', mel_outputs, epoch)
                self.writer.add_image(f'Mel-spectrogram/postnet_{i}', mel_outputs_postnet, epoch)

                # Alignment
                self.writer.add_image(f'Alignment/sample_{i}', alignments, epoch)

                # Audio
                audio_output = self.vocoder(mel_outputs_postnet).squeeze(0)
                audio_output = audio_output.detach().cpu().numpy()
                self.writer.add_audio(f'audio/sample_{i}', audio_output, epoch, bundle.sample_rate)

            self.save(epoch)

        print('\nEpoch: {}, Train Loss: {:.6f}, Train Grad Norm: {:.6f}, Test Loss: {:.6f}\n'.format(
            epoch, train_loss, train_grad_norm, val_loss
        ))

    def train(self):
        total_loss = 0
        total_grad_norm = 0

        with tqdm(self.train_loader, unit='batch') as pbar:
            for batch in pbar:
                self.optimizer.zero_grad(set_to_none=True)

                x, y = Tacotron2.parse_batch(batch)
                text_inputs, text_lengths, mels, max_len, output_lengths = x
                y_pred = self.model(text_inputs, text_lengths, mels, max_len, output_lengths)
                loss = self.criterion(y_pred, y)
                reduced_loss = loss.item()  # [1.0] -> 1.0
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                total_loss += reduced_loss
                total_grad_norm += grad_norm.item()

                pbar.set_postfix({'loss': reduced_loss, 'grad_norm': grad_norm.item()})

        return total_loss / len(self.train_loader), total_grad_norm / len(self.train_loader)

    def eval(self):
        total_loss = 0

        with torch.no_grad():
            with tqdm(self.test_loader, unit='batch') as pbar:
                for batch in pbar:
                    x, y = Tacotron2.parse_batch(batch)
                    text_inputs, text_lengths, mels, max_len, output_lengths = x
                    y_pred = self.model(text_inputs, text_lengths, mels, max_len, output_lengths)
                    loss = self.criterion(y_pred, y)
                    reduced_loss = loss.item()
                    total_loss += reduced_loss

                    pbar.set_postfix({'loss': reduced_loss})

        return total_loss / len(self.test_loader)
