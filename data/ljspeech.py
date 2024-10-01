import torch
import re
import unicodedata

from datasets import load_dataset, load_from_disk
from librosa.feature import melspectrogram
from torch.utils.data import DataLoader

LJSPEECH_STORE_PATH = "store/ljspeech"

vocab = " abcdefghijklmnopqrstuvwxyz'.?"  # P: Padding, E: EOS.
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for idx, char in enumerate(vocab)}

def preprocess(example):
    text = unicodedata.normalize('NFD', example['normalized_text'])
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    text = re.sub(f"[^{vocab}]+", " ", text.lower()).strip()

    example["encoded_text"] = [char2idx[ch.lower()] for ch in text]
    example["mel"] = melspectrogram(y=example['audio']['array'], sr=22050, n_fft=1024, hop_length=256, n_mels=80, win_length=1024, fmin=0.0, fmax=8000.0)
    return example

def collate_fn(batch):
    mel_list = [data['mel'] for data in batch]

    input_lengths = torch.LongTensor([len(data['encoded_text']) for data in batch])
    ids_sorted_decreasing = torch.argsort(input_lengths, descending=True)

    mel_lengths = torch.LongTensor([mel.shape[1] for mel in mel_list])
    max_input_len = input_lengths.max().item()
    max_target_len = mel_lengths.max().item()

    text_padded = torch.zeros((len(batch), max_input_len), dtype=torch.long)
    mel_padded = torch.zeros((len(batch), 80, max_target_len))
    gate_padded = torch.zeros((len(batch), max_target_len))
    output_lengths = torch.zeros(len(batch), dtype=torch.long)

    for i, idx in enumerate(ids_sorted_decreasing):
        text = batch[idx]['encoded_text']
        mel = dynamic_range_compression(mel_list[idx])

        mel_padded[i, :, :mel.size(1)] = mel
        gate_padded[i, mel.size(1)-1:] = 1
        output_lengths[i] = mel.size(1)

        text_padded[i, :len(text)] = text.clone().detach()
        # text_padded[i, :len(text)] = torch.tensor(text, dtype=torch.long)

    return text_padded, input_lengths[ids_sorted_decreasing], mel_padded, gate_padded, output_lengths

def dynamic_range_compression(x, C=1, clip_val=1e-5):
        return torch.log(torch.clamp(x, min=clip_val) * C)

def from_hf():
    ds = load_dataset("lj_speech",
                      split='train',
                      trust_remote_code=True)

    ds = ds.map(preprocess, num_proc=8)
    ds = ds.remove_columns(['id', 'file', 'text', 'normalized_text', 'audio'])
    ds = ds.with_format("torch")
    ds = ds.train_test_split(test_size=0.2, shuffle=True, seed=42)
    ds.save_to_disk(LJSPEECH_STORE_PATH)
    return ds

def from_disk():
    return load_from_disk(LJSPEECH_STORE_PATH)

def _load_data(ds):
    train_set, test_set = ds["train"], ds["test"]
    train_loader = DataLoader(train_set,
                              batch_size=64,
                              shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=4)
                              # pin_memory=True,
                              # pin_memory_device='cuda')
    test_loader = DataLoader(test_set,
                             batch_size=64,
                             shuffle=False,
                             collate_fn=collate_fn,
                             num_workers=4)
                             # pin_memory=True,
                             # pin_memory_device='cuda')

    return train_loader, test_loader

def load_data():
    import torchaudio

    vocab = " abcdefghijklmnopqrstuvwxyz'.?"  # P: Padding, E: EOS.
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}

    def text_normalize(text):
        text = ''.join(char for char in unicodedata.normalize('NFD', text)
                       if unicodedata.category(char) != 'Mn')  # Strip accents

        text = text.lower()
        text = re.sub("[^{}]".format(vocab), " ", text)
        text = re.sub("[ ]+", " ", text)
        return text

    class Collate:
        def __init__(self):
            self.wav_to_mel = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_mels=80, win_length=1024,
                                                                   hop_length=256, f_min=0.0, f_max=8000.0, n_fft=1024)

        def __call__(self, batch):
            # batch: N_batch * [wav, sample_rate, text, text_normalized]
            mel_list = []
            for data in batch:
                wav = data[0]
                mel_list.append(self.wav_to_mel(wav).squeeze())
            input_lengths, ids_sorted_decreasing = torch.sort(torch.LongTensor([len(data[3]) for data in batch]), dim=0,
                                                              descending=True)
            mel_lengths, ids_sorted_mel = torch.sort(torch.LongTensor([mel.shape[1] for mel in mel_list]), dim=0,
                                                     descending=True)

            max_input_len = input_lengths[0]
            max_target_len = mel_lengths[0]

            text_padded = torch.LongTensor(len(batch), max_input_len)
            mel_padded = torch.FloatTensor(len(batch), 80, max_target_len)
            gate_padded = torch.FloatTensor(len(batch), max_target_len)
            output_lengths = torch.LongTensor(len(batch))

            text_padded.zero_()
            mel_padded.zero_()
            gate_padded.zero_()

            for i in range(len(ids_sorted_decreasing)):
                _, _, _, text = batch[ids_sorted_decreasing[i]]
                mel = mel_list[ids_sorted_decreasing[i]]
                mel = self.dynamic_range_compression(mel)
                mel_padded[i, :, :mel.size(1)] = mel
                gate_padded[i, mel.size(1) - 1:] = 1
                output_lengths[i] = mel.size(1)
                text = text_normalize(text)
                text = [char2idx[char] for char in text]
                text_norm = torch.IntTensor(text)
                text_padded[i, :len(text)] = text_norm

            return text_padded, input_lengths, mel_padded, gate_padded, output_lengths

        def dynamic_range_compression(self, x, C=1, clip_val=1e-5):
            return torch.log(torch.clamp(x, min=clip_val) * C)

    LJSpeech_url = 'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2'
    dataset = torchaudio.datasets.LJSPEECH("", url=LJSpeech_url, download=True)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    collate_fn = Collate()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False,
                              collate_fn=collate_fn, pin_memory=True, pin_memory_device='cuda')
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False,
                              collate_fn=collate_fn, pin_memory=True, pin_memory_device='cuda')

    return train_loader, test_loader