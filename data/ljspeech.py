import torch
import re
import unicodedata

from datasets import load_dataset
from librosa.feature import melspectrogram
from torch.utils.data import DataLoader

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

    ds = ds.map(preprocess, num_proc=16)
    ds = ds.remove_columns(['id', 'file', 'text', 'normalized_text', 'audio'])
    ds = ds.with_format("torch")
    ds = ds.train_test_split(test_size=0.2, shuffle=True, seed=42)

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