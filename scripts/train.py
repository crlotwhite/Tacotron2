import argparse
import gc
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from data.ljspeech import vocab, LJSPEECH_STORE_PATH, load_data
from models.tacotron import Tacotron2
from models.loss import Tacotron2Loss
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torchaudio.prototype.pipelines import HIFIGAN_VOCODER_V3_LJSPEECH as bundle
from tqdm import tqdm

root_dir = Path(__file__).parent.parent.resolve()
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))

def train_loop():
    total_loss = 0
    total_grad_norm = 0

    with tqdm(train_loader, unit='batch') as pbar:
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)

            x, y = Tacotron2.parse_batch(batch)
            text_inputs, text_lengths, mels, max_len, output_lengths = x
            y_pred = model(text_inputs, text_lengths, mels, max_len, output_lengths)
            loss = criterion(y_pred, y)
            reduced_loss = loss.item()  # [1.0] -> 1.0
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += reduced_loss
            total_grad_norm += grad_norm.item()

            pbar.set_postfix({'loss': reduced_loss, 'grad_norm': grad_norm.item()})

    return total_loss / len(train_loader), total_grad_norm / len(train_loader)

def eval_loop():
    total_loss = 0

    with torch.no_grad():
        with tqdm(test_loader, unit='batch') as pbar:
            for batch in pbar:
                x, y = Tacotron2.parse_batch(batch)
                text_inputs, text_lengths, mels, max_len, output_lengths = x
                y_pred = model(text_inputs, text_lengths, mels, max_len, output_lengths)
                loss = criterion(y_pred, y)
                reduced_loss = loss.item()
                total_loss += reduced_loss

                pbar.set_postfix({'loss': reduced_loss})

    return total_loss / len(test_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--resume-from', type=int, default=-1)

    args = parser.parse_args()

    gc.disable()

    # if os.path.exists(LJSPEECH_STORE_PATH):
    #     train_loader, test_loader = from_disk()
    # else:
    #     train_loader, test_loader = from_hf()

    train_loader, test_loader = load_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Tacotron2(vocab_size=len(vocab),
                      mask_padding=True,
                      n_mel_channels=80,
                      n_frames_per_step=1,
                      symbols_embedding_dim=512)
    model.to(device)

    learning_rate = 1e-3
    weight_decay = 1e-6
    total_epochs = 100
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)
    criterion = Tacotron2Loss()
    vocoder = bundle.get_vocoder()
    writer = SummaryWriter(log_dir=f'logs/{args.experiment}')
    epoch_start = 1

    if args.resume_from > 0:
        checkpoint = torch.load(f'checkpoints/{args.experiment}/{args.experiment}_{args.resume_from}.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']

    try:
        for epoch in range(epoch_start, total_epochs + 1):
            train_loss, train_grad_norm = train_loop()
            test_loss = eval_loop()

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/valid', test_loss, epoch)

            if epoch % 5 == 0:
                batch = next(iter(test_loader))
                x, y = Tacotron2.parse_batch(batch)
                text_inputs, text_lengths, mels, max_len, output_lengths = x
                mel_out, _, _, _ = model(text_inputs, text_lengths, mels, max_len, output_lengths)
                mel_outputs = mel_out.detach().cpu().numpy()
                mel_targets = mels.detach().cpu().numpy()
                for idx in range(5):
                    mel_output = mel_outputs[idx]
                    mel_target = mel_targets[idx]
                    mel_length = torch.Tensor(mel_target.size(1))

                    # Mel-spectrograms
                    writer.add_image(f'Mel-spectrogram/output_epoch_{epoch}_sample_{idx}', mel_output, dataformats='HW')
                    writer.add_image(f'Mel-spectrogram/target_epoch_{epoch}_sample_{idx}', mel_target, dataformats='HW')
                    
                    # TODO: Audio 출력 기능 수정하기
                    # Audio
                    # audio_output = vocoder(mel_output, lengths=mel_length).squeeze(0)
                    # audio_target = vocoder(mel_target, lengths=mel_length).squeeze(0)
                    # writer.add_audio(f'Audio/output_epoch_{epoch}_sample_{idx}', audio_output,
                    #                  sample_rate=bundle.sample_rate)
                    # writer.add_audio(f'Audio/target_epoch_{epoch}_sample_{idx}', audio_target,
                    #                  sample_rate=bundle.sample_rate)

                    # Attention Map
                    attention_map = model.decoder.attention_map.detach().cpu().numpy()
                    writer.add_image(f'Attention Map/epoch_{epoch}_sample_{idx}', attention_map[idx], dataformats='HW')

                # checkpoints
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f'checkpoints/{args.experiment}/{args.experiment}_{epoch}.pt')

            print("\nEpoch: {}, Train Loss: {:.6f}, Train Grad Norm: {:.6f}, Test Loss: {:.6f}".format(
                epoch, train_loss, train_grad_norm, test_loss
            ))
    finally:
        writer.close()