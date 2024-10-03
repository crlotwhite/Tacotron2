import gc
import hydra
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from data.ljspeech import load_data
from models.tacotron import Tacotron2
from models.loss import Tacotron2Loss
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

root_dir = Path(__file__).parent.parent.resolve()
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))

def train_loop(data_loader, model, optimizer, criterion):
    total_loss = 0
    total_grad_norm = 0

    with tqdm(data_loader, unit='batch') as pbar:
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

    return total_loss / len(data_loader), total_grad_norm / len(data_loader)

def eval_loop(data_loader, model, criterion):
    total_loss = 0

    with torch.no_grad():
        with tqdm(data_loader, unit='batch') as pbar:
            for batch in pbar:
                x, y = Tacotron2.parse_batch(batch)
                text_inputs, text_lengths, mels, max_len, output_lengths = x
                y_pred = model(text_inputs, text_lengths, mels, max_len, output_lengths)
                loss = criterion(y_pred, y)
                reduced_loss = loss.item()
                total_loss += reduced_loss

                pbar.set_postfix({'loss': reduced_loss})

    return total_loss / len(data_loader)

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg):
    experiment_name = cfg.train.experiment_name
    resume_from = cfg.train.resume_from

    gc.disable()

    os.makedirs(f'checkpoints/{experiment_name}', exist_ok=True)

    train_loader, test_loader = load_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Tacotron2(cfg=cfg,
                      vocab_size=cfg.common.vocab_size,
                      mask_padding=cfg.common.mask_padding,
                      n_mel_channels=cfg.common.n_mel_channels,
                      n_frames_per_step=cfg.common.n_frames_per_step,
                      symbols_embedding_dim=cfg.common.symbols_embedding_dim)
    model.to(device)

    learning_rate = cfg.train.learning_rate
    weight_decay = cfg.train.weight_decay
    total_epochs = cfg.train.total_epochs
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)
    criterion = Tacotron2Loss()
    
    writer = SummaryWriter(log_dir=f'logs/{experiment_name}')
    epoch_start = 1

    if resume_from > 0:
        checkpoint = torch.load(f'checkpoints/{experiment_name}/{experiment_name}_{resume_from}.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']

    try:
        for epoch in range(epoch_start, total_epochs + 1):
            train_loss, train_grad_norm = train_loop(train_loader, model, optimizer, criterion)
            test_loss = eval_loop(test_loader, model, criterion)

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/valid', test_loss, epoch)

            if epoch % cfg.train.interval_tensorboard == 0:
                batch = next(iter(test_loader))
                x, y = Tacotron2.parse_batch(batch)
                text_inputs, text_lengths, mels, max_len, output_lengths = x
                mel_out, _, _, _ = model(text_inputs, text_lengths, mels, max_len, output_lengths)
                mel_outputs = mel_out.detach().cpu().numpy()
                mel_targets = mels.detach().cpu().numpy()
                for idx in range(cfg.train.sampling_data):
                    mel_output = mel_outputs[idx]
                    mel_target = mel_targets[idx]

                    # Mel-spectrograms
                    writer.add_image(f'Mel-spectrogram/output_epoch_{epoch}_sample_{idx}', mel_output, dataformats='HW')
                    writer.add_image(f'Mel-spectrogram/target_epoch_{epoch}_sample_{idx}', mel_target, dataformats='HW')

                    # Attention Map
                    attention_map = model.decoder.attention_map.detach().cpu().numpy()
                    writer.add_image(f'Attention Map/epoch_{epoch}_sample_{idx}', attention_map[idx], dataformats='HW')

                # checkpoints
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f'checkpoints/{experiment_name}/{experiment_name}_{epoch}.pt')

            print("\nEpoch: {}, Train Loss: {:.6f}, Train Grad Norm: {:.6f}, Test Loss: {:.6f}\n".format(
                epoch, train_loss, train_grad_norm, test_loss
            ))
    finally:
        writer.close()

if __name__ == '__main__':
    main()