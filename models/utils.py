import torch

def get_mask_from_lengths(lengths):
    max_len = lengths.max().item()
    ids = torch.arange(max_len, device=lengths.device, dtype=torch.long)
    return ids < lengths.unsqueeze(1)


def to_gpu(x):
    x = x.contiguous()
    return x.cuda(non_blocking=True) if torch.cuda.is_available() else x
