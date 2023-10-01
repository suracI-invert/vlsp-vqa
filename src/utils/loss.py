import torch
from torch import nn

class LabelSmoothingLoss(nn.Module):
    def __init__(self, pad_idx, label_smoothing: float=0.0) -> None:
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss(ignore_index    = pad_idx,
                                             label_smoothing = label_smoothing)

    def forward(self, logits, labels):
        vocab_size = logits.shape[-1]
        logits = logits.reshape(-1, vocab_size)
        labels = labels.reshape(-1).long()
        return self.loss_func(logits, labels)