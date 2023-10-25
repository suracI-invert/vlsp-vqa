import torch
from src.utils.metrics import BLEU_CIDEr
from lightning.pytorch.utilities.grads import grad_norm
import argparse
import timm

if __name__ == '__main__':
    # img = torch.rand((1, 3, 224, 224))
    # print(img.reshape(1, 3, 224 * 224).permute(0, 2, 1).shape)
    print(timm.list_models('*efficientnet*'))