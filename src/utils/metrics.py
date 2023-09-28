#TODO: Add CIDEr-D

from torchmetrics import Metric
import torch

class MeanBLEU(Metric):
    higher_is_better: bool = True
    def __init__(self):
        super().__init__()
        self.add_state("value", default= torch.tensor(0), dist_reduce_fx= 'sum')
    
    def update(self, list_values):
        bleu1, bleu2, bleu3, bleu4 = list_values
        self.value = bleu1 + bleu2 + bleu3 + bleu4
    
    def compute(self):
        return self.value / 4