#TODO: Add CIDEr-D

from typing import List
from torchmetrics import Metric
import torch

from src.utils.evaluation import compute_scores

from collections import defaultdict

class BLEU_CIDEr(Metric):
    higher_is_better: bool = True
    def __init__(self):
        super().__init__()
        self.add_state("preds", default= defaultdict(), dist_reduce_fx= 'sum')
        self.add_state('labels', default= defaultdict(), dist_reduce_fx= 'sum')

    def update(self, key, pred: List, label: List):
        self.preds[key] = pred
        self.labels[key] = label
    
    def compute(self):
        score = compute_scores(self.labels, self.preds)
        return (score['BLEU'], score['CIDEr'])