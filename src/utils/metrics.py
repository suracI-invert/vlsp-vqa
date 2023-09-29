#TODO: Add CIDEr-D

from typing import List
from torchmetrics import Metric
import torch

from src.utils.evaluation import compute_scores

class BLEU_CIDEr(Metric):
    higher_is_better: bool = True
    def __init__(self):
        super().__init__()
        self.add_state('keys', default= [])
        self.add_state('preds', default= [])
        self.add_state('labels', default= [])

    def update(self, pred: List, label: List):
        for i in range(len(pred)):
            self.preds.append(pred[i])
            self.labels.append(label[i])
    
    def compute(self):
        preds_dict = {}
        labels_dict = {}
        for i in range(len(self.labels)):
            preds_dict[i] = [self.preds[i]]
            labels_dict[i] = [self.labels[i]]
        score = compute_scores(labels_dict, preds_dict)
        return score