#TODO: Add CIDEr-D

from typing import List

from src.utils.evaluation import compute_scores

class BLEU_CIDEr(object):
    def __init__(self):
        self.val = []

    def update(self, pred: List, label: List):
        for i in range(len(pred)):
            self.val.append((pred[i], label[i]))
    
    def compute(self):
        preds_dict = {}
        labels_dict = {}
        for i in range(len(self.val)):
            preds_dict[i] = [self.val[i][0]]
            labels_dict[i] = [self.val[i][1]]
        score = compute_scores(labels_dict, preds_dict)
        return score
    
    def reset(self):
        self.val = []