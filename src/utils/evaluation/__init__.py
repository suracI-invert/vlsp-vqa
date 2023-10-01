from .bleu import Bleu
from .cider import Cider

import numpy as np
import json

def compute_scores(gts, gens):
    bleu = Bleu()
    cider = Cider()
    bleu_scores, _ = bleu.compute_score(gts, gens)
    avg_bleu_score = np.array(bleu_scores).mean()
    cider_score, _ = cider.compute_score(gts, gens)

    return {
        "BLEU": avg_bleu_score, 
        "CIDEr": cider_score
    }

def evaluate(truth_file: str, submission_file, code_phase) -> dict:
    gts = json.load(open(truth_file))
    gts = {id: [answer] for id, answer in gts.items()}

    gens = json.load(open(submission_file))
    gens = {id: [answer] for id, answer in gens.items()}

    if code_phase == "public_test":
        return compute_scores(gts, gens)
    else:
        gens = {id: answer for id, answer in gens.items() if id in gts}
        return compute_scores(gts, gens)