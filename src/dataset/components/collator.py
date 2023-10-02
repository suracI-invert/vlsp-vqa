#TODO: Add collator with masking

from typing import Any
import torch
import numpy as np 

class Collator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch) -> Any:
        img = []
        questions = []
        answers = []
        for sample in batch:
            img.append(sample['img'])
            questions.append(sample['question'])
            answers.append(sample['answer'])

        return {
            'img': torch.tensor(np.array(img, dtype= np.float32)),
            'src': self.tokenizer(questions, return_tensors= 'pt', padding= 'longest'),
            'tgt': self.tokenizer(answers, return_tensors= 'pt', padding= 'longest'),
            'answer': answers
        }