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
        pad_token_id = torch.tensor([self.tokenizer.pad_token_id])
        for sample in batch:
            img.append(sample['img'])
            questions.append(sample['question'])
            answers.append(sample['answer'])
        src = self.tokenizer(questions, return_tensors= 'pt', padding= 'longest')
        for i in range(len(src['input_ids'])):
            src['input_ids'][i] = torch.cat([src['input_ids'][i], pad_token_id], dim= -1)[1:]
            src['attention_mask'][i] = torch.cat([src['attention_mask'][i], pad_token_id], dim= -1)[1:]
        tgt = self.tokenizer(answers, return_tensors= 'pt', padding= 'longest')

        return {
            'img': torch.tensor(np.array(img, dtype= np.float32)),
            'src': src,
            'tgt': tgt,
            'answer': answers
        }
