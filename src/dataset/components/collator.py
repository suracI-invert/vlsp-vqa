#TODO: Add collator with masking

from typing import Any
import torch
import numpy as np 

class Collator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch) -> Any:
        fname = []
        img = []
        questions = []
        answers = []
        ocr = []
        pad_token_id = torch.tensor(self.tokenizer.pad_token_id)
        for sample in batch:
            fname.append(sample['img_fname'])
            img.append(sample['img'])
            questions.append(sample['question'])
            answers.append(sample['answer'])
            # ocr_tokenized = self.tokenizer(sample['ocr'], return_tensors= 'pt', padding= 'longest') if len(sample['ocr']) > 0 else {'input_ids': None, 'attention_mask': None}
            # ocr.append(ocr_tokenized)
        src = self.tokenizer(questions, return_tensors= 'pt', padding= 'longest')
        src['input_ids'] = src['input_ids'][:, 1:]
        src['attention_mask'] = src['attention_mask'][:, 1:]
        tgt = self.tokenizer(answers, return_tensors= 'pt', padding= 'longest')
        tgt_label = tgt['input_ids'][:, 1:].clone()
        for i in range(len(tgt['input_ids'])):
            tgt['input_ids'][i][tgt['input_ids'][i] == self.tokenizer.eos_token_id] = self.tokenizer.pad_token_id
        tgt['input_ids'] = tgt['input_ids'][:, :-1]
        tgt['attention_mask'] = (tgt['input_ids'] == self.tokenizer.pad_token_id)
        
        return {
            'fname': fname,
            'img': torch.stack(img),
            'src': src,
            'tgt': tgt,
            'tgt_label': tgt_label,
            'answer': answers,
            # 'ocr': ocr
        }
