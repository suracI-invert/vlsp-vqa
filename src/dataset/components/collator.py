#TODO: Add collator with masking
# TODO: ocr -> graph should be here

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
        graphs = []
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

        # for i in range(len(batch)):
        #     ocr_id = ocr[i]['input_ids']
        #     question_id = src['input_ids'][i]
        #     node, edge = self.make_node_edge_array(question_id, ocr_id)
            # graphs.append({
            #     'node_id': node
            # })
            
        return {
            'fname': fname,
            'img': torch.stack(img),
            'src': src,
            'tgt': tgt,
            'tgt_label': tgt_label,
            'answer': answers,
            # 'ocr': ocr
        }


    def reformat_input_ids(self, input_ids):
        """
            - input: input_ids of ocr 
            - output: remove BOS, EOS and padding from it
               
        """
        cleaned_input_ids = [[] for token_id in input_ids if token_id is None]

        tokens_to_remove = [
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id
        ]

        cleaned_input_ids = [token_id for token_id in input_ids if token_id not in tokens_to_remove and token_id is not None]

        return cleaned_input_ids
        

    def make_node_edge_array(self, question_ids, ocr_ids):
        """
            - input: question input ids, ocr results input ids 
            - output: node and edge array
               
        """
        question_ids = self.reformat_input_ids(question_ids)
        ocr_ids = self.reformat_input_ids(ocr_ids)

        nodes = question_ids + ocr_ids
        nodes = list(set(nodes))


        first_arr = [q for q in question_ids for _ in range(len(ocr_ids))]
        second_arr = ocr_ids * len(question_ids)
        edges = [first_arr, second_arr]
    
        return torch.tensor(nodes, dtype= torch.int32), torch.tensor(edges, dtype= torch.int32)
    
        
if __name__ == '__main__':
    gay = Collator(None)
    question_ids = [    0,  4447,  2557, 22462,     2]
    ocr_ids = [    0,  4447,  2557, 22462,     2, 1, 1, 1]
    nodes, edges = gay.make_node_edge_array(question_ids, ocr_ids)
    print(nodes)
    print()
    print(edges)