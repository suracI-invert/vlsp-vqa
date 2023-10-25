#TODO: Add collator with masking
# TODO: ocr -> graph should be here

from typing import Any
import torch
import numpy as np 

def join_token(tokens):
    return ' '.join(tokens)

class Collator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch) -> Any:
        idx = []
        img = []
        questions = []
        answers = []
        # ocr = []
        # graphs = []
        pad_token_id = torch.tensor(self.tokenizer.pad_token_id)
        for sample in batch:
            idx.append(sample['id'])
            img.append(sample['img'])
            questions.append(sample['question'])
            answers.append(sample['answer'])
            # ocr.append(sample['ocr'])
            # ocr_tokenized = self.tokenizer(sample['ocr'], return_tensors= 'pt', padding= 'longest') if len(sample['ocr']) > 0 else {'input_ids': None, 'attention_mask': None}
            # ocr_token = join_token(sample['ocr'])
            # ocr.append(ocr_token)
        src = self.tokenizer(questions, return_tensors= 'pt', padding= 'longest')
        src['input_ids'] = src['input_ids'][:, 1:]
        src['attention_mask'] = src['attention_mask'][:, 1:]
        tgt = self.tokenizer(answers, return_tensors= 'pt', padding= 'longest')
        tgt_label = tgt['input_ids'][:, 1:].clone()
        for i in range(len(tgt['input_ids'])):
            tgt['input_ids'][i][tgt['input_ids'][i] == self.tokenizer.eos_token_id] = self.tokenizer.pad_token_id
        tgt['input_ids'] = tgt['input_ids'][:, :-1]
        tgt['attention_mask'] = (tgt['input_ids'] != self.tokenizer.pad_token_id).long()
        # ocr = self.tokenizer(ocr, return_tensors= 'pt', padding= 'longest')
        # node_ids = []
        # edge_ids = []
        # for i in range(len(batch)):
        #     ocr_id = ocr[i]['input_ids']
        #     question_id = src['input_ids'][i]
        #     node, edge = self.make_node_edge_array(question_id, ocr_id)
            # graphs.append({
            #     'node_id': node
            # })
        # ocr_tokens = self.tokenizer(ocr, return_tensors= 'pt', padding= 'longest')
        return {
            'id': idx,
            'img': torch.stack(img),
            'src': src,
            'tgt': tgt,
            'tgt_label': tgt_label,
            'answer': answers,
            # 'ocr': ocr_tokens
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

    # Dummy Data
    question_ids = torch.tensor([[0, 4447, 2557, 22462, 2, 1, 1, 1], [0, 447, 23, 262, 23, 2, 1, 1], [0, 44, 57, 462, 233, 11, 111, 2], [111, 122, 1, 1, 1, 1, 1, 1], [332, 344, 1, 1, 1, 1, 1, 1]])
    ocr_ids = torch.tensor([[0, 47, 257, 262, 2, 1, 1, 1, 1], [0, 47, 5337, 262, 233, 133, 2, 1, 1], [0, 4, 2574, 26222, 233, 133, 121, 2, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 2, 1, 1, 1, 1, 1]])

    # Expected
    node_ids = torch.tensor([4447, 2557, 22462, # 0, 1, 2
                447, 23, 262, 23, # 3, 4, 5, 6
                44, 57, 462, 233, 11, 111, # 7, 8, 9, 10, 11, 12 
                111, 122, # 13, 14
                332, 334, # 15, 16 #question batch_size = 5
                47, 257, 262, # 17, 18, 19 
                47, 5337, 262, 233, 133, #  20, 21, 22, 23, 24
                4, 2574, 26222, 233, 133, 121]) # 25, 26, 27, 28, 29, 30 #ocr batch_size = 5
    edge_index = torch.tensor([
        [0, 0, 0, 
         1, 1, 1, 
         2, 2, 2, # batch 1
         3, 3, 3, 3, 3, 
         4, 4, 4, 4, 4, 
         5, 5, 5, 5, 5, 
         6, 6, 6, 6, 6, #batch 2
         7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8,
         9, 9, 9, 9, 9, 9,
         10, 10, 10, 10, 10, 10, 
         11, 11, 11, 11, 11, 11, 
         12, 12, 12, 12, 12, 12, # batch 3
         ],
        [17, 18, 19, #batch 1
         17, 18, 19, 
         17, 18, 19, 
         20, 21, 22, 23, 24, #batch 2
         20, 21, 22, 23, 24,
         20, 21, 22, 23, 24,
         20, 21, 22, 23, 24,
         20, 21, 22, 23, 24,
         25, 26, 27, 28, 29, 30, #batch 3
         25, 26, 27, 28, 29, 30,
         25, 26, 27, 28, 29, 30,
         25, 26, 27, 28, 29, 30,
         25, 26, 27, 28, 29, 30,
         25, 26, 27, 28, 29, 30,
        ]
    ])


    # Function to test
    nodes, edges, idx = gay.make_node_edge_array(question_ids, ocr_ids)
    print(nodes)
    print()
    print(edges)

    assert nodes.shape == node_ids.shape, f"node_ids different length, should be {node_ids.shape}"
    assert edges.shape == edge_index.shape, f"edge_index different length, should be {node_ids.shape}"
    assert nodes == node_ids, f'node_ids wrong element'
    assert edges == edge_index, f'edge_index wrong element'
    assert node_ids[idx[0][0]:idx[0][1]] == torch.tensor([17, 18, 19])
    assert node_ids[idx[1][0]:idx[0][1]] == torch.tensor([20, 21, 22, 23, 24])
    assert node_ids[idx[2][0]:idx[0][1]] == torch.tensor([25, 26, 27, 28, 29, 30])
    assert idx[3] == (-1, -1)
    assert idx[4] == (-1, -1)

# def make_node_edge_array(question_ids, ocr_ids):
#     # Flatten the input arrays and calculate batch sizes
#     question_ids_flat = [id for batch in question_ids for id in batch]
#     ocr_ids_flat = [id for batch in ocr_ids for id in batch]
    
#     question_batch_size = len(question_ids)
#     ocr_batch_size = len(ocr_ids)
    
#     # Combine flattened arrays to create node_ids
#     node_ids = question_ids_flat + ocr_ids_flat

#     print(node_ids)
#     print()
    
#     question_batch_size = []
#     ocr_batch_size = []

#     for batch in question_ids:
#         question_batch_size.append(len(batch)) 

#     for batch in ocr_ids:
#         ocr_batch_size.append(len(batch)) 

#     print(question_batch_size)
#     print(ocr_batch_size)

#     question_index = []
#     q_batch_no = 1  # Initialize batch number
#     q_index = 0

#     for batch in question_ids:
#         for element in batch:
#             question_index.append([q_index, q_batch_no])
#             q_index += 1
#         q_batch_no += 1

#     print(question_index)

#     ocr_index = []
#     o_batch_no = 1  # Initialize batch number
#     o_index = 0 + len(question_index)
#     print(o_index)

#     for batch in ocr_ids:
#         for element in batch:
#             ocr_index.append([o_index, o_batch_no])
#             o_index += 1
#         o_batch_no += 1

#     print(ocr_index)

#     # # Calculate start and end indices for question batches
#     # question_batch_indices = []
#     # start_idx = 0
#     # for batch_size in question_batch_size:
#     #     end_idx = start_idx + batch_size - 1
#     #     question_batch_indices.append((start_idx, end_idx))
#     #     start_idx = end_idx + 1

#     # Calculate start and end indices for ocr batches
#     ocr_batch_indices = []
#     start_idx = 0
#     for batch_size in ocr_batch_size:
#         end_idx = start_idx + batch_size - 1
#         ocr_batch_indices.append((start_idx, end_idx))
#         start_idx = end_idx + 1

#     # Combine question and ocr batch indices into a single tuple
#     node_tuple = (ocr_batch_indices)

#     print("Node tuples (OCR only):", node_tuple)
#     print()

#     first_arr = [] 

#     for entity in question_index:
#         current_batch_no1 = entity[1]
#         len_to_dup = ocr_batch_size[current_batch_no1 - 1]
#         while(len_to_dup):
#             first_arr.append(entity[0])
#             len_to_dup -= 1

#     print(first_arr)

#     second_arr = [] 

#     for entity in ocr_index:
#         current_batch_no2 = entity[1]
#         len_to_dup = question_batch_size[current_batch_no2 - 1]
        
#         tmp = []
#         for i in ocr_index:
#             if i[1] == current_batch_no2:
#                 tmp.append(i[0])
        
#         print(tmp)
        
#         while(len_to_dup):
#             for item in tmp:
#                 second_arr.append(item)
#             len_to_dup -= 1

#         # DANG BI SAI DOAN NAY, DUP BI THUA -------------------------------------

#     # second_arr = ocr_ids * len(question_ids)
    

#     print(second_arr)

# Input: 
# question_ids = [[4447, 2557, 22462], [447, 23, 262, 23], [44, 57, 462, 233, 11, 111]]
# ocr_ids = [[47, 257, 262], [47, 5337, 262, 233, 133], [4, 2574, 26222, 233, 133, 121]]
# make_node_edge_array(question_ids, ocr_ids)

# Output:
# [4447, 2557, 22462, 447, 23, 262, 23, 44, 57, 462, 233, 11, 111, 47, 257, 262, 47, 5337, 262, 233, 133, 4, 2574, 26222, 233, 133, 121]

# [3, 4, 6]
# [3, 5, 6]
# [[0, 1], [1, 1], [2, 1], [3, 2], [4, 2], [5, 2], [6, 2], [7, 3], [8, 3], [9, 3], [10, 3], [11, 3], [12, 3]]
# 13
# [[13, 1], [14, 1], [15, 1], [16, 2], [17, 2], [18, 2], [19, 2], [20, 2], [21, 3], [22, 3], [23, 3], [24, 3], [25, 3], [26, 3]]
# Node tuples (OCR only): [(0, 2), (3, 7), (8, 13)]

# [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12]
# [13, 14, 15]
# [13, 14, 15]
# [13, 14, 15]
# [16, 17, 18, 19, 20]
# [16, 17, 18, 19, 20]
# [16, 17, 18, 19, 20]
# [16, 17, 18, 19, 20]
# [16, 17, 18, 19, 20]
# [21, 22, 23, 24, 25, 26]
# [21, 22, 23, 24, 25, 26]
# [21, 22, 23, 24, 25, 26]
# [21, 22, 23, 24, 25, 26]
# [21, 22, 23, 24, 25, 26]
# [21, 22, 23, 24, 25, 26]
# [13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 13, 14, 15, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26, 21, 22, 23, 24, 25, 26]
    