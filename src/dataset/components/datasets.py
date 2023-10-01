from torch.utils.data import Dataset
from PIL import Image
from os.path import join
from json import load
from torchvision.transforms.functional import pil_to_tensor

import torch


class VQADataset(Dataset):
    def __init__(self, root_dir: str, map_file: str, data_dir: str, max_length: int,
                tokenizer= None, processor= None, transforms= None
            ):
        self.root_dir = root_dir
        self.map_file = map_file
        self.data_dir = data_dir
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length

        with open(join(self.root_dir, self.map_file), 'r', encoding= 'utf8') as f:
            data = load(f)

            self.images = data['images']
            self.annotations= list(data['annotations'].values())
        
        

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_id = str(self.annotations[index]['image_id'])
        question = self.annotations[index]['question']
        answer = self.annotations[index]['answer']
        img_path = self.images[img_id]
        img = Image.open(join(self.root_dir, self.data_dir, img_path))

        # TODO: Move to another seperate func
        if self.processor:
            img = self.processor.preprocess_images(img)['pixel_values'][0]
            # print(img)
            # print(img.shape)
        elif self.transforms:
            img = self.transforms(img)
        else:
            img = img.resize((32, 32))
            img = pil_to_tensor(img)
        
        # TODO: Move to another seperate func
        if self.tokenizer:
            # bos_token_id = torch.tensor([self.tokenizer.pad_token_id]) # Hacky way using pad instead of bos for ViT5 TODO: fix this
            bos_token_id = torch.tensor([self.tokenizer.bos_token_id]) # Only with bartpho
            # question_tokenized = self.tokenizer(question, return_tensors = "pt", padding = 'longest') 
            question_tokenized = self.tokenizer(question, return_tensors = "pt", padding = 'max_length', max_length= self.max_length) 
            answer_tokenized = self.tokenizer(answer, return_tensors = "pt", padding = 'max_length', max_length= self.max_length) 
            
            question_tokenized['input_ids'] = question_tokenized['input_ids'].squeeze(dim= 0)
            question_tokenized['attention_mask'] = question_tokenized['attention_mask'].squeeze(dim= 0)
            answer_tokenized['input_ids'] = answer_tokenized['input_ids'].squeeze(dim= 0)
            answer_tokenized['input_ids'] = torch.cat([bos_token_id, answer_tokenized['input_ids']], dim= -1)
            answer_tokenized['attention_mask'] = answer_tokenized['attention_mask'].squeeze(dim= 0)
            answer_tokenized['attention_mask'] = torch.cat([torch.tensor([1]), answer_tokenized['attention_mask']], -1)
        
        return {
            'img_fname': img_path,
            'img': img,
            'question': question,
            'tokenized_question': question_tokenized,
            'answer': answer,
            'tokenized_answer': answer_tokenized,
        }
        

        