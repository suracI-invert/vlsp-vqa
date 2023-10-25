from torch.utils.data import Dataset
from PIL import Image
from os.path import join
from json import load
from torchvision.transforms.functional import pil_to_tensor
from src.model.components.ocr import OCR

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
        # self.ocr_predictor = OCR()

        with open(join(self.root_dir, self.map_file), 'r', encoding= 'utf8') as f:
            data = load(f)

            self.images = data['images']
            self.idx = list(data['annotations'].keys())
            self.annotations= data['annotations']

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        idx = self.idx[index]
        img_id = str(self.annotations[idx]['image_id'])
        question = self.annotations[idx]['question']
        answer = self.annotations[idx]['answer']
        # ocr_tokens = self.annotations[idx]['ocr']
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
        
        # TODO: seperate vietocr and easyocr for initialization
        # ocr_tokens = self.ocr_predictor(join(self.root_dir, self.data_dir, img_path))
        
        # TODO: Move to another seperate func

        return {
            'id': idx,
            'img': img,
            'question': question,
            'answer': answer,
            # 'ocr': ocr_tokens
        }
    
        