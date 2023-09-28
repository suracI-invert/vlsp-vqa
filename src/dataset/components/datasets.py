from torch.utils.data import Dataset
from PIL import Image
from os.path import join
from json import load
from torchvision.transforms.functional import pil_to_tensor


class VQADataset(Dataset):
    def __init__(self, root_dir: str, map_file: str, data_dir: str, labels2id, max_length: int,
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
        
        self.labels2id = labels2id
        

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_id = str(self.annotations[index]['image_id'])
        question = self.annotations[index]['question']
        answer = self.annotations[index]['answer']
        label_id = self.labels2id.get(answer)
        img_path = self.images[img_id]
        img = Image.open(join(self.root_dir, self.data_dir, img_path))

        if self.processor:
            img = self.processor.preprocess_images(img)['pixel_values'][0]
            # print(img)
            # print(img.shape)
        elif self.transforms:
            img = self.transforms(img)
        else:
            img = img.resize((32, 32))
            img = pil_to_tensor(img)
        
        if self.tokenizer:
            question_tokenized = self.tokenizer.encode_plus(question, return_tensors = "pt", padding = 'max_length', max_length= self.max_length) 
            answer_tokenized = self.tokenizer(answer, return_tensors = "pt", padding = 'max_length', max_length= self.max_length) 
            
            question_tokenized['input_ids'] = question_tokenized['input_ids'].squeeze(dim= 0)
            question_tokenized['attention_mask'] = question_tokenized['attention_mask'].squeeze(dim= 0)
            answer_tokenized['input_ids'] = answer_tokenized['input_ids'].squeeze(dim= 0)
            answer_tokenized['attention_mask'] = answer_tokenized['attention_mask'].squeeze(dim= 0)

        return {
            'img_fname': img_path,
            'img': img,
            'question': question,
            'tokenized_question': question_tokenized,
            'answer': answer,
            'tokenized_answer': answer_tokenized,
            'label': label_id
        }
        

        