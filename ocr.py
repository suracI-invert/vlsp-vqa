from src.model.components.ocr import OCR
import json
import os
from tqdm import tqdm

# with open('./data/dev_data_ocr.json', 'r', encoding= 'utf8') as f:
#     data = json.load(f)
# keys = list(data['annotations'].keys())
# print(len(keys))
# entry = data['annotations'][keys[50]]
# print(entry)
# image_path = data['images'][str(entry['image_id'])]
# print(image_path)

ocr = OCR()
# with open(os.path.join('./data', 'vlsp2023_dev_data.json'), 'r', encoding= 'utf8') as f:
#     data = json.load(f)
# keys = list(data['annotations'].keys())
# entry = data['annotations'][keys[136]]
# print(ocr(os.path.join('./data/', 'dev-images', data['images'][str(entry['image_id'])])))

def get_tokens(path, map_file, data_dir, predictor, new_file):
    with open(os.path.join(path, map_file), 'r', encoding= 'utf8') as f:
        data = json.load(f)
    keys = list(data['annotations'].keys())
    for i in tqdm(range(len(keys)), desc= f'Reading {map_file}'):
        entry = data['annotations'][keys[i]]
        image_path = data['images'][str(entry['image_id'])]
        tokens = predictor(os.path.join(path, data_dir, image_path))
        tokens = ' '.join(tokens) if len(tokens) > 0 else ''
        entry['ocr'] = tokens
    with open(os.path.join(path, new_file), 'w', encoding= 'utf8') as f:
        json.dump(data, f, indent= 4)

get_tokens('./data', 'vlsp2023_dev_data.json', 'dev-images', ocr, 'dev_data_ocr.json')
get_tokens('./data', 'vlsp2023_test_data.json', 'test-images', ocr, 'test_data_ocr.json')
get_tokens('./data', 'vlsp2023_train_data.json', 'training-images', ocr, 'train_data_ocr.json')
    