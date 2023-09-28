import pytest
from src.model.lit import VQALitModule, BaselineLitModule
from src.model.model import VLMo, Baseline

from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from src.dataset.datamodule import VQADataModule
from src.dataset.components.DataAugmentation import ImageAugmentation

from src.model.components.vision.encoders import ImageProcessorViT

import torch
from torch import set_float32_matmul_precision, rand
from torchvision import transforms

from lightning.pytorch import loggers
from lightning import Trainer

if __name__ == '__main__':
    # set_float32_matmul_precision('medium')

    MAX_LEN = 258
    LABEL_SPACE = 27896

    # scheduler_params = {
    #     'mode': 'min',
    #     'factor': 0.1,
    #     'patience': 3,
    #     'threshold': 1e-4,
    #     'threshold_mode': 'rel',
    #     'cooldown': 0,
    #     'min_lr': 0,
    #     'eps': 1e-8,
    #     'verbose': True,
    # }

    # dm = VQADataModule(
    #     './data', 
    #     'training-images', 
    #     'vlsp2023_train_data.json', 
    #     'dev-images', 
    #     'vlsp2023_dev_data.json', 
    #     transforms= ImageAugmentation(), 
    #     batch_size= 2,
    #     max_length= MAX_LEN,
    #     # train_val_split= (28000, 2833),
    #     tokenizer= AutoTokenizer.from_pretrained('vinai/phobert-base'),
    #     processor= ImageProcessorViT()
    # )

    # dm.setup()

    # net = VLMo(LABEL_SPACE, MAX_LEN)

    # model = BaselineLitModule(
    #     net, torch.optim.AdamW, torch.optim.lr_scheduler.ReduceLROnPlateau,
    #     learning_rate= 0.01,
    #     scheduler_params= scheduler_params,
    #     interval= 'epoch',
    #     mapfile= dm.labels2id
    # )

    # print(net.text_encoder)

    # sample = next(iter(dm.train_dataloader()))

    # # print(sample['question'])
    # # print(sample['answer'])
    # # print(sample['img'].shape)
    # # # print(sample['tokenized_question'])
    # # print(sample['tokenized_question']['input_ids'].shape)
    # # # print(sample['tokenized_answer'])
    # # print('=' * 50)

    # # print(model.model_step(sample))
    # # print(net.image_encoder(sample['img']))
    # # print(sample)
    # loss, loggits = model.model_step(sample)
    # print(loss)
    # print(loggits.shape)
    
