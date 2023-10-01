import pytest
from src.model.lit import VQALitModule
from src.model.model import VLMo, Baseline, GA

from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from src.dataset.datamodule import VQADataModule
from src.dataset.components.DataAugmentation import ImageAugmentation, ImageAugmentationCNN

from src.model.components.vision.encoders import ImageProcessorViT

from torchmetrics.text.bleu import BLEUScore

import torch
from torch import set_float32_matmul_precision, rand
from torchvision import transforms

from src.utils.translate import translate

from lightning.pytorch import loggers
from lightning import Trainer

if __name__ == '__main__':
    # set_float32_matmul_precision('medium')

    MAX_LEN = 128
    LABEL_SPACE = 27896

    # tokenizer = AutoTokenizer.from_pretrained('VietAI/vit5-base')
    tokenizer = AutoTokenizer.from_pretrained('vinai/bartpho-syllable-base')
    # if tokenizer.bos_token is None:
    #     tokenizer.add_special_tokens({'bos_token': '<s>'})

    scheduler_params = {
        'mode': 'min',
        'factor': 0.1,
        'patience': 3,
        'threshold': 1e-4,
        'threshold_mode': 'rel',
        'cooldown': 0,
        'min_lr': 0,
        'eps': 1e-8,
        'verbose': True,
    }

    dm = VQADataModule(
        './data', 
        'training-images', 
        'vlsp2023_train_data.json', 
        'dev-images', 
        'vlsp2023_dev_data.json', 
        transforms= ImageAugmentationCNN(), 
        batch_size= 2,
        max_length= MAX_LEN,
        # train_val_split= (28000, 2833),
        tokenizer= tokenizer,
        # processor= ImageProcessorViT()
    )

    dm.setup()

    net = GA(tokenizer.vocab_size, tokenizer.pad_token_id, return_loss= True, freeze= False)
    model = VQALitModule(
        net, tokenizer, torch.optim.AdamW, torch.optim.lr_scheduler.ReduceLROnPlateau,
        learning_rate= 0.001,
        scheduler_params= scheduler_params,
    )

    # count = 0
    # for p in net.parameters():
    #     if p.requires_grad:
    #         count += p.numel()
    # print(count)
    
    # print(model.global_step)

    sample = next(iter(dm.train_dataloader()))

    # ids = torch.tensor([[tokenizer.bos_token_id, 12, 5545, 123, tokenizer.eos_token_id, 333, 31313], [tokenizer.bos_token_id, 12, 5545, 123, tokenizer.eos_token_id, 333, 31313]])
    # for i in range(ids.shape[0]):
    #     ids[i,ids[i].tolist().index(tokenizer.eos_token_id) + 1:] = tokenizer.pad_token_id
    # print(ids)


    # print(tokenizer.batch_decode(ids))

    # print(tokenizer.bos_token)
    # print(tokenizer.eos_token)
    # print(tokenizer.pad_token_id)
    # print(tokenizer.mask_token)
    # # print(sample['question'])
    # # print(sample['answer'])
    # print(sample['img'].shape)
    # print(sample['img'])
    # print(sample['tokenized_answer'])
    # print(tokenizer.decode(sample['tokenized_answer']['input_ids'][0], False))
    # # print(sample['tokenized_question']['input_ids'].shape)
    print(net(sample['tokenized_question'], sample['img'], sample['tokenized_answer']).loss + 1e-10)
    # print(sample['tokenized_answer']['input_ids'].view(-1).shape)
    # # print('=' * 50)

    # print(tokenizer.batch_decode(translate(net, sample['img'], sample['tokenized_question'], tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, 16)))

    # # print(model.model_step(sample))
    # # print(net.image_encoder(sample['img']))
    # # print(sample)
    # loss, loggits = model.model_step(sample)
    # print(loss)
    # print(loggits.shape)
    
