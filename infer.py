import argparse
import json

from src.model.lit import VQALitModule
from src.model.model import GAv2, VLMo, Baseline, GA
from src.model.components.vision.encoders import ImageProcessorViT
from src.dataset.datamodule import VQADataModule
from src.dataset.components.DataAugmentation import ImageAugmentation, ImageAugmentationCNN, ImageAugmentationCNNStripped, ImageAugmentationStripped
from src.dataset.components.collator import Collator

from src.utils.tokenizer import get_tokenizer
from src.utils.optim import WarmupScheduler

from transformers import AutoTokenizer

import torch
from torch import nn
from torch import set_float32_matmul_precision, rand
from torchvision import transforms

from lightning.pytorch import loggers
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.profilers import AdvancedProfiler
from lightning import Trainer

import warnings
warnings.filterwarnings("ignore", "Detected call of", UserWarning)
"""
    Disable warning (scheduler.step before optimizer.step). 
    This is due to amp implementation, scaler will sometimes skip optimizer.step hench scheduler.step call raises warning. (this should not be concerned when trained with f32 precision)
    Should not affect too much
    Link: https://github.com/Lightning-AI/lightning/issues/5558
"""

if __name__ == '__main__':
    set_float32_matmul_precision('medium')
    CKPT_PATH = './weights/vqa_v1_epoch=14_step=14460.ckpt'


    MAX_LEN = 256

    D_MODEL = 512


    DATA_DIR = './data'
    num_workers = 6

    llm_url = 'vinai/bartpho-syllable-base'
    tokenizer = AutoTokenizer.from_pretrained(llm_url)



    dm = VQADataModule(
        DATA_DIR, 
        'training-images', 
        'vlsp2023_train_data.json', 
        'dev-images', 
        'vlsp2023_dev_data.json', 
        'test-images',
        'vlsp2023_test_data.json',
        transforms= ImageAugmentationCNN(), 
        batch_size= 32,
        max_length= MAX_LEN,
        num_workers= num_workers,
        tokenizer= tokenizer,
        collate_fn= Collator(tokenizer),
        # processor= ImageProcessorViT()
    )
    dm.setup()
    net = GA(
        tokenizer.vocab_size, 
        tokenizer.bos_token_id, 
        num_encoder_layers= 3, 
        num_decoder_layers= 3,
        d_model= D_MODEL, 
        freeze= 'text', 
        act= nn.GELU,
        hidden_dim= 2048,
        dropout_encoder= 0.3
    )

    # print(net)

    model = VQALitModule.load_from_checkpoint(net= net, tokenizer= tokenizer, checkpoint_path= CKPT_PATH)
    # model = VQALitModule(net, tokenizer)

    # print(model)

    trainer = Trainer(accelerator= 'gpu', logger= False)

    res = trainer.predict(model, dm.test_dataloader())
    print(res)
    predictions = {}
    for d in res:
        for i, v in d.items():
            predictions[i] = v

    with open('./data/private_results.json', 'w', encoding= 'utf8') as f:
        json.dump(predictions, f, indent= 4)
    print('Done')

