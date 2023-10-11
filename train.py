import argparse
import json

from src.model.lit import VQALitModule, VQAv2LitModule
from src.model.model import CompoundToken, CompoundTokenOCR, GAv2, GAvOCR, VLMo, Baseline, GA
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required= False)
    parser.add_argument('--worker', required= False)
    args = parser.parse_args()

    set_float32_matmul_precision('medium')
    MAX_LEN = 256

    D_MODEL = 512
    WARMUP_STEPS = 8000

    DATA_DIR = './data'
    if args.dir is None:
        print('No data directory set, default to: ./data')
    else:
        print('data dir set to: ' + args.dir)
        DATA_DIR = args.dir

    num_workers = int(args.worker) if args.worker is not None else 2

    llm_url = 'vinai/bartpho-syllable-base'
    tokenizer = AutoTokenizer.from_pretrained(llm_url)
    # tokenizer = get_tokenizer(llm_url)

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

    scheduler_params = {
        'dim_embed': D_MODEL,
        'warmup_steps': WARMUP_STEPS,
    }
    # scheduler_params = {
    #     'T_max': 20,
    #     'eta_min': 1e-5,
    # }

    es_cb = EarlyStopping('val/loss', min_delta= 0.00001, patience= 10)
    lr_monitor = LearningRateMonitor('step', True)
    ckpt_cb = ModelCheckpoint(
        dirpath= './weights',
        filename= 'vqa_final_compound_{epoch:02d}_{step:02d}_cider',
        monitor= 'val/cider',
        save_on_train_epoch_end= True,
        save_top_k= 1,
    )
    profiler = AdvancedProfiler('./log/profiler', filename= 'perf_logs')

    dm = VQADataModule(
        root_dir= DATA_DIR,  
        data_dir= 'training-images', 
        map_file= 'train_data_ocr.json', 
        val_dir= 'dev-images', 
        val_map_file= 'dev_data_ocr.json', 
        test_dir= 'test-images',
        test_map_file= 'test_data_ocr.json',
        transforms= ImageAugmentationCNN(), 
        batch_size= 32,
        max_length= MAX_LEN,
        num_workers= num_workers,
        tokenizer= tokenizer,
        collate_fn= Collator(tokenizer),
        # processor= ImageProcessorViT()
    )
    dm.setup()
    # net = CompoundToken(
    #     tokenizer.vocab_size, 
    #     tokenizer.pad_token_id, 
    #     num_encoder_layers= 3, 
    #     num_decoder_layers= 3,
    #     d_model= D_MODEL, 
    #     freeze= 'both', 
    #     act= nn.GELU,
    #     hidden_dim= 2048,
    #     dropout_encoder= 0.5
    # )
    net = GA(
        tokenizer.vocab_size, 
        tokenizer.pad_token_id, 
        num_encoder_layers= 3, 
        num_decoder_layers= 3,
        d_model= D_MODEL, 
        freeze= 'both', 
        act= nn.GELU,
        hidden_dim= 2048,
        dropout_encoder= 0.5,
        dropout_decoder= 0.5
    )
    # net = GAv2(
    #     tokenizer.vocab_size, 
    #     tokenizer.pad_token_id, 
    #     num_encoder_layers= 3, 
    #     num_decoder_layers= 3,
    #     d_model= D_MODEL, 
    #     freeze= 'text', 
    #     act= nn.GELU,
    #     hidden_dim= 2048,
    #     dropout_encoder= 0.3
    # )
    # net = GAvOCR(
    #     tokenizer.vocab_size, 
    #     tokenizer.pad_token_id, 
    #     num_encoder_layers= 3, 
    #     num_decoder_layers= 3,
    #     d_model= 768, 
    #     freeze= 'text', 
    #     act= nn.GELU,
    #     hidden_dim= 2048,
    #     dropout_encoder= 0.3  
    # )

    model = VQALitModule(
        net, tokenizer, 
        torch.optim.RAdam, WarmupScheduler, 
        scheduler_params= scheduler_params, learning_rate= 1e-9,
        interval= 'step'
    )

    # model = VQAv2LitModule(
    #     net, tokenizer, 
    #     torch.optim.RAdam, WarmupScheduler, 
    #     scheduler_params= scheduler_params, learning_rate= 1e-9,
    #     interval= 'step'
    # )

    tb_logger = loggers.TensorBoardLogger(
        save_dir= './log',
    )

    trainer = Trainer(
        accelerator= 'gpu',
        precision= '32',
        max_time= '00:14:00:00',
        max_epochs= 30,
        benchmark= True,
        logger= tb_logger,
        log_every_n_steps= 5,
        num_sanity_val_steps= 2,
        check_val_every_n_epoch= 1,
        callbacks= [lr_monitor, ckpt_cb, es_cb],
        # profiler= profiler,
        gradient_clip_val= 0.5,
        # limit_train_batches= 0.1,
        # limit_val_batches= 0.1,
        # detect_anomaly= True,
        # fast_dev_run= True
    )

    trainer.fit(model, datamodule= dm)

    print('Start predicting public')

    res = trainer.predict(dataloaders= dm.val_dataloader(), ckpt_path= 'best')

    predictions = {}
    for d in res:
        for i, v in d.items():
            predictions[i] = v

    with open('./data/public_results.json', 'w', encoding= 'utf8') as f:
        json.dump(predictions, f, indent= 4)
    print('Done')

    print('Start predicting private')

    res = trainer.predict(dataloaders= dm.test_dataloader(), ckpt_path= 'best')

    predictions = {}
    for d in res:
        for i, v in d.items():
            predictions[i] = v

    with open('./data/private_results.json', 'w', encoding= 'utf8') as f:
        json.dump(predictions, f, indent= 4)
    print('Done')
