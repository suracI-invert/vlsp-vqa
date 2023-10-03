import argparse

from src.model.lit import VQALitModule
from src.model.model import VLMo, Baseline, GA
from src.model.components.vision.encoders import ImageProcessorViT
from src.dataset.datamodule import VQADataModule
from src.dataset.components.DataAugmentation import ImageAugmentation, ImageAugmentationCNN
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

    D_MODEL = 768
    WARMUP_STEPS = 10000

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
    #     'gamma': 0.5
    # }


    es_cb = EarlyStopping('val/loss', min_delta= 0.0000001, patience= 3)
    lr_monitor = LearningRateMonitor('step', True)
    ckpt_cb = ModelCheckpoint(
        dirpath= './weights',
        filename= 'vqa_v2_{epoch:02d}_{step:02d}',
        monitor= 'val/loss',
        save_on_train_epoch_end= True,
        save_top_k= 1,
    )
    profiler = AdvancedProfiler('./log/profiler', filename= 'perf_logs')

    dm = VQADataModule(
        DATA_DIR, 
        'training-images', 
        'vlsp2023_train_data.json', 
        'dev-images', 
        'vlsp2023_dev_data.json', 
        transforms= ImageAugmentation(), 
        batch_size= 16,
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
        num_encoder_layers= 6, 
        num_decoder_layers= 6,
        d_model= D_MODEL, 
        freeze= True, 
        act= nn.GELU(),
        hidden_dim= 2048,
        dropout_encoder= 0.3
    )

    model = VQALitModule(
        net, tokenizer, torch.optim.RAdam, WarmupScheduler, scheduler_params= scheduler_params, learning_rate= 1e-9
    )

    tb_logger = loggers.TensorBoardLogger(
        save_dir= './log',
    )

    trainer = Trainer(
        accelerator= 'gpu',
        precision= '32',
        # max_time= '00:08:00:00',
        max_epochs= 40,
        benchmark= True,
        logger= tb_logger,
        log_every_n_steps= 5,
        num_sanity_val_steps= 2,
        check_val_every_n_epoch= 1,
        callbacks= [lr_monitor, ckpt_cb, es_cb],
        # profiler= profiler,
        gradient_clip_val= 0.5,
        # fast_dev_run= True,
        # limit_train_batches= 0.1,
        # limit_val_batches= 0.1,
        # detect_anomaly= True
    )

    trainer.fit(model, datamodule= dm)
