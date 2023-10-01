from src.model.lit import VQALitModule
from src.model.model import VLMo, Baseline, GA
from src.model.components.vision.encoders import ImageProcessorViT
from src.dataset.datamodule import VQADataModule
from src.dataset.components.DataAugmentation import ImageAugmentation, ImageAugmentationCNN

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
    MAX_LEN = 256

    D_MODEL = 256
    WARMUP_STEPS = 1500

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

    es_cb = EarlyStopping('val/loss', min_delta= 0.0000001, patience= 2)
    lr_monitor = LearningRateMonitor('step', True)
    ckpt_cb = ModelCheckpoint(
        dirpath= './weights',
        filename= 'vqa_{epoch:02d}_{step:02d}',
        monitor= 'val/cider',
        save_on_train_epoch_end= True,
        save_top_k= 1,
    )
    profiler = AdvancedProfiler('./log/profiler', filename= 'perf_logs')

    dm = VQADataModule(
        './data', 
        'training-images', 
        'vlsp2023_train_data.json', 
        'dev-images', 
        'vlsp2023_dev_data.json', 
        transforms= ImageAugmentationCNN(), 
        batch_size= 16,
        max_length= MAX_LEN,
        num_workers= 6,
        tokenizer= tokenizer,
        # processor= ImageProcessorViT()
    )
    dm.setup()
    net = GA(
        tokenizer.vocab_size, 
        tokenizer.bos_token_id, 
        num_encoder_layers= 6, 
        d_model= D_MODEL, 
        freeze= True, 
        act= nn.GELU(),
        hidden_dim= 1024,
        dropout_encoder= 0.3
    )

    model = VQALitModule(
        net, tokenizer, torch.optim.RAdam, WarmupScheduler,
        learning_rate= 1.0e-6,
        scheduler_params= scheduler_params
    )

    tb_logger = loggers.TensorBoardLogger(
        save_dir= './log',
    )

    trainer = Trainer(
        accelerator= 'gpu',
        precision= '16-mixed',
        # max_time= '00:08:00:00',
        max_epochs= 20,
        benchmark= True,
        logger= tb_logger,
        log_every_n_steps= 5,
        num_sanity_val_steps= 2,
        check_val_every_n_epoch= 1,
        callbacks= [lr_monitor, ckpt_cb],
        # profiler= profiler,
        gradient_clip_val= 0.5,
        # fast_dev_run= True,
        # limit_train_batches= 0.1,
        # limit_val_batches= 0.1,
        # detect_anomaly= True
    )

    trainer.fit(model, datamodule= dm)
