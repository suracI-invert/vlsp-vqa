from src.model.lit import VQALitModule, BaselineLitModule
from src.model.model import VLMo, Baseline, GA
from src.model.components.vision.encoders import ImageProcessorViT
from src.dataset.datamodule import VQADataModule
from src.dataset.components.DataAugmentation import ImageAugmentation

from src.utils.tokenizer import get_tokenizer

from transformers import AutoTokenizer

import torch
from torch import set_float32_matmul_precision, rand
from torchvision import transforms

from lightning.pytorch import loggers
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.profilers import AdvancedProfiler
from lightning import Trainer

if __name__ == '__main__':
    set_float32_matmul_precision('medium')
    MAX_LEN = 128

    llm_url = 'VietAI/vit5-base'
    tokenizer = AutoTokenizer.from_pretrained(llm_url)
    # tokenizer = get_tokenizer(llm_url)

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

    es_cb = EarlyStopping('val/loss', min_delta= 0.0001, patience= 2)
    lr_monitor = LearningRateMonitor('step', True)
    ckpt_cb = ModelCheckpoint(
        dirpath= './weights',
        filename= 'vqa_{epoch:02d}_{val_bleu:0.2f}',
        monitor= 'val/bleu_mean',
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
        transforms= ImageAugmentation(), 
        batch_size= 16,
        max_length= MAX_LEN,
        # train_val_split= (28000, 2833),
        tokenizer= tokenizer,
        processor= ImageProcessorViT()
    )
    dm.setup()
    net = GA(tokenizer.vocab_size, tokenizer.pad_token_id, num_encoder_layers= 12)

    model = VQALitModule(
        net, tokenizer, torch.optim.AdamW, torch.optim.lr_scheduler.ReduceLROnPlateau,
        learning_rate= 0.001,
        scheduler_params= scheduler_params,
        interval= 'epoch',
    )

    tb_logger = loggers.TensorBoardLogger(
        save_dir= './log',
    )

    trainer = Trainer(
        accelerator= 'gpu',
        precision= '16-mixed',
        max_time= '00:08:00:00',
        max_epochs= 20,
        benchmark= True,
        logger= tb_logger,
        log_every_n_steps= 5,
        num_sanity_val_steps= 2,
        check_val_every_n_epoch= 1,
        callbacks= [es_cb, lr_monitor, ckpt_cb],
        profiler= profiler
    )

    trainer.fit(model, datamodule= dm)
