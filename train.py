from src.model.lit import VQALitModule, BaselineLitModule
from src.model.model import VLMo, Baseline
from src.model.components.vision.encoders import ImageProcessorViT
from src.dataset.datamodule import VQADataModule
from src.dataset.components.DataAugmentation import ImageAugmentation

from transformers import AutoTokenizer

import torch
from torch import set_float32_matmul_precision, rand
from torchvision import transforms

from lightning.pytorch import loggers
from lightning import Trainer

if __name__ == '__main__':
    # set_float32_matmul_precision('medium')
    MAX_LEN = 256
    LABEL_SPACE = 27896

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
        transforms= ImageAugmentation(), 
        batch_size= 16,
        max_length= MAX_LEN,
        # train_val_split= (28000, 2833),
        tokenizer= AutoTokenizer.from_pretrained('vinai/phobert-base'),
        processor= ImageProcessorViT()
    )
    dm.setup()
    net = Baseline(LABEL_SPACE)

    model = BaselineLitModule(
        net, torch.optim.AdamW, torch.optim.lr_scheduler.ReduceLROnPlateau,
        learning_rate= 0.001,
        scheduler_params= scheduler_params,
        interval= 'epoch',
        mapfile= dm.labels2id
    )
    # dm = VQADataModule(
    #     './data', 
    #     'training-images', 
    #     'vlsp2023_train_data.json', 
    #     'dev-images', 
    #     'vlsp2023_dev_data.json', 
    #     transforms= ImageAugmentation(), 
    #     batch_size= 16,
    #     max_length= MAX_LEN,
    #     # train_val_split= (28000, 2833),
    #     tokenizer= AutoTokenizer.from_pretrained('VietAI/vit5-base'),
    #     processor= ImageProcessorViT()
    # )
    # dm.setup()
    # net = VLMo(LABEL_SPACE, MAX_LEN)

    # model = VQALitModule(
    #     net, torch.optim.AdamW, torch.optim.lr_scheduler.ReduceLROnPlateau,
    #     learning_rate= 0.01,
    #     scheduler_params= scheduler_params,
    #     interval= 'epoch',
    #     mapfile= dm.labels2id
    # )

    tb_logger = loggers.TensorBoardLogger(
        save_dir= './log',
    )

    trainer = Trainer(
        accelerator= 'gpu',
        # precision= '16-mixed',
        max_time= '00:08:00:00',
        max_epochs= 10,
        benchmark= True,
        logger= tb_logger,
        log_every_n_steps= 5,
        check_val_every_n_epoch= 1,
    )

    trainer.fit(model, datamodule= dm)
