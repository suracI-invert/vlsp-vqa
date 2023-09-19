from typing import Any, Dict, List, Tuple, Union

from torch import nn, optim
from lightning import LightningModule
from torchmetrics.text.bleu import BLEUScore
from torchmetrics import MaxMetric, MeanMetric

from src.model.model import VLMo

class VQALitModule(LightningModule):
    def __init__(self,
                net: nn.Module,
                # tokenizer,
                optimizer: optim.Optimizer,
                lr_scheduler: optim.lr_scheduler,
                optimizer_params: Dict[str, Any] = {},
                scheduler_params: Dict[str, Any] = {},
                learning_rate: float = 0.001,
                monitor_metric: str = 'val_loss',
                interval: str = 'epoch',
                frequency: int = 3,
            ):
        """
        wow
        """
        super().__init__()
        self.save_hyperparameters(logger= False, ignore= ['net'])

        self.net = net
        self.criterion = None #TODO add loss compute
    
        self.val_bleu_1 = BLEUScore(1)
        self.val_bleu_2 = BLEUScore(2)
        self.val_bleu_3 = BLEUScore(3)
        self.val_bleu_4 = BLEUScore(4)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_bleu_est = MaxMetric()
        self.val_bleu_2_best = MaxMetric()
        self.val_bleu_3_best = MaxMetric()
        self.val_bleu_4_best = MaxMetric()

    def forward(self, img, text):
        return self.net(img, text)

    def model_step(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass