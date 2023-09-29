from typing import Any, Dict, List, Tuple, Union

import torch
from torch import nn, optim
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
from torch.optim.optimizer import Optimizer

from torchmetrics import MaxMetric, MeanMetric
from src.utils.metrics import BLEU_CIDEr

from src.model.model import VLMo
from src.utils.translate import translate

class VQALitModule(LightningModule):
    """
        Currently doing multilabel classification with a mapping label to id. No gen yet
    """
    def __init__(self,
                net: nn.Module,
                tokenizer,
                optimizer: optim.Optimizer,
                lr_scheduler: optim.lr_scheduler,
                optimizer_params: Dict[str, Any] = {},
                scheduler_params: Dict[str, Any] = {},
                max_len: int = 64,
                learning_rate: float = 0.001,
                monitor_metric: str = 'val/loss',
                interval: str = 'epoch',
                frequency: int = 3,
            ):
        """
        wow
        """
        super().__init__()
        self.save_hyperparameters(logger= False, ignore= ['net', 'tokenizer'])

        self.net = net
        for p in self.net.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform(p)
        self.tokenizer = tokenizer
    

        self.val_score = BLEU_CIDEr()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_bleu_best = MaxMetric()
        self.val_cider_best = MaxMetric()

    def forward(self, text, img, tgt):
        return self.net(text, img, tgt) 

    def model_step(self, batch):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of tokenized_question, img, tokenized_answer.

        :return: A tuple containing (in order):
            - Loss.
            - Logits.
        """
        img = batch['img']
        text = batch['tokenized_question']
        tgt = batch['tokenized_answer']
        output = self.forward(text, img, tgt)

        # loss.requires_grad = True #??? why does loss lost grad
        return output['loss'], output['logits']


    def training_step(self, batch, batch_idx):
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of tokenized_question, img, tokenized_answer.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, logits = self.model_step(batch)
        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss
   

    def validation_step(self, batch, batch_idx):
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of tokenized_question, img, tokenized_answer.
        :param batch_idx: The index of the current batch.
        """
        loss, logits = self.model_step(batch)
        # preds = self.net.text_encoder.tokenizer.batch_decode(torch.argmax(logits, -1))
        targets = batch['answer']

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)

        preds_text_id = translate(
            self.net, batch['img'], 
            batch['tokenized_question'], 
            self.tokenizer.pad_token_id, # ViT5 no bos_token, cant add due to embedding size limit TODO: stop this hack
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.hparams.max_len,
        )

        preds_text = self.tokenizer.batch_decode(preds_text_id)
        
        self.val_score(preds_text, targets)


    def on_validation_epoch_end(self) -> None:
            "Lightning hook that is called when a validation epoch ends."

            score = self.val_score.compute()
            bleu = score['BLEU']
            cider = score['CIDEr']

            self.val_bleu_best(bleu)
            self.val_cider_best(cider)

            # log `val_bleu_x_best` as a value through `.compute()` method, instead of as a metric object
            # otherwise metric would be reset by lightning after each epoch
            self.log("val/bleu", bleu, sync_dist=True, prog_bar=True)
            self.log("val/cider", cider, sync_dist=True, prog_bar=True)
            self.log("val/bleu_best", self.val_bleu_best.compute(), sync_dist=True, prog_bar=True)
            self.log("val/cider_best", self.val_cider_best.compute(), sync_dist=True, prog_bar=True)
    
    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        norm = grad_norm(self.net, norm_type= 2)
        self.log_dict(norm)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params= self.parameters(), lr= self.hparams.learning_rate, **self.hparams.optimizer_params)
        if self.hparams.lr_scheduler is not None:
            scheduler = self.hparams.lr_scheduler(optimizer= optimizer, **self.hparams.scheduler_params)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': self.hparams.monitor_metric,
                    'interval': self.hparams.interval,
                    'frequency': self.hparams.frequency
                }
            }
        return {'optimizer': optimizer}
    