from typing import Any, Dict, List, Tuple, Union

import torch
from torch import nn, optim
from lightning import LightningModule
from torchmetrics.text.bleu import BLEUScore
from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics import MaxMetric, MeanMetric

from src.model.model import VLMo

class VQALitModule(LightningModule):
    """
        Currently doing multilabel classification with a mapping label to id. No gen yet
    """
    def __init__(self,
                net: nn.Module,
                # tokenizer,
                optimizer: optim.Optimizer,
                lr_scheduler: optim.lr_scheduler,
                mapfile= None,
                optimizer_params: Dict[str, Any] = {},
                scheduler_params: Dict[str, Any] = {},
                learning_rate: float = 0.001,
                monitor_metric: str = 'val/loss',
                interval: str = 'epoch',
                frequency: int = 3,
            ):
        """
        wow
        """
        super().__init__()
        self.save_hyperparameters(logger= False, ignore= ['net'])

        self.net = net
        self.criterion = nn.CrossEntropyLoss()

        self.acc = MulticlassAccuracy(27896, average= 'micro')
    
        self.val_bleu_1 = BLEUScore(1)
        self.val_bleu_2 = BLEUScore(2)
        self.val_bleu_3 = BLEUScore(3)
        self.val_bleu_4 = BLEUScore(4)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.acc_best = MaxMetric()

        self.val_bleu_1_best = MaxMetric()
        self.val_bleu_2_best = MaxMetric()
        self.val_bleu_3_best = MaxMetric()
        self.val_bleu_4_best = MaxMetric()

    def forward(self, img, text):
        return self.net(img, text) 

    def model_step(self, batch):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        img = batch['img']
        text = batch['tokenized_question']
        label = batch['label'].cuda()
        logits = self.forward(img, text)

        # hacky bit: turn all Long tensor to Float tensor, might need to revisit this again
        loss = self.criterion(
             logits,
             label,
            )
        # loss.requires_grad = True #??? why does loss lost grad
        return loss, logits


    def training_step(self, batch, batch_idx):
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
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

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, logits = self.model_step(batch)
        # preds = self.net.text_encoder.tokenizer.batch_decode(torch.argmax(logits, -1))
        preds = torch.argmax(torch.nn.functional.softmax(logits, dim= -1), dim= -1).cpu().detach()
        labels = batch['label'].cpu().detach()
        targets = batch['answer']

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)

        self.acc(preds, labels)
        self.log('val/acc', self.acc, on_step= False, on_epoch= True, prog_bar= True)

        if self.hparams.mapfile:
            preds_text = [[k for k, v in self.hparams.mapfile.items() if v == i][0] for i in preds]
            self.val_bleu_1(preds_text, targets)
            self.val_bleu_2(preds_text, targets)
            self.val_bleu_3(preds_text, targets)
            self.val_bleu_4(preds_text, targets)
            self.log("val/bleu_1", self.val_bleu_1, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/bleu_2", self.val_bleu_2, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/bleu_3", self.val_bleu_3, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/bleu_4", self.val_bleu_4, on_step=False, on_epoch=True, prog_bar=True)


    def on_validation_epoch_end(self) -> None:
            "Lightning hook that is called when a validation epoch ends."
            bleu_1 = self.val_bleu_1.compute() # get current val bleu
            bleu_2 = self.val_bleu_2.compute() 
            bleu_3 = self.val_bleu_3.compute() 
            bleu_4 = self.val_bleu_4.compute() 

            acc = self.acc.compute()

            self.val_bleu_1_best(bleu_1)  # update best so far val bleu
            self.val_bleu_2_best(bleu_2)
            self.val_bleu_3_best(bleu_3)
            self.val_bleu_4_best(bleu_4)

            self.acc_best(acc)
            self.log('val/acc_best', self.acc_best.compute(), sync_dist= True, prog_bar= True)

            # log `val_bleu_x_best` as a value through `.compute()` method, instead of as a metric object
            # otherwise metric would be reset by lightning after each epoch
            self.log("val/bleu_1_best", self.val_bleu_1_best.compute(), sync_dist=True, prog_bar=True)
            self.log("val/bleu_2_best", self.val_bleu_2_best.compute(), sync_dist=True, prog_bar=True)
            self.log("val/bleu_3_best", self.val_bleu_3_best.compute(), sync_dist=True, prog_bar=True)
            self.log("val/bleu_4_best", self.val_bleu_4_best.compute(), sync_dist=True, prog_bar=True)
        
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
    

class BaselineLitModule(LightningModule):
    """
        Currently doing multilabel classification with a mapping label to id. No gen yet
    """
    def __init__(self,
                net: nn.Module,
                # tokenizer,
                optimizer: optim.Optimizer,
                lr_scheduler: optim.lr_scheduler,
                mapfile= None,
                optimizer_params: Dict[str, Any] = {},
                scheduler_params: Dict[str, Any] = {},
                learning_rate: float = 0.001,
                monitor_metric: str = 'val/loss',
                interval: str = 'epoch',
                frequency: int = 3,
            ):
        """
        wow
        """
        super().__init__()
        self.save_hyperparameters(logger= False, ignore= ['net'])

        self.net = net

        self.acc = MulticlassAccuracy(27896, average= 'micro')
    
        self.val_bleu_1 = BLEUScore(1)
        self.val_bleu_2 = BLEUScore(2)
        self.val_bleu_3 = BLEUScore(3)
        self.val_bleu_4 = BLEUScore(4)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.acc_best = MaxMetric()

        self.val_bleu_1_best = MaxMetric()
        self.val_bleu_2_best = MaxMetric()
        self.val_bleu_3_best = MaxMetric()
        self.val_bleu_4_best = MaxMetric()

    def forward(self, input_ids, pixel_values, attention_mask= None, token_type_ids= None, labels= None):
        return self.net(input_ids, pixel_values, attention_mask, token_type_ids, labels) 

    def model_step(self, batch):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        img = batch['img']
        input_ids = batch['tokenized_question']['input_ids']
        attention_mask = batch['tokenized_question']['attention_mask']
        label = batch['label'].cuda()
        output = self.forward(input_ids, img, attention_mask, labels= label)
        loss = output['loss']
        logits = output['logits']
        # loss.requires_grad = True #??? why does loss lost grad
        return loss, logits


    def training_step(self, batch, batch_idx):
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
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

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, logits = self.model_step(batch)
        # preds = self.net.text_encoder.tokenizer.batch_decode(torch.argmax(logits, -1))
        preds = torch.argmax(torch.nn.functional.softmax(logits, dim= -1), dim= -1).cpu().detach()
        labels = batch['label'].cpu().detach()
        targets = batch['answer']

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)

        self.acc(preds, labels)
        self.log('val/acc', self.acc, on_step= False, on_epoch= True, prog_bar= True)

        if self.hparams.mapfile:
            preds_text = [[k for k, v in self.hparams.mapfile.items() if v == i][0] for i in preds]
            self.val_bleu_1(preds_text, targets)
            self.val_bleu_2(preds_text, targets)
            self.val_bleu_3(preds_text, targets)
            self.val_bleu_4(preds_text, targets)
            self.log("val/bleu_1", self.val_bleu_1, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/bleu_2", self.val_bleu_2, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/bleu_3", self.val_bleu_3, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/bleu_4", self.val_bleu_4, on_step=False, on_epoch=True, prog_bar=True)


    def on_validation_epoch_end(self) -> None:
            "Lightning hook that is called when a validation epoch ends."
            bleu_1 = self.val_bleu_1.compute() # get current val bleu
            bleu_2 = self.val_bleu_2.compute() 
            bleu_3 = self.val_bleu_3.compute() 
            bleu_4 = self.val_bleu_4.compute() 

            acc = self.acc.compute()

            self.val_bleu_1_best(bleu_1)  # update best so far val bleu
            self.val_bleu_2_best(bleu_2)
            self.val_bleu_3_best(bleu_3)
            self.val_bleu_4_best(bleu_4)

            self.acc_best(acc)
            self.log('val/acc_best', self.acc_best.compute(), sync_dist= True, prog_bar= True)

            # log `val_bleu_x_best` as a value through `.compute()` method, instead of as a metric object
            # otherwise metric would be reset by lightning after each epoch
            self.log("val/bleu_1_best", self.val_bleu_1_best.compute(), sync_dist=True, prog_bar=True)
            self.log("val/bleu_2_best", self.val_bleu_2_best.compute(), sync_dist=True, prog_bar=True)
            self.log("val/bleu_3_best", self.val_bleu_3_best.compute(), sync_dist=True, prog_bar=True)
            self.log("val/bleu_4_best", self.val_bleu_4_best.compute(), sync_dist=True, prog_bar=True)
        
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