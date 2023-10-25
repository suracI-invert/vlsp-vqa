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

import random

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
                interval: str = 'step',
                frequency: int = 1
            ):
        """
        wow
        """
        super().__init__()
        self.save_hyperparameters(logger= False, ignore= ['net', 'tokenizer'])

        self.net = net

        self.tokenizer = tokenizer
    

        self.val_score = BLEU_CIDEr()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_bleu_best = MaxMetric()
        self.val_cider_best = MaxMetric()

    def forward(self, text, img, tgt, tgt_label):
        return self.net(text, img, tgt, tgt_label) 

    def model_step(self, batch):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of tokenized_question, img, tokenized_answer.

        :return: A tuple containing (in order):
            - Loss.
            - Logits.
        """
        img = batch['img']
        text = batch['src']
        tgt = batch['tgt']
        tgt_label = batch['tgt_label']
        output = self.forward(text, img, tgt, tgt_label)

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
            batch['src'], 
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.hparams.max_len,
            'greedy', 4
        )

        preds_text = self.tokenizer.batch_decode(preds_text_id, skip_special_tokens= True)
        
        self.val_score.update(preds_text, targets)

    def predict_step(self, batch, batch_idx, dataloader_idx= 0):
        idx_list = batch['id']
        preds_text_id = translate(
            self.net, batch['img'], 
            batch['src'], 
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.hparams.max_len,
            'greedy', 4
        )


        preds_text = self.tokenizer.batch_decode(preds_text_id, skip_special_tokens= True)
        preds = {}
        for i in range(len(idx_list)):
            id = idx_list[i]
            pred = preds_text[i]
            preds[id] = pred 
        return preds


    def on_validation_epoch_end(self) -> None:
            "Lightning hook that is called when a validation epoch ends."

            score = self.val_score.compute()

            p, l = random.sample(self.val_score.val, 1)[0]
            self.logger.experiment.add_text(f'Target', l, self.current_epoch)
            self.logger.experiment.add_text(f'Prediction', p, self.current_epoch)

            self.val_score.reset()
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
                    'interval': self.hparams.interval,
                    'frequency': self.hparams.frequency
                }
            }
        return {'optimizer': optimizer}
    
class VQAv2LitModule(LightningModule):
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
                interval: str = 'step',
                frequency: int = 1
            ):
        """
        wow
        """
        super().__init__()
        self.save_hyperparameters(logger= False, ignore= ['net', 'tokenizer'])

        self.net = net

        self.tokenizer = tokenizer
    

        self.val_score = BLEU_CIDEr()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_bleu_best = MaxMetric()
        self.val_cider_best = MaxMetric()

    def forward(self, ocr, text, img, tgt, tgt_label):
        return self.net(text, img, ocr, tgt, tgt_label) 

    def model_step(self, batch):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of tokenized_question, img, tokenized_answer.

        :return: A tuple containing (in order):
            - Loss.
            - Logits.
        """
        ocr = batch['ocr']
        img = batch['img']
        text = batch['src']
        tgt = batch['tgt']
        tgt_label = batch['tgt_label']
        output = self.forward(ocr, text, img, tgt, tgt_label)

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

        preds_text_id = self.greedy(
            self.net, batch['ocr'], batch['img'], 
            batch['src'], 
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.hparams.max_len,
        )

        preds_text = self.tokenizer.batch_decode(preds_text_id, skip_special_tokens= True)
        
        self.val_score.update(preds_text, targets)

    def predict_step(self, batch, batch_idx, dataloader_idx= 0):
        preds_text_id = self.greedy(
            self.net, batch['ocr'], batch['img'], 
            batch['src'], 
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.hparams.max_len,
        )

        preds_text = self.tokenizer.batch_decode(preds_text_id, skip_special_tokens= True)
        return preds_text

    def greedy(self, model, ocr, img, text, bos_token_id, eos_token_id, pad_token_id, max_len):
        device = img.device

        batch_size = img.shape[0]

        src, src_mask = model.encoder_forward(text, img, ocr)

        sent = torch.tensor([[bos_token_id] * batch_size], device= device).T

        for _ in range(max_len):
            output = model.decoder_forward(src, sent, src_attn_mask= src_mask)
            output = torch.nn.functional.softmax(output, dim= -1)

            _, idx = torch.topk(output, 3)

            idx = idx[:, -1, 0].reshape(batch_size, -1)

            sent = torch.cat([sent, idx], 1)

        sent = sent.cpu().detach()
        # for i in range(sent.shape[0]):
        #     idx = (sent[i] == eos_token_id).nonzero().flatten()
        #     if idx.dim != 0:
        #         sent = sent[i, idx[0] + 1:] = pad_token_id
        return sent

    def on_validation_epoch_end(self) -> None:
            "Lightning hook that is called when a validation epoch ends."

            score = self.val_score.compute()

            p, l = random.sample(self.val_score.val, 1)[0]
            self.logger.experiment.add_text(f'Target', l, self.current_epoch)
            self.logger.experiment.add_text(f'Prediction', p, self.current_epoch)

            self.val_score.reset()
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
                    'interval': self.hparams.interval,
                    'frequency': self.hparams.frequency
                }
            }
        return {'optimizer': optimizer}
    