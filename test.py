import pytest
from src.model.lit import VQALitModule, VQAv2LitModule
from src.model.model import CompoundToken, CompoundTokenOCR, GAv2, GAvOCR, VLMo, Baseline, GA

from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from src.dataset.datamodule import VQADataModule
from src.dataset.components.DataAugmentation import ImageAugmentation, ImageAugmentationCNN, ImageAugmentationCNNStripped
from src.dataset.components.collator import Collator
from src.model.components.vision.encoders import ImageProcessorViT, ImageEncoderRCNN

from torchmetrics.text.bleu import BLEUScore

import torch
from torch import set_float32_matmul_precision, rand
from torchvision import transforms
import torchvision

from src.utils.translate import translate

from lightning.pytorch import loggers
from lightning import Trainer

from time import time

if __name__ == '__main__':
    # set_float32_matmul_precision('medium')

    MAX_LEN = 128
    LABEL_SPACE = 27896

    # tokenizer = AutoTokenizer.from_pretrained('VietAI/vit5-base')
    tokenizer = AutoTokenizer.from_pretrained('vinai/bartpho-syllable-base')
    # if tokenizer.bos_token is None:
    #     tokenizer.add_special_tokens({'bos_token': '<s>'})

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
        root_dir= './data', 
        data_dir= 'training-images', 
        map_file= 'train_data_ocr.json', 
        val_dir= 'dev-images', 
        val_map_file= 'dev_data_ocr.json', 
        test_dir= 'test-images',
        test_map_file= 'test_data_ocr.json',
        transforms= ImageAugmentationCNNStripped(), 
        batch_size= 16,
        max_length= MAX_LEN,
        # train_val_split= (28000, 2833),
        tokenizer= tokenizer,
        # processor= ImageProcessorViT(),
        collate_fn= Collator(tokenizer)
    )
    dm.prepare_data()
    dm.setup()

    net = GA(tokenizer.vocab_size, tokenizer.pad_token_id, return_loss= True, d_model= 768, freeze= 'both')
    # image_encoder.eval()
    # model = VQAv2LitModule(
    #     net, tokenizer, torch.optim.AdamW, torch.optim.lr_scheduler.ReduceLROnPlateau,
    #     learning_rate= 0.001,
    #     scheduler_params= scheduler_params,
    # )

    print(net.image_encoder)

    # image_encoder = ImageEncoderRCNN()
    # image_encoder = torchvision.models.detection.faster_rcnn.fasterrcnn_resnet50_fpn(weights= torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # image_encoder.eval()
    # count = 0
    # for p in net.parameters():
    #     if p.requires_grad:
    #         count += p.numel()
    # print(count)
    
    # print(model.current_epoch)
    # print(image_encoder.roi_heads.box_head.fc7)
    # sample = next(iter(dm.val_dataloader()))
    # for sample in dm.val_dataloader():
    #     out = image_encoder(sample['img'])
    #     # print(out)
    #     print(out.shape)
    # features = []
    # for i in range(sample['img'].shape[0]):
    #     img, _ = image_encoder.transform(sample['img'][i].unsqueeze(0))
    #     img_features = image_encoder.backbone(img.tensors)
    #     # print(len(img_features))
    #     # print(img_features['0'].shape)
    #     proposal, _ = image_encoder.rpn(img, img_features)
    #     box_feature = image_encoder.roi_heads.box_roi_pool(img_features, proposal, img.image_sizes)
    #     # print(box_feature.shape)
    #     box_feature = image_encoder.roi_heads.box_head(box_feature)
    #     # print(.shape)
    #     features.append(box_feature)
    # last_state = torch.stack(features)
    # print(last_state.shape)
    # features = []
    # def save_feature(mod, inp, outp):
    #     features.append(outp.data)
    # image_encoder.roi_heads.box_head.fc7.register_forward_hook(save_feature)
    # out = image_encoder(sample['img'])
    # print(torch.stack(features).shape)
    # img, _ = image_encoder.transform([im for im in sample['img']])
    # img_features = image_encoder.backbone(img.tensors)
    # print(len(img_features))
    # print(img.tensors.shape)
    # print(img_features.keys())
    # print(img_features['pool'].shape)
    # keys = list(img_features.keys())
    # new_features = {}
    # for k in keys:
    #     new_features[k] = img_features[k][0].unsqueeze(0)
    # print(new_features['pool'].shape)
    # proposal, _ = image_encoder.rpn(img, img_features)
    # print(torch.stack(proposal).shape)
    # print(len(proposal))
    # box_feature = image_encoder.roi_heads.box_roi_pool(img_features, proposal, img.image_sizes)
    # box_feature = image_encoder.roi_heads.box_head(box_feature)
    # print(box_feature.shape)
    # box_features = []
    # for i in range(0, box_feature.shape[0], 1000):
        # box_features.append(box_feature[i:i+1000, :])
    # print(.shape)
    # features.append(box_feature)
    # last_statev2 = torch.stack(box_features)
    # print(last_statev2.shape)
    # print((last_statev2 == last_state))

    # print(image_encoder.backbone(sample['img'])['0'].shape)

    # print(net(sample['src'], sample['img'], sample['ocr'], sample['tgt'], sample['tgt_label']))
    # print(model.step())
    # print(sample['ocr'])
    # print(net.text_encoder.model.get_input_embeddings())

    # ids = torch.tensor([[tokenizer.bos_token_id, 12, 5545, 123, tokenizer.eos_token_id, 333, 31313], [tokenizer.bos_token_id, 12, 5545, 123, tokenizer.eos_token_id, 333, 31313]])
    # for i in range(ids.shape[0]):
    #     ids[i,ids[i].tolist().index(tokenizer.eos_token_id) + 1:] = tokenizer.pad_token_id
    # print(ids)

    # print(net.decoder)
    # print(next(iter(net.text_encoder.model.get_decoder().parameters())).requires_grad)
    # output = net.text_encoder.model.get_encoder()(input_ids = sample['src']['input_ids'], attention_mask= sample['src']['attention_mask'])
    # print(output.last_hidden_state.shape)
    # img_output = net.image_encoder(sample['img'])
    # print(img_output.shape) 
    # print(sample['src']['attention_mask'])
    # attention_mask = torch.cat([torch.tensor([[1] * img_output.shape[-2]]), sample['src']['attention_mask']], dim= -1)
    # encoder_output = torch.cat([img_output, output.last_hidden_state], dim= -2)
    # print(encoder_output.shape)
    # decoder_output = net.text_encoder.model.get_decoder()(
    #     input_ids= sample['tgt']['input_ids'], 
    #     attention_mask= sample['tgt']['attention_mask'],
    #     encoder_hidden_states= encoder_output,
    #     encoder_attention_mask= attention_mask,
    #     return_dict= True
    # )
    # print(sample['ocr'])
    # print(sample['question'])
    # print(net.encoder_forward(sample['src'], sample['img']))

    # print(decoder_output.last_hidden_state.shape)
    # print(tokenizer.batch_decode(ids))
    # text = net.text_encoder(sample['src'])
    # img = net.image_encoder(sample['img'])
    # print(text.keys())
    # print(img.keys())
    # print(tokenizer.bos_token)
    # print(tokenizer.eos_token)
    # print(tokenizer.pad_token_id)
    # print(tokenizer.mask_token)
    # print(sample)
    # print(sample['src'])
    # print(sample['tgt'])
    # print(sample['answer'])
    # print(sample['img'].shape)
    # print(sample['img'])
    # print(sample['tokenized_answer'])
    # print(tokenizer.decode(sample['src']['input_ids'][0], False))
    # print(tokenizer.decode(sample['tgt']['input_ids'][0], False))
    # print(tokenizer.decode(sample['tgt_label'][0], False))

    # # print(sample['tokenized_question']['input_ids'].shape)
    # for sample in dm.test_dataloader():
    #     print(sample['fname'])
    #     print(net(sample['src'], sample['img'], sample['tgt'], sample['tgt_label']))
    # print(sample['tokenized`_answer']['input_ids'].view(-1).shape)
    # # print('=' * 50)

    # print(tokenizer.batch_decode(translate(net, sample['img'], sample['src'], tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, 16, 'beam', 4, 0, 1)))
    # print(tokenizer.batch_decode(translate(net, sample['img'], sample['src'], tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, 16, 'greedy', 4, 0, 1)))
    # # print(model.model_step(sample))
    # # print(net.image_encoder(sample['img']))
    # # print(sample)
    # loss, logits = model.model_step(sample)
    # print(loss)
    # print(logits.shape)
    
