import os
import numpy as np 
import pandas as pd 
import torch
import timm 
from PIL import Image
import torchvision
import torchvision.ops as ops


import torch.nn as nn
from transformers import (
    # Preprocessing / Common
    AutoTokenizer, AutoFeatureExtractor,
    # Text & Image Models (Now, image transformers like ViTModel, DeiTModel, BEiT can also be loaded using AutoModel)
    AutoModel,            
    # Training / Evaluation
    TrainingArguments, Trainer,
    # Misc
    logging
)
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
from torchvision import transforms

class ImageProcessorViT:
    def __init__(self, pretrained_model_name_or_path=None):
        if pretrained_model_name_or_path is None:
            # Use a default pre-trained Vision Transformer model
            pretrained_model_name_or_path = "microsoft/beit-base-patch16-224"  # Change to the desired ViT model

        self.preprocessor = AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path)

    def preprocess_images(self, images):
        processed_images = self.preprocessor(
            images=images,
            return_tensors="pt",
        )
        return {
            "pixel_values": processed_images['pixel_values'],
        }
    
class ImageEncoderViT(nn.Module):
    def __init__(self, pretrained_image_name: str = "microsoft/beit-base-patch16-224", dim= 768):
        super(ImageEncoderViT, self).__init__()
        self.image_encoder = AutoModel.from_pretrained(pretrained_image_name)
        #self.preprocessor = AutoFeatureExtractor.from_pretrained(pretrained_image_name)
        self.proj = nn.Linear(768, dim)
    
    def forward(self, pixel_values):
        """
        - input: image
        - output shape: (batch_size, sequence_length, hidden_size) [1, sequence_length, 768]
        """
        # processed_image = self.preprocess_images(images)
        encoded_image = self.image_encoder(pixel_values, return_dict = True)
        return self.proj(encoded_image['last_hidden_state'])
    
    def freeze(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = False


class EfficientNetEncoder(nn.Module):
    def __init__(self, pretrained_image_name: str = "efficientnet_b0", dim= 768):
        super(EfficientNetEncoder, self).__init__()
        model = timm.create_model(pretrained_image_name, pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(1280, dim)
    
    def forward(self, images):
        """
        - input: image
        - output shape: (batch_size, feature_map_size, hidden_size) [1, feature_map_size, 1280]
        """
        # images = torch.stack(images)
        batch_size, c, h, w = images.shape

        x_resized_1 = images.view(batch_size , c, h, w)
        fmap = self.model(x_resized_1)
        
        batch_size, dim, h, w = fmap.shape
        fmap = fmap.view(batch_size, h * w, dim)
        #print(fmap.shape)

        return self.proj(fmap)

class ImageEncoderRCNN(nn.Module):
    def __init__(self, pretrained_image_name: str = "COCO-Detection/faster_rcnn_R_50_FPN_3x", dim= 768):
        super(ImageEncoderRCNN, self).__init__()
        #model = model_zoo.get("COCO-Detection/faster_rcnn_R_50_FPN_3x")
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.proj = nn.Linear(1280, dim)
    
    def forward(self, images):
        """
        - input: image
        - output shape: (batch_size, feature_map_size, hidden_size) [1, feature_map_size, 768]
        """
        #images = torch.stack(images)
        batch_size, c, h, w = images.shape
        #print(images.shape)

        x_resized_1 = images.view(batch_size , c, h, w)
        fmap = self.model(x_resized_1)
        print(fmap[0]['boxes'])
        feature_list = []
        #print(type(feature_list))
        #print(images[0].shape)
        for i in range(batch_size):
            #print(type(feature_list))
            rois = fmap[i]['boxes']
            #feature_maps = fmap[i]['features']
            rois = [roi.unsqueeze(0) for roi in rois]  # RoIs
            
            print(i, rois[0])

            image_shapes = [(224,224)]
            roi_features = self.model.roi_heads.box_roi_pool(self.model.backbone(images[i].unsqueeze(0)), 
                                                             rois, [torch.tensor(image_shapes)])  
            #print(roi_features.shape)
        
            batch_size_features, dim, h, w = roi_features.shape
            reshaped_roi_features = roi_features.view(h * w, dim * batch_size_features)
            print(reshaped_roi_features.shape)
            
            proj = nn.Linear(dim * batch_size_features, 768)
            ln_feature = proj(reshaped_roi_features)
            print((ln_feature))

            feature_list.append(ln_feature)     
            #outputs = torch.stack(feature_list)
            #print(outputs.shape)
            
        outputs = torch.stack(feature_list)
        return outputs
