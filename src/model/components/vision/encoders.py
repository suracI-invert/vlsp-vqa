import os
import numpy as np 
import pandas as pd 
import torch
import timm 
from PIL import Image

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

from sklearn.metrics import accuracy_score, f1_score


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
    def __init__(self, pretrained_image_name: str = "microsoft/beit-base-patch16-224"):
        super(ImageEncoderViT, self).__init__()
        self.image_encoder = AutoModel.from_pretrained(pretrained_image_name)
        #self.preprocessor = AutoFeatureExtractor.from_pretrained(pretrained_image_name)
    
    def forward(self, pixel_values):
        """
        - input: image
        - output shape: (batch_size, sequence_length, hidden_size) [1, sequence_length, 768]
        """
        encoded_image = self.image_encoder(pixel_values=pixel_values, return_dict=True)
        return encoded_image['last_hidden_state']


class EfficientNetEncoder(nn.Module):
    def __init__(self, pretrained_image_name: str = "efficientnet_b0"):
        super(EfficientNetEncoder, self).__init__()
        model = timm.create_model(pretrained_image_name, pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, images):
        """
        - input: image
        - output shape: (batch_size, feature_map_size, hidden_size) [1, feature_map_size, 1280]
        """
        images = torch.stack(images)
        batch_size, c, h, w = images.shape

        x_resized_1 = images.view(batch_size , c, h, w)
        fmap = self.model(x_resized_1)
        
        batch_size, dim, h, w = fmap.shape
        fmap = fmap.view(batch_size, h * w, dim)
        #print(fmap.shape)

        return fmap