import os
import numpy as np 
import pandas as pd 
import torch
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


class ImageEncoderViT(nn.Module):
    def __init__(self, pretrained_image_name: str = "microsoft/beit-base-patch16-224"):
        super(ImageEncoderViT, self).__init__()
        self.image_encoder = AutoModel.from_pretrained(pretrained_image_name)
        self.preprocessor = AutoFeatureExtractor.from_pretrained(pretrained_image_name)

    def preprocess_images(self, images):
        processed_images = self.preprocessor(
            images=images,
            return_tensors="pt",
        )
        return {
            "pixel_values": processed_images['pixel_values'].squeeze(),
        }
    
    def forward(self, images):
        """
        - input: image
        - output shape: (batch_size, sequence_length, hidden_size) [1, sequence_length, 768]
        """
        processed_image = self.preprocess_images(images)
        encoded_image = self.image_encoder(pixel_values=processed_image['pixel_values'], return_dict = True)
        return encoded_image['last_hidden_state']