from torch import nn
from src.model.components.vision.encoders import EfficientNetEncoder, ImageEncoderViT
from src.model.components.language.encoders import BARTphoEncoder
from src.model.components.attentions import Block

class VLMo(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_encoder = ImageEncoderViT()
        self.text_encoder = BARTphoEncoder()

        self.transformer = Block(
            786, 6
        )