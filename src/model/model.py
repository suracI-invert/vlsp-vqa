from torch import nn, concat
from src.model.components.vision.encoders import EfficientNetEncoder, ImageEncoderViT
from src.model.components.language.encoders import BARTphoEncoder, ViT5Encoder
from src.model.components.attentions import Block

class VLMo(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_encoder = ImageEncoderViT()
        self.text_encoder = ViT5Encoder()

        self.transformer = Block(
            768, 6
        )

    def forward(self, img, text):
        img_feature = self.image_encoder(img)
        text_feature = self.text_encoder(text)
        
        img_feature = self.transformer(img_feature, modality_type= 'image')
        text_feature = self.transformer(text_feature, modality_type= 'text')

        vt_feature = concat([text_feature, img_feature], 1)

        return self.transformer(vt_feature)