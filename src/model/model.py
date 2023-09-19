from torch import nn, concat, no_grad, tensor
from src.model.components.vision.encoders import EfficientNetEncoder, ImageEncoderViT
from src.model.components.language.encoders import BARTphoEncoder, ViT5Encoder
from src.model.components.transformer import MultiwayTransformer

class VLMo(nn.Module):
    def __init__(self, freeze= True):
        super().__init__()

        self.image_encoder = ImageEncoderViT()
        self.text_encoder = ViT5Encoder()
        if freeze:
            self.image_encoder.freeze()
            self.text_encoder.freeze()

        self.transformer = MultiwayTransformer()

        self.classifier = nn.Linear(768, self.text_encoder.tokenizer.vocab_size)

    def forward(self, img, text):
        img_feature = self.image_encoder(img)
        text_feature = self.text_encoder(text)
        
        idx = self.transformer.vlffn_start_layer_index
        for blk in self.transformer.blocks[:idx]:
            img_feature = blk(img_feature, modality_type= 'image')
            text_feature = blk(text_feature, modality_type= 'text')

        x = concat([text_feature, img_feature], 1)
        # vt_feature = self.transformer(vt_feature)

        for i, blk in enumerate(self.transformer.blocks[idx:]):
            x = blk(x)
        return self.classifier(x)
    
    def predict(self, img, text, max_length, eos_token_id, pad_token_id):
        self.eval()
        device = img.device
        batch_len = len(img)

        with no_grad():
            sent_len = 0

            translated_sent = [[pad_token_id] * batch_len]

            img_feature = self.image_encoder(img)

            img_feature = self.transformer(img_feature, modality_type= 'image')

            while sent_len <= max_length and not all(any(tensor(translated_sent).T == eos_token_id, dim= 1)):
                inp = self.text_encoder.tokenizer.batch_decode(translated_sent)
                text_feature = self.transformer()