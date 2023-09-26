from torch import nn, concat, no_grad, tensor
from src.model.components.vision.encoders import EfficientNetEncoder, ImageEncoderViT
from src.model.components.language.encoders import BARTphoEncoder, ViT5Encoder
from src.model.components.transformer import MultiwayTransformer
from src.model.components.pooler import Pooler

from transformers import AutoModel


class VLMo(nn.Module):
    def __init__(self, vocab_size, max_text_len, freeze= True):
        super().__init__()

        self.image_encoder = ImageEncoderViT()
        self.text_encoder = ViT5Encoder()
        if freeze:
            self.image_encoder.freeze()
            self.text_encoder.freeze()

        self.transformer = MultiwayTransformer(max_text_len= max_text_len)
        self.pooler = Pooler(self.transformer.num_features)

        self.classifier = nn.Linear(self.transformer.num_features, vocab_size)

    def forward(self, img, text):
        img_feature = self.image_encoder(img)
        text_feature = self.text_encoder(text)

        x = concat([text_feature, img_feature], dim= 1)
        # print(text_feature.shape)
        # print(img_feature.shape)
        # print(x.shape)
        # print('^' * 50)
        # idx = self.transformer.vlffn_start_layer_index
        # for blk in self.transformer.blocks[:idx]:
        #     img_feature = blk(img_feature, modality_type= 'image')
        #     text_feature = blk(text_feature, modality_type= 'text')

        # vt_feature = self.transformer(vt_feature)

        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x)
        x = self.transformer.norm(x)
        return self.classifier(self.pooler(x))
    
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

class Baseline(nn.Module):
    def __init__(self, num_labels, hidden= 768, dropout= 0.2, pretrained_text= 'vinai/phobert-base', pretrained_img= 'microsoft/beit-base-patch16-224'):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(pretrained_text)
        self.img_encoder = AutoModel.from_pretrained(pretrained_img)

        self.fusion = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size + self.img_encoder.config.hidden_size, hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(hidden, num_labels)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, pixel_values, attention_mask= None, token_type_ids= None, labels= None):
        text = self.text_encoder(input_ids= input_ids, attention_mask= attention_mask, token_type_ids= token_type_ids, return_dict= True)
        img = self.img_encoder(pixel_values= pixel_values, return_dict= True)

        fused = self.fusion(concat([text['pooler_output'], img['pooler_output']], dim= 1))
        logits = self.classifier(fused).cuda()
        # print(logits.device)

        out = {'logits': logits}
        if labels is not None:
            loss = self.criterion(logits, labels.cuda())
            out['loss'] = loss
        
        return out