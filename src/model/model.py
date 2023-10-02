from torch import nn, cat, no_grad, tensor, rand, BoolTensor
from src.model.components.vision.encoders import EfficientNetEncoder, ImageEncoderViT
from src.model.components.language.encoders import ViT5Encoder, BARTphoEncoder
from src.model.components.transformer import MultiwayTransformer
from src.model.components.pooler import Pooler
from src.model.components.attentions import GuidedAttention, TransformerDecoderLayer, PositionalEncoding

from src.utils.loss import LabelSmoothingLoss

from transformers import AutoModel

import math

class VLMo(nn.Module):
    def __init__(self, vocab_size, max_text_len, freeze= True):
        super().__init__()

        self.image_encoder = ImageEncoderViT()
        self.text_encoder = BARTphoEncoder()
        if freeze:
            self.image_encoder.freeze()
            self.text_encoder.freeze()

        self.transformer = MultiwayTransformer(max_text_len= max_text_len)
        self.pooler = Pooler(self.transformer.num_features)

        self.classifier = nn.Linear(self.transformer.num_features, vocab_size)

    def forward(self, img, text):
        img_feature = self.image_encoder(img)
        text_feature = self.text_encoder(text)

        x = cat([text_feature, img_feature], dim= 1)
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

        fused = self.fusion(cat([text['pooler_output'], img['pooler_output']], dim= 1))
        logits = self.classifier(fused).cuda()
        # print(logits.device)

        out = {'logits': logits}
        if labels is not None:
            loss = self.criterion(logits, labels.cuda())
            out['loss'] = loss
        
        return out
    

class GA(nn.Module):
    def __init__(self, vocab_size, pad_id, 
                 d_model= 768, nheads_encoder= 8, 
                 nheads_decoder= 8, num_encoder_layers= 6, 
                 num_decoder_layers= 6, hidden_dim= 2048, 
                 dropout_encoder= 0.2, dropout_decoder= 0.2, act= nn.ReLU(), norm_first= False, freeze= True, return_loss= True
                ):
        super().__init__()

        self.image_encoder = ImageEncoderViT(dim= d_model)
        # self.image_encoder = EfficientNetEncoder(dim= d_model)
        # self.text_encoder = ViT5Encoder() # TODO: logic to change between diffenrent encoder
        self.text_encoder = BARTphoEncoder(hidden_dim= d_model)

        self.return_loss = return_loss

        self.d_model = d_model

        if freeze:
            self.image_encoder.freeze()
            self.text_encoder.freeze()

        self.encoder_layers = nn.ModuleList([GuidedAttention(
            dim= d_model,
            nheads=nheads_encoder,  
            dropout=dropout_encoder,  
            hidden_dim= hidden_dim,
            act= act,
            norm_first= norm_first
        ) for _ in range(num_encoder_layers)])

        # self.encoder_fnn = nn.Sequential(
        #     nn.Linear(d_model, hidden_dim),
        #     nn.Dropout(dropout_encoder),
        #     nn.Linear(hidden_dim, d_model),
        # )

        self.encoder_fnn_drop = nn.Dropout(dropout_encoder)
        self.encoder_fnn_norm = nn.LayerNorm(d_model)

        self.encoder_fnn = nn.Linear(d_model, d_model)

        self.decoder = TransformerDecoderLayer(
            vocab_size= vocab_size,
            d_model= d_model,
            nhead=nheads_decoder, 
            dim_feedforward=hidden_dim,  
            num_layers= num_decoder_layers,
            norm_first= norm_first,
            act= act,
            dropout= dropout_decoder,
        )

        self.classifier = nn.Linear(d_model, vocab_size)
        self.criterion = LabelSmoothingLoss(pad_id, 0.1)
        

    def forward(self, text, img, tgt):
        label_ids = tgt['input_ids']
        # text_feature = self.text_encoder(text)
        # img_feature = self.image_encoder(img)

        # # Swap dim 0, dim 1. From (batch_size, seq_length, hidden_dim) to (seq_length, batch_size, hidden_dim)
        # img_feature = img_feature.permute(1, 0, 2) 
        # text_feature = text_feature.permute(1, 0, 2)
        
        # # x.shape = (batch_size, seq_length, hidden_dim)
        # img_feature, text_feature = self.encoder_layers((img_feature, text_feature))

        # src = cat([img_feature, text_feature], dim= 0)

        # # src = self.encoder_fnn_norm(self.encoder_fnn_drop(src + self.encoder_fnn(src)))
        # src = self.encoder_fnn(self.encoder_fnn_drop(src))

        src, mask = self.encoder_forward(text, img)

        # decoder_output = self.decoder(src, tgt['input_ids'], tgt['attention_mask'])
        # decoder_output = self.classifier(decoder_output)

        decoder_output = self.decoder_forward(src, tgt['input_ids'], mask, tgt['attention_mask'])

        if self.return_loss:
            return {
                'loss': self.criterion(decoder_output, label_ids),
                'logits': decoder_output
            }
        return {
            'logits': decoder_output
        }
    
    def encoder_forward(self, text, img):
        text_attn_mask = (text['attention_mask'] == 0)
        text_feature = self.text_encoder(text)
        img_feature = self.image_encoder(img)
        text_attn_mask = text_attn_mask.to(text_feature.device)
        img_feature = img_feature.permute(1, 0, 2) 
        text_feature = text_feature.permute(1, 0, 2)
        for l in self.encoder_layers:
            img_feature, text_feature = l(img_feature, text_feature, text_attn_mask)
        src = cat([img_feature, text_feature], dim= 0)
        img_attn_mask = BoolTensor(size= (text_attn_mask.shape[0], img_feature.shape[0])).to(text_attn_mask.device)
        src_attn_mask = cat([text_attn_mask, img_attn_mask], dim= -1)
        src = self.encoder_fnn(self.encoder_fnn_drop(src))

        return self.encoder_fnn_norm(src), src_attn_mask

    def decoder_forward(self, src, tgt, src_attn_mask= None, tgt_attn_mask= None):
        """
        src: output of encoder (S, B, D)
        tgt: input (shifted right with bos_token) (T, B, D)
        """
        output = self.classifier(self.decoder(src, tgt, src_attn_mask, tgt_attn_mask))
        return output
    
