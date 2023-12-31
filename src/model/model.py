from torch import nn, cat, no_grad, tensor, rand, ones, uint8
from src.model.components.vision.encoders import EfficientNetEncoder, ImageEncoderViT, ImageEncoderRCNN
from src.model.components.language.encoders import ViT5Encoder, BARTphoEncoder
from src.model.components.transformer import MultiwayTransformer
from src.model.components.pooler import Pooler
from src.model.components.attentions import Compound, CompoundOCR, GuidedAttention, GuidedAttentionV2, OCRFuse, TransformerDecoderLayer, PositionalEncoding, GAT, TransformerEncoderLayer

from src.utils.loss import LabelSmoothingLoss

from transformers import AutoModel

import math

class VLMo(nn.Module):
    def __init__(self, vocab_size, max_text_len, freeze= True):
        super().__init__()

        self.image_encoder = ImageEncoderRCNN()
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
                 dropout_encoder= 0.2, dropout_decoder= 0.2, act= nn.ReLU, norm_first= False, freeze= None, return_loss= True
                ):
        super().__init__()

        # self.image_encoder = ImageEncoderViT(dim= d_model)
        # self.image_encoder= ImageEncoderRCNN(dim= d_model)
        self.image_encoder = EfficientNetEncoder(dim= d_model)
        # self.text_encoder = ViT5Encoder() # TODO: logic to change between diffenrent encoder
        self.text_encoder = BARTphoEncoder(hidden_dim= d_model)

        self.return_loss = return_loss

        self.d_model = d_model

        if freeze == 'both':
            self.image_encoder.freeze()
            self.text_encoder.freeze()
        elif freeze == 'img':
            self.image_encoder.freeze()
        elif freeze == 'text':
            self.text_encoder.freeze()
        # Comment out if use vanilla decoder
        # for params in self.text_encoder.model.get_decoder().parameters():
        #     params.requires_grad = True

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
        

    def forward(self, text, img, tgt, tgt_label):
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
                'loss': self.criterion(decoder_output, tgt_label),
                'logits': decoder_output
            }
        return {
            'logits': decoder_output
        }
    
    def encoder_forward(self, text, img):
        """
            - Output: src(seq_len, batch_size, d_model) mask(batch_size, seq_len)
        """
        text_attn_mask = text['attention_mask']
        text_feature = self.text_encoder(text)
        img_feature = self.image_encoder(img)
        img_attn_mask = ones(size= (text_attn_mask.shape[0], img_feature.shape[1]), dtype= uint8).to(text_attn_mask.device)

        text_attn_mask = text_attn_mask.to(text_feature.device)
        img_feature = img_feature.permute(1, 0, 2) 
        text_feature = text_feature.permute(1, 0, 2)
        for l in self.encoder_layers:
            img_feature, text_feature = l(img_feature, text_feature, (text_attn_mask == 0))
        src = cat([img_feature, text_feature], dim= 0)
        
        src_attn_mask = cat([img_attn_mask, text_attn_mask], dim= -1)
        src = self.encoder_fnn(self.encoder_fnn_drop(src))

        return self.encoder_fnn_norm(src), src_attn_mask

    def decoder_forward(self, src, tgt, src_attn_mask= None, tgt_attn_mask= None):
        """
        src: output of encoder (S, B, D)
        src_attn_mask: (B, S) 1 for not pad 0 for pad
        tgt: input (shifted right with bos_token) (T, B, D)
        """
        output = self.classifier(self.decoder(src, tgt, (src_attn_mask == 0), tgt_attn_mask))
        # print(output.shape)
        # using bert decoder
        # output = self.text_encoder.model.get_decoder()(
        #     input_ids= tgt,
        #     attention_mask= tgt_attn_mask,
        #     encoder_hidden_states = src.permute(1, 0, 2),
        #     encoder_attention_mask = src_attn_mask,
        #     return_dict= True
        # ).last_hidden_state

        # output = self.classifier(output)

        return output
    
class CompoundToken(nn.Module):
    def __init__(self,
                 vocab_size, pad_id,
                 d_model= 768,
                 nheads_encoder= 8,
                 nheads_decoder= 8,
                 num_encoder_layers= 6,
                 num_decoder_layers= 6,
                 hidden_dim= 2048,
                 dropout_encoder= 0.2,
                 dropout_decoder= 0.2,
                 act= nn.ReLU,
                 norm_first= False,
                 freeze= None,
                 return_loss= True
                ):
        super().__init__()

        self.image_encoder = EfficientNetEncoder(dim= int(d_model / 2))
        # self.image_encoder = ImageEncoderRCNN(dim= int(d_model / 2))
        self.text_encoder = BARTphoEncoder(hidden_dim= int(d_model / 2))

        self.return_loss = return_loss

        self.d_model = d_model
        if freeze == 'both':
            self.image_encoder.freeze()
            self.text_encoder.freeze()
        elif freeze == 'img':
            self.image_encoder.freeze()
        elif freeze == 'text':
            self.text_encoder.freeze()

        self.fuse = Compound(
            int(d_model / 2),
            nheads_encoder,
            hidden_dim= hidden_dim,
            act= act,
            dropout= dropout_encoder,
            norm_first= norm_first
        )
    
        self.encoder = TransformerEncoderLayer(
            vocab_size= vocab_size,
            d_model= d_model,
            nhead= nheads_encoder,
            dim_feedforward= hidden_dim,
            num_layers= num_encoder_layers,
            act= act,
            norm_first= norm_first,
            dropout= dropout_encoder
        )

        self.decoder = TransformerDecoderLayer(
            vocab_size= vocab_size,
            d_model= d_model,
            nhead= nheads_decoder,
            dim_feedforward= hidden_dim,
            num_layers= num_decoder_layers,
            act= act,
            norm_first= norm_first,
            dropout= dropout_decoder
        )

        self.classifier = nn.Linear(d_model, vocab_size)
        self.criterion = LabelSmoothingLoss(pad_id, 0.1)

    def forward(self, text, img, tgt, tgt_label):
        src, mask = self.encoder_forward(text, img)
        decoder_output = self.decoder_forward(src, tgt['input_ids'], mask, tgt['attention_mask'])

        if self.return_loss:
            return {
                'loss': self.criterion(decoder_output, tgt_label),
                'logits': decoder_output
            }
        return {
            'logits': decoder_output
        }

    def encoder_forward(self, text, img):
        """
            - Output: src(seq_len, batch_size, d_model) mask(batch_size, seq_len)
        """
        text_attn_mask = text['attention_mask']
        text_feature = self.text_encoder(text)
        img_feature = self.image_encoder(img)
        img_attn_mask = ones(size= (text_attn_mask.shape[0], img_feature.shape[1]), dtype= uint8).to(text_attn_mask.device)

        text_attn_mask = text_attn_mask.to(text_feature.device)
        img_feature = img_feature.permute(1, 0, 2) 
        text_feature = text_feature.permute(1, 0, 2)
        img_feature, text_feature = self.fuse(text_feature, img_feature, (text_attn_mask == 0))
        src = cat([img_feature, text_feature], dim= 0)
        src_attn_mask = cat([img_attn_mask, text_attn_mask], dim= -1)

        src = self.encoder(src, (src_attn_mask == 0))
        
        return src, src_attn_mask
    
    def decoder_forward(self, src, tgt, src_attn_mask= None, tgt_attn_mask= None):
        """
        src: output of encoder (S, B, D)
        src_attn_mask: (B, S) 1 for not pad 0 for pad
        tgt: input (shifted right with bos_token) (T, B, D)
        """
        output = self.classifier(self.decoder(src, tgt, (src_attn_mask == 0), tgt_attn_mask))
        return output   
        
class CompoundTokenOCR(nn.Module):
    def __init__(self,
                 vocab_size, pad_id,
                 d_model= 768,
                 nheads_encoder= 8,
                 nheads_decoder= 8,
                 num_encoder_layers= 6,
                 num_decoder_layers= 6,
                 hidden_dim= 2048,
                 dropout_encoder= 0.2,
                 dropout_decoder= 0.2,
                 act= nn.ReLU,
                 norm_first= False,
                 freeze= None,
                 return_loss= True
                ):
        super().__init__()

        self.image_encoder = EfficientNetEncoder(dim= int(d_model / 2))
        # self.image_encoder = ImageEncoderRCNN(dim= int(d_model / 2))
        self.text_encoder = BARTphoEncoder(hidden_dim= int(d_model / 2))

        self.return_loss = return_loss

        self.d_model = d_model
        if freeze == 'both':
            self.image_encoder.freeze()
            self.text_encoder.freeze()
        elif freeze == 'img':
            self.image_encoder.freeze()
        elif freeze == 'text':
            self.text_encoder.freeze()

        self.fuse = CompoundOCR(
            int(d_model / 2),
            nheads_encoder,
            hidden_dim= hidden_dim,
            act= act,
            dropout= dropout_encoder,
            norm_first= norm_first
        )
    
        self.encoder = TransformerEncoderLayer(
            vocab_size= vocab_size,
            d_model= d_model,
            nhead= nheads_encoder,
            dim_feedforward= hidden_dim,
            num_layers= num_encoder_layers,
            act= act,
            norm_first= norm_first,
            dropout= dropout_encoder
        )

        self.decoder = TransformerDecoderLayer(
            vocab_size= vocab_size,
            d_model= d_model,
            nhead= nheads_decoder,
            dim_feedforward= hidden_dim,
            num_layers= num_decoder_layers,
            act= act,
            norm_first= norm_first,
            dropout= dropout_decoder
        )

        self.classifier = nn.Linear(d_model, vocab_size)
        self.criterion = LabelSmoothingLoss(pad_id, 0.1)

    def forward(self, text, img, ocr, tgt, tgt_label):
        src, mask = self.encoder_forward(text, img, ocr)
        decoder_output = self.decoder_forward(src, tgt['input_ids'], mask, tgt['attention_mask'])

        if self.return_loss:
            return {
                'loss': self.criterion(decoder_output, tgt_label),
                'logits': decoder_output
            }
        return {
            'logits': decoder_output
        }

    def encoder_forward(self, text, img, ocr):
        """
            - Output: src(seq_len, batch_size, d_model) mask(batch_size, seq_len)
        """
        text_attn_mask = text['attention_mask']
        ocr_attn_mask = ocr['attention_mask']
        text_feature = self.text_encoder(text)
        ocr_feature = self.text_encoder(ocr)
        img_feature = self.image_encoder(img)
        img_attn_mask = ones(size= (text_attn_mask.shape[0], img_feature.shape[1]), dtype= uint8).to(text_attn_mask.device)

        text_attn_mask = text_attn_mask.to(text_feature.device)
        img_feature = img_feature.permute(1, 0, 2) 
        text_feature = text_feature.permute(1, 0, 2)
        ocr_feature = ocr_feature.permute(1, 0, 2)
        img_feature, text_feature, ocr_feature = self.fuse(text_feature, img_feature, ocr_feature, (text_attn_mask == 0))
        src = cat([ocr_feature, img_feature, text_feature], dim= 0)
        src_attn_mask = cat([ocr_attn_mask, img_attn_mask, text_attn_mask], dim= -1)

        src = self.encoder(src, (src_attn_mask == 0))
        
        return src, src_attn_mask
    
    def decoder_forward(self, src, tgt, src_attn_mask= None, tgt_attn_mask= None):
        """
        src: output of encoder (S, B, D)
        src_attn_mask: (B, S) 1 for not pad 0 for pad
        tgt: input (shifted right with bos_token) (T, B, D)
        """
        output = self.classifier(self.decoder(src, tgt, (src_attn_mask == 0), tgt_attn_mask))
        return output       

class GAv2(nn.Module):
    def __init__(self, vocab_size, pad_id, 
                 d_model= 768, nheads_encoder= 8, 
                 nheads_decoder= 8, num_encoder_layers= 6, 
                 num_decoder_layers= 6, hidden_dim= 2048, 
                 dropout_encoder= 0.2, dropout_decoder= 0.2, act= nn.ReLU, norm_first= False, freeze= None, return_loss= True
                ):
        super().__init__()

        # self.image_encoder = ImageEncoderViT(dim= d_model)
        # self.image_encoder= ImageEncoderRCNN(dim= d_model)
        self.image_encoder = EfficientNetEncoder(dim= d_model)
        # self.text_encoder = ViT5Encoder() # TODO: logic to change between diffenrent encoder
        self.text_encoder = BARTphoEncoder(hidden_dim= d_model)

        self.img_pos = PositionalEncoding(d_model, dropout_encoder)

        self.return_loss = return_loss

        self.d_model = d_model

        if freeze == 'both':
            self.image_encoder.freeze()
            self.text_encoder.freeze()
        elif freeze == 'img':
            self.image_encoder.freeze()
        elif freeze == 'text':
            self.text_encoder.freeze()
        # Comment out if use vanilla decoder
        # for params in self.text_encoder.model.get_decoder().parameters():
        #     params.requires_grad = True

        self.encoder_layers = nn.ModuleList([GuidedAttentionV2(
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

        self.encoder = TransformerEncoderLayer(
            vocab_size= vocab_size,
            d_model= d_model,
            nhead= nheads_encoder,
            dim_feedforward= hidden_dim,
            num_layers= num_encoder_layers,
            act= act,
            norm_first= norm_first,
            dropout= dropout_encoder
        )

        self.decoder = TransformerDecoderLayer(
            vocab_size= vocab_size,
            d_model= d_model,
            nhead= nheads_decoder,
            dim_feedforward= hidden_dim,
            num_layers= num_decoder_layers,
            act= act,
            norm_first= norm_first,
            dropout= dropout_decoder
        )

        self.encoder_fnn_drop = nn.Dropout(dropout_encoder)
        self.encoder_fnn_norm = nn.LayerNorm(d_model)

        self.encoder_fnn = nn.Linear(d_model, d_model)


        self.classifier = nn.Linear(d_model, vocab_size)
        self.criterion = LabelSmoothingLoss(pad_id, 0.1)
        

    def forward(self, text, img, tgt, tgt_label):
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
                'loss': self.criterion(decoder_output, tgt_label),
                'logits': decoder_output
            }
        return {
            'logits': decoder_output
        }
    
    def encoder_forward(self, text, img):
        """
            - Output: src(seq_len, batch_size, d_model) mask(batch_size, seq_len)
        """
        text_attn_mask = text['attention_mask']
        text_feature = self.text_encoder(text)
        img_feature = self.image_encoder(img)
        img_feature = self.img_pos(img_feature * math.sqrt(self.d_model))
        img_attn_mask = ones(size= (text_attn_mask.shape[0], img_feature.shape[1]), dtype= uint8).to(text_attn_mask.device)

        text_attn_mask = text_attn_mask.to(text_feature.device)
        img_feature = img_feature.permute(1, 0, 2) 
        text_feature = text_feature.permute(1, 0, 2)
        text_feature = self.encoder(text_feature, (text_attn_mask == 0))
        for l in self.encoder_layers:
            img_feature = l(img_feature, text_feature, (text_attn_mask == 0))
        src = cat([img_feature, text_feature], dim= 0)
        
        src_attn_mask = cat([img_attn_mask, text_attn_mask], dim= -1)
        src = self.encoder_fnn(self.encoder_fnn_drop(src))

        return self.encoder_fnn_norm(src), src_attn_mask

    def decoder_forward(self, src, tgt, src_attn_mask= None, tgt_attn_mask= None):
        """
        src: output of encoder (S, B, D)
        src_attn_mask: (B, S) 1 for not pad 0 for pad
        tgt: input (shifted right with bos_token) (T, B, D)
        """
        output = self.classifier(self.decoder(src, tgt, (src_attn_mask == 0), tgt_attn_mask))
        
        # using bert decoder
        # output = self.text_encoder.model.get_decoder()(
        #     input_ids= tgt,
        #     attention_mask= tgt_attn_mask,
        #     encoder_hidden_states = src.permute(1, 0, 2),
        #     encoder_attention_mask = src_attn_mask,
        #     return_dict= True
        # ).last_hidden_state

        # output = self.classifier(output)

        return output
    
class GAvOCR(nn.Module):
    def __init__(self, vocab_size, pad_id, 
                 d_model= 768, nheads_encoder= 8, 
                 nheads_decoder= 8, num_encoder_layers= 6, 
                 num_decoder_layers= 6, hidden_dim= 2048, 
                 dropout_encoder= 0.2, dropout_decoder= 0.2, act= nn.ReLU, norm_first= False, freeze= None, return_loss= True
                ):
        super().__init__()

        # self.image_encoder = ImageEncoderViT(dim= d_model)
        # self.image_encoder= ImageEncoderRCNN(dim= d_model)
        self.image_encoder = EfficientNetEncoder(dim= d_model)
        # self.text_encoder = ViT5Encoder() # TODO: logic to change between diffenrent encoder
        self.text_encoder = BARTphoEncoder(hidden_dim= d_model)

        self.img_pos = PositionalEncoding(d_model, dropout_encoder)

        self.return_loss = return_loss

        self.d_model = d_model

        if freeze == 'both':
            self.image_encoder.freeze()
            self.text_encoder.freeze()
        elif freeze == 'img':
            self.image_encoder.freeze()
        elif freeze == 'text':
            self.text_encoder.freeze()
        # Comment out if use vanilla decoder
        for params in self.text_encoder.model.get_decoder().parameters():
            params.requires_grad = True

        self.ocr_fusion = OCRFuse(self.text_encoder.model.get_input_embeddings(), 
                                  nheads= nheads_encoder, 
                                  num_layer= num_encoder_layers, 
                                  dim= d_model, dropout= dropout_encoder, 
                                  norm_first= norm_first, act= act,
                                  hidden_dim= hidden_dim
                                )

        self.encoder_layers = nn.ModuleList([GuidedAttentionV2(
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


        self.classifier = nn.Linear(d_model, vocab_size)
        self.criterion = LabelSmoothingLoss(pad_id, 0.1)
        

    def forward(self, ocr, text, img, tgt, tgt_label):
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

        src, mask = self.encoder_forward(ocr, text, img)

        # decoder_output = self.decoder(src, tgt['input_ids'], tgt['attention_mask'])
        # decoder_output = self.classifier(decoder_output)
        decoder_output = self.decoder_forward(src, tgt['input_ids'], mask, tgt['attention_mask'])

        if self.return_loss:
            return {
                'loss': self.criterion(decoder_output, tgt_label),
                'logits': decoder_output
            }
        return {
            'logits': decoder_output
        }
    
    def encoder_forward(self, ocr, text, img):
        """
            - Output: src(seq_len, batch_size, d_model) mask(batch_size, seq_len)
        """
        ocr_attn_mask = ocr['attention_mask']
        text_attn_mask = text['attention_mask']

        text_feature = self.text_encoder(text)
        img_feature = self.image_encoder(img)
        img_feature = self.img_pos(img_feature * math.sqrt(self.d_model))
        img_attn_mask = ones(size= (text_attn_mask.shape[0], img_feature.shape[1]), dtype= uint8).to(text_attn_mask.device)

        text_attn_mask = text_attn_mask.to(text_feature.device)
        ocr_attn_mask = ocr_attn_mask.to(text_feature.device)
        img_feature = img_feature.permute(1, 0, 2) 
        text_feature = text_feature.permute(1, 0, 2)
        ocr_feature = self.ocr_fusion(ocr['input_ids'], text_feature, (ocr_attn_mask == 0), (text_attn_mask == 0))
        for l in self.encoder_layers:
            img_feature = l(img_feature, text_feature, (text_attn_mask == 0))
        src = cat([ocr_feature, img_feature, text_feature], dim= 0)
        
        src_attn_mask = cat([ocr_attn_mask, img_attn_mask, text_attn_mask], dim= -1)
        src = self.encoder_fnn(self.encoder_fnn_drop(src))

        return self.encoder_fnn_norm(src), src_attn_mask

    def decoder_forward(self, src, tgt, src_attn_mask= None, tgt_attn_mask= None):
        """
        src: output of encoder (S, B, D)
        src_attn_mask: (B, S) 1 for not pad 0 for pad
        tgt: input (shifted right with bos_token) (T, B, D)
        """
        # output = self.classifier(self.decoder(src, tgt, (src_attn_mask == 0), tgt_attn_mask))
        
        # using bert decoder
        output = self.text_encoder.model.get_decoder()(
            input_ids= tgt,
            attention_mask= tgt_attn_mask,
            encoder_hidden_states = src.permute(1, 0, 2),
            encoder_attention_mask = src_attn_mask,
            return_dict= True
        ).last_hidden_state

        output = self.classifier(output)

        return output

# class GAvOCR(nn.Module):
#     def __init__(self, vocab_size, pad_id, 
#                  d_model= 768, nheads_encoder= 8, 
#                  nheads_decoder= 8, num_encoder_layers= 6, 
#                  num_decoder_layers= 6, hidden_dim= 2048, 
#                  dropout_encoder= 0.2, dropout_decoder= 0.2, act= nn.ReLU, norm_first= False, freeze= None, return_loss= True
#                 ):
#         super().__init__()

#         # self.image_encoder = ImageEncoderViT(dim= d_model)
#         # self.image_encoder= ImageEncoderRCNN(dim= d_model)
#         self.image_encoder = EfficientNetEncoder(dim= d_model)
#         # self.text_encoder = ViT5Encoder() # TODO: logic to change between diffenrent encoder
#         self.text_encoder = BARTphoEncoder(hidden_dim= d_model)

#         self.gat = GAT(self.text_encoder.model.get_input_embeddings(), d_model)

#         self.return_loss = return_loss

#         self.d_model = d_model

#         if freeze == 'both':
#             self.image_encoder.freeze()
#             self.text_encoder.freeze()
#         elif freeze == 'img':
#             self.image_encoder.freeze()
#         elif freeze == 'text':
#             self.text_encoder.freeze()
#         # Comment out if use vanilla decoder
#         # for params in self.text_encoder.model.get_decoder().parameters():
#         #     params.requires_grad = True

#         self.encoder_layers = nn.ModuleList([GuidedAttention(
#             dim= d_model,
#             nheads=nheads_encoder,  
#             dropout=dropout_encoder,  
#             hidden_dim= hidden_dim,
#             act= act,
#             norm_first= norm_first
#         ) for _ in range(num_encoder_layers)])

#         # self.encoder_fnn = nn.Sequential(
#         #     nn.Linear(d_model, hidden_dim),
#         #     nn.Dropout(dropout_encoder),
#         #     nn.Linear(hidden_dim, d_model),
#         # )

#         self.encoder_fnn_drop = nn.Dropout(dropout_encoder)
#         self.encoder_fnn_norm = nn.LayerNorm(d_model)

#         self.encoder_fnn = nn.Linear(d_model, d_model)

#         self.decoder = TransformerDecoderLayer(
#             vocab_size= vocab_size,
#             d_model= d_model,
#             nhead=nheads_decoder, 
#             dim_feedforward=hidden_dim,  
#             num_layers= num_decoder_layers,
#             norm_first= norm_first,
#             act= act,
#             dropout= dropout_decoder,
#         )

#         self.classifier = nn.Linear(d_model, vocab_size)
#         self.criterion = LabelSmoothingLoss(pad_id, 0.1)
        

#     def forward(self, text, img, tgt, tgt_label):
#         # text_feature = self.text_encoder(text)
#         # img_feature = self.image_encoder(img)

#         # # Swap dim 0, dim 1. From (batch_size, seq_length, hidden_dim) to (seq_length, batch_size, hidden_dim)
#         # img_feature = img_feature.permute(1, 0, 2) 
#         # text_feature = text_feature.permute(1, 0, 2)
        
#         # # x.shape = (batch_size, seq_length, hidden_dim)
#         # img_feature, text_feature = self.encoder_layers((img_feature, text_feature))

#         # src = cat([img_feature, text_feature], dim= 0)

#         # # src = self.encoder_fnn_norm(self.encoder_fnn_drop(src + self.encoder_fnn(src)))
#         # src = self.encoder_fnn(self.encoder_fnn_drop(src))

#         src, mask = self.encoder_forward(text, img)

#         # decoder_output = self.decoder(src, tgt['input_ids'], tgt['attention_mask'])
#         # decoder_output = self.classifier(decoder_output)
#         decoder_output = self.decoder_forward(src, tgt['input_ids'], mask, tgt['attention_mask'])

#         if self.return_loss:
#             return {
#                 'loss': self.criterion(decoder_output, tgt_label),
#                 'logits': decoder_output
#             }
#         return {
#             'logits': decoder_output
#         }
    
#     def encoder_forward(self, text, img):
#         """
#             - Output: src(seq_len, batch_size, d_model) mask(batch_size, seq_len)
#         """
#         text_attn_mask = text['attention_mask']
#         text_feature = self.text_encoder(text)
#         img_feature = self.image_encoder(img)
#         img_attn_mask = ones(size= (text_attn_mask.shape[0], img_feature.shape[1]), dtype= uint8).to(text_attn_mask.device)

#         text_attn_mask = text_attn_mask.to(text_feature.device)
#         img_feature = img_feature.permute(1, 0, 2) 
#         text_feature = text_feature.permute(1, 0, 2)
#         for l in self.encoder_layers:
#             img_feature, text_feature = l(img_feature, text_feature, (text_attn_mask == 0))
#         src = cat([img_feature, text_feature], dim= 0)
        
#         src_attn_mask = cat([img_attn_mask, text_attn_mask], dim= -1)
#         src = self.encoder_fnn(self.encoder_fnn_drop(src))

#         return self.encoder_fnn_norm(src), src_attn_mask

#     def decoder_forward(self, src, tgt, src_attn_mask= None, tgt_attn_mask= None):
#         """
#         src: output of encoder (S, B, D)
#         src_attn_mask: (B, S) 1 for not pad 0 for pad
#         tgt: input (shifted right with bos_token) (T, B, D)
#         """
#         output = self.classifier(self.decoder(src, tgt, (src_attn_mask == 0), tgt_attn_mask))
        
#         # using bert decoder
#         # output = self.text_encoder.model.get_decoder()(
#         #     input_ids= tgt,
#         #     attention_mask= tgt_attn_mask,
#         #     encoder_hidden_states = src.permute(1, 0, 2),
#         #     encoder_attention_mask = src_attn_mask,
#         #     return_dict= True
#         # ).last_hidden_state

#         # output = self.classifier(output)

#         return output
    
# This was intended for GAT -> decap
# class GAWithOCR(nn.Module):
#     def __init__(self, vocab_size, pad_id, 
#                  d_model= 768, nheads_encoder= 8, 
#                  nheads_decoder= 8, num_encoder_layers= 6, 
#                  num_decoder_layers= 6, hidden_dim= 2048, 
#                  dropout_encoder= 0.2, dropout_decoder= 0.2, act= nn.ReLU, norm_first= False, freeze= None, return_loss= True
#                 ):
#         super().__init__()

#         # self.image_encoder = ImageEncoderViT(dim= d_model)
#         # self.image_encoder= ImageEncoderRCNN(dim= d_model)
#         self.image_encoder = EfficientNetEncoder(dim= d_model)
#         # self.text_encoder = ViT5Encoder() # TODO: logic to change between diffenrent encoder
#         self.text_encoder = BARTphoEncoder(hidden_dim= d_model)

#         self.gat = GAT(self.text_encoder.model.get_input_embeddings(), d_model)

#         self.return_loss = return_loss

#         self.d_model = d_model

#         if freeze == 'both':
#             self.image_encoder.freeze()
#             self.text_encoder.freeze()
#         elif freeze == 'img':
#             self.image_encoder.freeze()
#         elif freeze == 'text':
#             self.text_encoder.freeze()
#         # Comment out if use vanilla decoder
#         # for params in self.text_encoder.model.get_decoder().parameters():
#         #     params.requires_grad = True

#         self.encoder_layers = nn.ModuleList([GuidedAttention(
#             dim= d_model,
#             nheads=nheads_encoder,  
#             dropout=dropout_encoder,  
#             hidden_dim= hidden_dim,
#             act= act,
#             norm_first= norm_first
#         ) for _ in range(num_encoder_layers)])

#         # self.encoder_fnn = nn.Sequential(
#         #     nn.Linear(d_model, hidden_dim),
#         #     nn.Dropout(dropout_encoder),
#         #     nn.Linear(hidden_dim, d_model),
#         # )

#         self.encoder_fnn_drop = nn.Dropout(dropout_encoder)
#         self.encoder_fnn_norm = nn.LayerNorm(d_model)

#         self.encoder_fnn = nn.Linear(d_model, d_model)

#         self.decoder = TransformerDecoderLayer(
#             vocab_size= vocab_size,
#             d_model= d_model,
#             nhead=nheads_decoder, 
#             dim_feedforward=hidden_dim,  
#             num_layers= num_decoder_layers,
#             norm_first= norm_first,
#             act= act,
#             dropout= dropout_decoder,
#         )

#         self.classifier = nn.Linear(d_model, vocab_size)
#         self.criterion = LabelSmoothingLoss(pad_id, 0.1)
        

#     def forward(self, text, img, tgt, tgt_label):
#         # text_feature = self.text_encoder(text)
#         # img_feature = self.image_encoder(img)

#         # # Swap dim 0, dim 1. From (batch_size, seq_length, hidden_dim) to (seq_length, batch_size, hidden_dim)
#         # img_feature = img_feature.permute(1, 0, 2) 
#         # text_feature = text_feature.permute(1, 0, 2)
        
#         # # x.shape = (batch_size, seq_length, hidden_dim)
#         # img_feature, text_feature = self.encoder_layers((img_feature, text_feature))

#         # src = cat([img_feature, text_feature], dim= 0)

#         # # src = self.encoder_fnn_norm(self.encoder_fnn_drop(src + self.encoder_fnn(src)))
#         # src = self.encoder_fnn(self.encoder_fnn_drop(src))

#         src, mask = self.encoder_forward(text, img)

#         # decoder_output = self.decoder(src, tgt['input_ids'], tgt['attention_mask'])
#         # decoder_output = self.classifier(decoder_output)
#         decoder_output = self.decoder_forward(src, tgt['input_ids'], mask, tgt['attention_mask'])

#         if self.return_loss:
#             return {
#                 'loss': self.criterion(decoder_output, tgt_label),
#                 'logits': decoder_output
#             }
#         return {
#             'logits': decoder_output
#         }
    
#     def encoder_forward(self, text, img):
#         """
#             - Output: src(seq_len, batch_size, d_model) mask(batch_size, seq_len)
#         """
#         text_attn_mask = text['attention_mask']
#         text_feature = self.text_encoder(text)
#         img_feature = self.image_encoder(img)
#         img_attn_mask = ones(size= (text_attn_mask.shape[0], img_feature.shape[1]), dtype= uint8).to(text_attn_mask.device)

#         text_attn_mask = text_attn_mask.to(text_feature.device)
#         img_feature = img_feature.permute(1, 0, 2) 
#         text_feature = text_feature.permute(1, 0, 2)
#         for l in self.encoder_layers:
#             img_feature, text_feature = l(img_feature, text_feature, (text_attn_mask == 0))
#         src = cat([img_feature, text_feature], dim= 0)
        
#         src_attn_mask = cat([img_attn_mask, text_attn_mask], dim= -1)
#         src = self.encoder_fnn(self.encoder_fnn_drop(src))

#         return self.encoder_fnn_norm(src), src_attn_mask

#     def decoder_forward(self, src, tgt, src_attn_mask= None, tgt_attn_mask= None):
#         """
#         src: output of encoder (S, B, D)
#         src_attn_mask: (B, S) 1 for not pad 0 for pad
#         tgt: input (shifted right with bos_token) (T, B, D)
#         """
#         output = self.classifier(self.decoder(src, tgt, (src_attn_mask == 0), tgt_attn_mask))
        
#         # using bert decoder
#         # output = self.text_encoder.model.get_decoder()(
#         #     input_ids= tgt,
#         #     attention_mask= tgt_attn_mask,
#         #     encoder_hidden_states = src.permute(1, 0, 2),
#         #     encoder_attention_mask = src_attn_mask,
#         #     return_dict= True
#         # ).last_hidden_state

#         # output = self.classifier(output)

#         return output
    
# TODO: Multiway Transformer
# TODO: DT-fixup implementation???
# TODO: add GAT layer for ocr token