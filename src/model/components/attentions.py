from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
import math

from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)

    def drop_path(self, x, drop_prob: float = 0., training: bool = False):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output
    
class FNN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, mask=None, relative_position_bias=None):
        B, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # print('Weights shape: ', self.qkv.weight.shape)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        # print('QKV shape: ', qkv.shape)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        # print('Query Shape: ', q.shape)
        # print('Key Shape: ', k.shape)
        # print('Value Shape: ', v.shape)
        q = q * self.scale
        attn = (q.float() @ k.float().transpose(-2, -1))
        
        if relative_position_bias is not None:
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1).type_as(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        fnn_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        # norm_layer=nn.LayerNorm,
        with_vlffn=False,
        layer_scale_init_values=0.1,
        max_text_len=40,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # self.attn_v2 = nn.MultiheadAttention(
        #     dim, num_heads, attn_drop, add_bias_kv= qkv_bias, batch_first= True
        # )
        # NOTE: Can change to dropout / droppath used in original vlmo src code
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2_text = nn.LayerNorm(dim)
        self.norm2_img = nn.LayerNorm(dim)
        fnn_hidden_dim = int(dim * fnn_ratio)
        self.fnn_text = FNN(
            in_features=dim,
            hidden_features=fnn_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.fnn_img = FNN(
            in_features=dim,
            hidden_features=fnn_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.fnn_vl = None
        if with_vlffn:
            self.fnn_vl = FNN(
                in_features=dim,
                hidden_features=fnn_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
            self.norm2_vl = nn.LayerNorm(dim)
    
        self.gamma_1 = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)),requires_grad=True) \
            if layer_scale_init_values is not None else 1.0
        self.gamma_2 = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)),requires_grad=True) \
            if layer_scale_init_values is not None else 1.0

        self.max_text_len = max_text_len

    def forward(self, x, mask=None, modality_type=None, relative_position_bias=None):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), mask=mask, relative_position_bias=relative_position_bias))
        # x = x + self.drop_path(self.gamma_1 * self.attn_v2(self.norm1(x), self.norm1(x), self.norm1(x), ))

        if modality_type == "text":
            x = x + self.drop_path(self.gamma_2 * self.fnn_text(self.norm2_text(x))) 
        elif modality_type == "image":
            x = x + self.drop_path(self.gamma_2 * self.fnn_img(self.norm2_img(x))) 
        else:
            if self.fnn_vl is None:
                x_text = x[:, : self.max_text_len]
                x_img = x[:, self.max_text_len :]
                x_text = x_text + self.drop_path(self.gamma_2 * self.fnn_text(self.norm2_text(x_text)))
                x_img = x_img + self.drop_path(self.gamma_2 * self.fnn_img(self.norm2_img(x_img)))
                x = torch.cat([x_text, x_img], dim=1)
                # print(x_text.shape)
                # print(x_img.shape)
                # print('*' * 50)
                # print(x.shape)
            else:
                x = x + self.drop_path(self.gamma_2 * self.fnn_vl(self.norm2_vl(x)))

        return x
    
class ResidualConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Layer norm was pushed to the front for better performance and stability (facing grad vanishing/exploding while training using norm last).
    https://arxiv.org/pdf/2002.04745.pdf
    """
    def __init__(self, size, dropout, norm_first= False):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        if self.norm_first:
            x = x + self.dropout(sublayer(self.norm(x)))
        else:
            x = self.norm(x + self.dropout(sublayer(x)))
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, 
            vocab_size,
            d_model,
            nhead,
            dim_feedforward,
            num_layers,
            act, norm_first= False,
            dropout= 0.1
        ):
        super().__init__()

        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            act(),
            norm_first
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers,
        )
    
    def forward(self, x, mask):
        x = self.transformer_encoder(x, src_key_padding_mask= mask)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self,
            vocab_size,
            d_model,
            nhead,
            dim_feedforward,
            num_layers,
            act,
            norm_first= False,
            dropout= 0.1,
        ):
        super().__init__()

        self.d_model = d_model

        self.pos_enc = PositionalEncoding(d_model)
        self.emb = nn.Embedding(vocab_size, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model = d_model,
            nhead = nhead,
            activation= act(),
            dim_feedforward = dim_feedforward,
            norm_first= norm_first,
            dropout= dropout
        ) 

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer = decoder_layer,
            num_layers = num_layers
        )

        self.nhead = nhead

    def forward(self, src, tgt, src_attn_mask= None, tgt_attn_mask= None):
        # tgt_padding_mask = tgt_padding_mask.to(tgt.device)
        tgt = tgt.transpose(0, 1)
        subsequent_mask = self.gen_mask(tgt, self.nhead, tgt_attn_mask).to(tgt.device)
        tgt = self.pos_enc(self.emb(tgt) * math.sqrt(self.d_model))
        output = self.transformer_decoder(tgt, src, memory_key_padding_mask= src_attn_mask, tgt_mask= subsequent_mask)
        
        return output.transpose(0, 1)
    
    def gen_mask(self, tgt, num_head, key_padding= None):
        "Mask out subsequent positions. tgt shape: (T, N)"
        device = tgt.device
        size, batch_size = tgt.shape[0], tgt.shape[1]
        key_padding = key_padding if key_padding is not None else torch.zeros((batch_size, size), dtype= torch.bool)
        key_padding = key_padding.unsqueeze(1).repeat([1, size, 1]).to(device)
        attn_shape = (batch_size, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = (torch.from_numpy(subsequent_mask).to(device) | key_padding) == 1
        # According to pytorch test case this is how 3d mask is stacked: https://github.com/pytorch/pytorch/blob/c74c0c571880df886474be297c556562e95c00e0/test/test_nn.py#L5039 line 5039
        return torch.repeat_interleave(subsequent_mask, num_head, dim= 0)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len= 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]

        return self.dropout(x)
        
class GuidedAttention(nn.Module):
    def __init__(self, dim, nheads, dropout, hidden_dim, act, norm_first= False):
        super().__init__()
        self.d_model = dim

        self.img_pos = PositionalEncoding(dim, dropout)
        # self.text_pos = PositionalEncoding(dim, dropout)

        self.text_attn = nn.MultiheadAttention(dim, nheads, dropout)
        self.img_attn = nn.MultiheadAttention(dim, nheads, dropout) 
        
        # self.text_drop = nn.Dropout(dropout)
        # self.img_drop = nn.Dropout(dropout)

        # self.text_norm = nn.LayerNorm(dim)
        # self.img_norm = nn.LayerNorm(dim)

        self.img_attn_res = ResidualConnection(dim, dropout, norm_first)
        self.text_attn_res = ResidualConnection(dim, dropout, norm_first)

        self.ga = nn.MultiheadAttention(dim, nheads, dropout)
        # self.ga_drop = nn.Dropout(dropout)
        # self.ga_norm = nn.LayerNorm(dim)
        self.ga_res = ResidualConnection(dim, dropout, norm_first)

        self.img_ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        # self.img_ffn_drop = nn.Dropout(dropout)
        # self.img_ffn_norm = nn.LayerNorm(dim)
        self.img_ffn_res = ResidualConnection(dim, dropout, norm_first)

        self.text_ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        # self.text_ffn_drop = nn.Dropout(dropout)
        # self.text_ffn_norm = nn.LayerNorm(dim)
        self.text_ffn_res = ResidualConnection(dim, dropout, norm_first)

    def forward(self, img, text, text_mask):
        # text = self.text_norm(self.text_drop(text + self.text_attn(text, text, text)[0]))
        text = self.text_attn_res(text, lambda text: self.text_attn(text, text, text, key_padding_mask= text_mask)[0])

        # img = self.img_norm(self.img_drop(img + self.img_attn(img, img, img)[0]))
        img = self.img_pos(img * math.sqrt(self.d_model))
        img = self.img_attn_res(img, lambda img: self.img_attn(img, img, img)[0])

        # ga = self.ga_norm(self.ga_drop(img + self.ga(img, text, text)[0]))
        ga = self.ga_res(img, lambda img: self.ga(img, text, text, key_padding_mask= text_mask)[0])

        # text = self.text_ffn_norm(self.text_ffn_drop(text + self.text_ffn(text)))
        text = self.text_ffn_res(text, lambda text: self.text_ffn(text))
        # img = self.img_ffn_norm(self.img_ffn_drop(ga + self.img_ffn(ga)))
        img = self.img_ffn_res(ga, lambda ga: self.img_ffn(ga))

        return ((img, text))
    
class Compound(nn.Module):
    def __init__(self, dim, nhead, hidden_dim, act, dropout, norm_first):
        super().__init__()

        self.img_cross_attn = nn.MultiheadAttention(
            embed_dim= dim,
            num_heads= nhead,
            dropout= dropout
        )
        self.img_cross_attn_res = ResidualConnection(dim, dropout, norm_first)

        self.img_fnn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )
        self.img_fnn_res = ResidualConnection(dim, dropout)

        self.text_cross_attn = nn.MultiheadAttention(
            embed_dim= dim,
            num_heads= nhead,
            dropout= dropout
        )   
        self.text_cross_attn_res = ResidualConnection(dim, dropout, norm_first)
        self.text_fnn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, text, img, text_attn):
        """
            input should have dim/2 for concatenation
        """
        text_feature = self.text_cross_attn_res(text, lambda text: self.text_cross_attn(text, img, img))

class GAT(nn.Module):
    def __init__(self, vocab_size, d_model, hidden_channels, dropout= 0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.conv1 = GATConv(d_model, hidden_channels, dropout= dropout)
        self.conv2 = GATConv(hidden_channels, hidden_channels, dropout= dropout)
        self.conv3 = GATConv(hidden_channels, hidden_channels, dropout= dropout)

    def foward(self, node_ids, edge_index, batch):
        x = self.emb(node_ids)
        x = self.conv1(x, edge_index)
        x = x.ReLU()
        x = self.conv2(x, edge_index)
        x = x.ReLU()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        return x
