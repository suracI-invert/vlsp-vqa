from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

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
