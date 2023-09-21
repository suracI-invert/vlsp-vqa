from torch import nn, linspace
from src.model.components.attentions import Block

class MultiwayTransformer(nn.Module):
    def __init__(
            self, 
            embed_dim= 768,
            depth= 4,
            num_heads= 6,
            fnn_ratio= 4,
            drop_rate= 0.0,
            attn_drop_rate= 0.0,
            drop_path_rate= 0.0,
            qk_scale= None,
            qkv_bias= True,
            vlffn_start_layer_index= 2,
            need_relative_position_embed= True,
            use_abs_pos_emb= True,
            layer_scale_init_value= 0.1,
            max_text_len= 128,
    ):
        super().__init__()

        drop_path_rate = drop_path_rate
        self.use_abs_pos_emb = use_abs_pos_emb
        self.need_relative_position_embed = need_relative_position_embed

        self.num_features = embed_dim

        self.num_heads = num_heads
        self.vlffn_start_layer_index = vlffn_start_layer_index

        dpr = [
            x.item() for x in linspace(0, drop_path_rate, depth)
        ]
        
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim= embed_dim,
                    num_heads= num_heads,
                    fnn_ratio= fnn_ratio,
                    qkv_bias= qkv_bias,
                    qk_scale= qk_scale,
                    drop= drop_rate,
                    attn_drop= attn_drop_rate,
                    drop_path= dpr[i],
                    # norm_layer= norm_layer,
                    with_vlffn= (i > self.vlffn_start_layer_index),
                    layer_scale_init_values= layer_scale_init_value,
                    max_text_len= max_text_len,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

