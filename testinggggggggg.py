import os
os.environ['TRANSFORMERS_CACHE'] = './.cache'
import json
import torch
from src.model.components.language.encoders import ViT5Encoder, BARTphoEncoder
from src.model.components.attentions import Block
from transformers import AutoTokenizer

bartpho_encoder = BARTphoEncoder(pretrained = 'vinai/bartpho-syllable-base')

bartpho_tokenizer =  AutoTokenizer.from_pretrained("vinai/bartpho-syllable-base")

text = "something"

