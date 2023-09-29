"""import modules"""
from torch import nn
from torch.nn import Module
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, T5EncoderModel, AutoConfig
import torch

class ViT5Encoder(Module):
    """
    ViT5 Encoder for question
        - pretrained: name of pretrained weights
    """
    def __init__(
        self,
        pretrained = 'VietAI/vit5-base', # Can change to large or base
        hidden_dim = 768
    ) -> None:
        super().__init__()
        self.model = T5EncoderModel.from_pretrained(pretrained)
        self.hidden_dim = hidden_dim

    def forward(self, input):
        """
            - input: input_ids, attention_mask (encoding in this case)
            - output shape: (batch_size, sequence_length, hidden_size) [1, max_length, 768] (768 for base, 1024 for large)
        """
        vit5_input_ids, vit5_attention_masks = input["input_ids"], input["attention_mask"]
        outputs = self.model.encoder(
            input_ids = vit5_input_ids,
            attention_mask = vit5_attention_masks,
            return_dict = True
        ).last_hidden_state
        if self.hidden_dim != outputs.shape[-1]:
            outputs = nn.Linear(outputs.shape[-1], self.hidden_dim)
        return outputs
    
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False


class BARTphoEncoder(Module):
    """
    BARTpho Encoder for question
        - pretrained: name of pretrained weights
    """
    def __init__(
        self,
        pretrained = 'vinai/bartpho-syllable-base', # Can change to word or syllable
        hidden_dim = 768
    ) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrained)
        self.hidden_dim = hidden_dim
        self.config = AutoConfig.from_pretrained(pretrained)
        self.proj = nn.Linear(self.config.hidden_size*4, hidden_dim)

    def forward(self, input):
        """
        - input: input_ids, attention_mask
        - output shape: (batch_size, sequence_length, hidden_size) [1, max_length, 768] (768 for base, 1024 for large)
        """
        # # Remove token_type_ids from the input dictionary [ONLY IF USE BARTPHO-WORD], tại trong MBart k có token_type_ids 
        # input.pop('token_type_ids', None)

        # ------------------------------------------------------------------------------------
        # TODO: fix this shit -> output no shape attribute (Seq2SeqModelOutput type) -> done?
        # outputs = self.model(**input)

        # outputs_encoder_lhs = outputs.encoder_last_hidden_state

        # # Get the last value of outputs_encoder_lhs (last hidden state) for each sequence in the batch - (batch size, hidden size)
        # last_hidden_state = outputs_encoder_lhs[:, -1, :]
        # # Extract the hidden_size dimension from last_hidden_state
        # hidden_size = last_hidden_state.size(-1)
        # # print(hidden_size)

        # if self.hidden_dim != hidden_size:
        #     outputs = nn.Linear(hidden_size, self.hidden_dim)
  
        # return outputs_encoder_lhs
        # ------------------------------------------------------------------------------------

        # Return 4 layers of encoder concatinated for better performance
        # See: https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently
        outputs = self.model(input['input_ids'], input['attention_mask'], output_hidden_states= True, return_dict= True)
        all_hidden_states = outputs.encoder_hidden_states

        concatenate_pooling = torch.cat(
            (all_hidden_states[-1], all_hidden_states[-2], all_hidden_states[-3], all_hidden_states[-4]),-1
        )

        logits = self.proj(concatenate_pooling) 

        return logits

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False