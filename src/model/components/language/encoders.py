"""import modules"""
from torch import nn
from torch.nn import Module
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, T5EncoderModel

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

    def forward(self, input):
        """
        - input: input_ids, attention_mask
        - output shape: (batch_size, sequence_length, hidden_size) [1, max_length, 768] (768 for base, 1024 for large)
        """
        # # Remove token_type_ids from the input dictionary [ONLY IF USE BARTPHO-WORD], tại trong MBart k có token_type_ids 
        # input.pop('token_type_ids', None)

        outputs = self.model(**input)
        if self.hidden_dim != outputs.shape - 1:
            outputs = nn.Linear(outputs.shape - 1, self.hidden_dim)
      
        return outputs.last_hidden_state