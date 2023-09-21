"""import modules"""
from torch.nn import Module
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM

class ViT5Encoder(Module):
    """
    ViT5 Encoder for question
        - pretrained: name of pretrained weights
    """
    def __init__(
        self,
        pretrained = 'VietAI/vit5-base', # Can change to large or base
    ) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrained)

    def forward(self, input):
        """
            - input: input_ids (encoding in this case)
            - output shape: (batch_size, sequence_length, hidden_size) [1, max_length, 768]
        """
        vit5_input_ids, vit5_attention_masks = input["input_ids"], input["attention_mask"]
        outputs = self.model.encoder(
            input_ids = vit5_input_ids,
            attention_mask = vit5_attention_masks,
            return_dict = True
        )

        return outputs.last_hidden_state


class BARTphoEncoder(Module):
    """
    BARTpho Encoder for question
        - pretrained: name of pretrained weights
    """
    def __init__(
        self,
        pretrained = 'vinai/bartpho-syllable-base', # Can change to word or syllable
    ) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrained)
    
    def forward(self, input):
        """
        - input: input_ids
        - output shape: (batch_size, sequence_length, hidden_size) [1, 1024, 1024]
        """
        outputs = self.model(**input)
        return outputs.last_hidden_state