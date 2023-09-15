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
        pretrained = 'VietAI/vit5-large' # Can change to large or base
    ) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrained)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)

    def forward(self, text):
        """
            - input: text
            - output shape: (batch_size, sequence_length, hidden_size) [1, ?, 768]
        """
        vit5_encoding = self.tokenizer(text, return_tensors = "pt", padding = 'max_length') 
        # Chưa biết có cần thêm max_length = 1024 khum, nếu thêm thì seq_len = 1024
        vit5_input_ids, vit5_attention_masks = vit5_encoding["input_ids"], vit5_encoding["attention_mask"]
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
        pretrained = 'vinai/bartpho-syllable' # Can change to word or syllable
    ) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrained)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
    
    def forward(self, text):
        """
        - input: text
        - output shape: (batch_size, sequence_length, hidden_size) [1, 1024, 1024]
        """
        bartpho_input_ids = self.tokenizer(text, return_tensors = "pt", padding = 'max_length')
        outputs = self.model(**bartpho_input_ids)
        return outputs.last_hidden_state