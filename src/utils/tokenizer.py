from transformers import AutoTokenizer

def get_tokenizer(pretrained):
    print(pretrained)
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({'bos_token': '<s>'})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '</s>'})
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '<mask>'})
    return tokenizer