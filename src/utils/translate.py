import torch
from tqdm import tqdm

# TODO: Add beam search
def translate(model, img, text, bos_token_id, eos_token_id, pad_token_id, max_len):
    device = img.device

    batch_size = img.shape[0]

    src = model.encoder_forward(text, img)

    sent = torch.tensor([[bos_token_id] * batch_size], device= device)

    for _ in range(max_len):
        output = model.decoder_forward(src, sent.T)
        output = torch.nn.functional.softmax(output, dim= -1)

        _, idx = torch.topk(output, 3)

        idx = idx[:, -1, 0]

        sent = torch.cat([sent, idx.reshape(-1, batch_size)], 0)

    sent = sent.T.cpu().detach()
    for i in range(sent.shape[0]):
        idx = (sent[i] == eos_token_id).nonzero().flatten()
        if idx.dim == 0:
            sent = sent[i, idx[0] + 1:] = pad_token_id
    return sent