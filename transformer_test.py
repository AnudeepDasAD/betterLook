import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torchvision import models

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedder = nn.Embedding(ntoken, d_model)

        #d_model is hidden dimension
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # src = self.embedder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):
    '''PositionalEncoding module injects some information about the relative or absolute position of the tokens in the sequence. 
        The positional encodings have the same dimension as the embeddings so that the two can be summed. 
        Here, we use sine and cosine functions of different frequencies.'''
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 8, d_model) # Changed from 1 to 8 to match shape
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] (embedding_dim = d_model?)
        """
        # print(f'x shape: {x.shape}')
        # print(self.pe[:x.size(0)].shape)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


import copy
import time

def train(model: nn.Module, epoch: int, dl, resnet_model) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 20
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    batch = 1
    seq_len = 6
    for data, targets in dl:
        data = data.to(device)   # for use with a GPU
        data.requires_grad = True
        # print(data.shape)
        # targets = targets.type(torch.FloatTensor)

        conv_output = resnet_model.cuda()(data).to(device)
        conv_output = torch.stack([conv_output for i in range(seq_len)])
        attn_input = conv_output.view(seq_len, bptt, -1).to(device)
        print(attn_input.shape) #6 x 8 x 1000
        if attn_input.shape[-1] != emsize:
            continue

        # seq_len = data.size(0)
        if seq_len != bptt:  # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]
        output = model(attn_input, src_mask)
        # We should actually change the output dimension of the attn network
        # and pass it through a fully connected layer thereafter
        output = output.view(bptt, seq_len, -1)[:, -1, :].to('cpu')
        # preds = torch.argmax(output.cpu(), dim=1).type(torch.FloatTensor).to(device)
        # print(f'output: {output.shape}') # 8 x 100
        # print(f'targets: {targets}')
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | '#{batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()
        batch += 1


def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            seq_len = data.size(0)
            if seq_len != bptt:
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


if __name__ == '__main__':
    ntokens = 100  # number of labels
    emsize = 1000  # embedding dimension (choose to be same as dim of output of conv)
    d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bptt = 8
    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    for param in model.parameters():
        print(param.shape)
    # lr = 5.0  # learning rate
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    # best_val_loss = float('inf')
    # epochs = 3
    # best_model = None
    # ds = torch.load('imagenette_train_preproc_dataset0.pt')
    # dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)
    # weights=models.ResNet50_Weights.IMAGENET1K_V2
    # resnet_model = models.resnet50(weights=weights)
    # resnet_model.eval()
    # for epoch in range(1, epochs + 1):
    #     epoch_start_time = time.time()
    #     train(model, epoch, dl, resnet_model)
    #     # val_loss = evaluate(model, val_data)
    #     # val_ppl = math.exp(val_loss)
    #     elapsed = time.time() - epoch_start_time
    #     print('-' * 89)
    #     print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | ')
    #         # f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    #     print('-' * 89)

    #     # if val_loss < best_val_loss:
    #     #     best_val_loss = val_loss
    #     #     best_model = copy.deepcopy(model)

    #     scheduler.step()