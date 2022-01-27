import math

import numpy as np
import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, n_head, dim_feedforward, num_att_layers, embedding_size, input_size=14, drop=0.3):
        super(Attention, self).__init__()
        # Assuming encoder_dim=embedding_dim, since no embedding is used.
        # if add position, not concat, input size will not be changed.
        self.embedding_size = embedding_size
        self.dense = nn.Linear(in_features=self.embedding_size, out_features=1)
        self.embedding = nn.Linear(in_features=input_size, out_features=self.embedding_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.embedding_size,
                                                    nhead=n_head,
                                                    dim_feedforward=dim_feedforward,
                                                    dropout=drop)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layers,
                                             num_layers=num_att_layers,
                                             norm=None)  # why does normalization matter?
        self.dropout = nn.Dropout(drop)
        self.pos_encoder = PositionalEncoding(d_model=self.embedding_size)
        self.mask = None  # positional mask.

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor):
        # input : (S,N,E):
        x_d = x_d.transpose(0, 1)
        # concat all inputs
        if x_s.nelement() > 0:
            x_s = x_s.unsqueeze(0).repeat(x_d.shape[0], 1, 1)  # seq, batch, attributions.
            x_d = torch.cat([x_d, x_s], dim=-1)  # seq, batch, feature+attr
        else:
            pass
        x_d = self.embedding(x_d)
        x_d = self.pos_encoder(x_d)
        if self.mask is None or self.mask.size(0) != len(x_d):  # either first time or last batch.
            mask = (torch.triu(torch.ones(len(x_d), len(x_d))) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.mask = mask.to(x_d.device)
        output = self.encoder(x_d, self.mask)
        y_hat = self.dense(self.dropout(output.transpose(0, 1)))
        return y_hat, output, output  # keep the same number of outputs with LSTM.


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.3, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, int(np.ceil(d_model / 2) * 2))
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(max_len * 2) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe[:, :d_model].unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos_type='sum'):  # x: Seq, Batch, Features
        if pos_type.lower() in ['concat', 'cat', 'concatenate']:
            x = torch.cat((x, self.pe[:x.size(0), :].repeat(1, x.size(1), 1)), 2)
        elif pos_type.lower() in ['sum', 'add', 'addition']:
            x = x + self.pe[:x.size(0), :]
        else:
            raise RuntimeError(f"Unrecognized positional encoding type: {pos_type}")
        return self.dropout(x)
