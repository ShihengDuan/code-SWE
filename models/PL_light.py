# For attention model since the training time is soooooo long. 

import pytorch_lightning as pl
from models.attention import Attention
from models.lstm import LSTM
from torch import nn
import torch

class Attention_PL(pl.LightningModule):
    def __init__(self, n_head, dim_feedforward, num_att_layers, embedding_size, input_size):
        super().__init__()
        self.save_hyperparameters()
        self.attention = Attention(n_head=n_head, dim_feedforward=dim_feedforward,
        num_att_layers=num_att_layers, embedding_size=embedding_size, input_size=input_size)
        self.loss = nn.MSELoss()

    def configure_optimizers(self):
        # lr = self.hparams.lr
        lr = 1e-4
        opt = torch.optim.Adam(lr=lr, params=self.attention.parameters())
        return opt

    def att_forward(self, x_d_new, x_attr):
        y_hat = self.attention(x_d_new, x_attr)[0]
        y_hat_sub = y_hat[:,-1:,:]
        return y_hat_sub

    def training_step(self, batch, batch_idx):
        x_d_new, x_attr, y_d_new, qstd = batch
        y_hat_sub = self.lstm_forward(x_d_new, x_attr)
        y_sub = y_d_new[:, -1:]
        loss = self.loss(y_hat_sub, y_sub)
        self.log('MSELoss', loss, on_epoch=True, on_step=False)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx):
        x_d_new, x_attr, y_d_new, qstd = batch
        y_hat_sub = self.lstm_forward(x_d_new, x_attr)
        y_sub = y_d_new[:, -1:]
        return y_sub, y_hat_sub # real, pred
    
class LSTM_PL(pl.LightningModule):
    def __init__(self, hidden_units, input_size):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = LSTM(hidden_units=hidden_units, input_size=input_size, relu_flag=False)
        self.loss = nn.MSELoss()

    def configure_optimizers(self):
        # lr = self.hparams.lr
        lr = 1e-4
        opt = torch.optim.Adam(lr=lr, params=self.lstm.parameters())
        return opt

    def att_forward(self, x_d_new, x_attr):
        y_hat = self.lstm(x_d_new, x_attr)[0]
        y_hat_sub = y_hat[:,-1:,:]
        return y_hat_sub

    def training_step(self, batch, batch_idx):
        x_d_new, x_attr, y_d_new, qstd = batch
        y_hat_sub = self.lstm_forward(x_d_new, x_attr)
        y_sub = y_d_new[:, -1:]
        loss = self.loss(y_hat_sub, y_sub)
        self.log('MSELoss', loss, on_epoch=True, on_step=False)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx):
        x_d_new, x_attr, y_d_new, qstd = batch
        y_hat_sub = self.lstm_forward(x_d_new, x_attr)
        y_sub = y_d_new[:, -1:]
        return y_sub, y_hat_sub # real, pred