import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, hidden_units, input_size=14, drop=0.3, relu_flag=False):
        super(LSTM, self).__init__()
        self.relu_flag = relu_flag
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=hidden_units)
        self.dropout = nn.Dropout(drop)
        self.dense = nn.Linear(in_features=hidden_units, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor):
        # transpose to [seq_length, batch_size, n_features]
        x_d = x_d.transpose(0, 1)

        # print(x_s.shape, ' x_s')
        # concat all inputs
        if x_s.nelement() > 0:
            x_s = x_s.unsqueeze(0).repeat(x_d.shape[0], 1, 1)  # seq, batch, attributions.
            x_d = torch.cat([x_d, x_s], dim=-1)  # seq, batch, feature+attr
        else:
            pass

        lstm_output, (h_n, c_n) = self.lstm(input=x_d)

        # reshape to [batch_size, seq_length, n_hiddens]
        h_n = h_n.transpose(0, 1)
        c_n = c_n.transpose(0, 1)
        y_hat = self.dense(self.dropout(lstm_output.transpose(0, 1)))
        if self.relu_flag:
            y_hat = self.relu(y_hat)
        return y_hat, h_n, c_n

class LSTM_classifier(nn.Module):
    def __init__(self, hidden_units, input_size, drop=0.3):
        super(LSTM_classifier, self).__init__()
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=hidden_units)
        self.dropout = nn.Dropout(drop)
        self.dense = nn.Linear(in_features=hidden_units, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor):
        # transpose to [seq_length, batch_size, n_features]
        x_d = x_d.transpose(0, 1)

        # print(x_s.shape, ' x_s')
        # concat all inputs
        if x_s.nelement() > 0:
            x_s = x_s.unsqueeze(0).repeat(x_d.shape[0], 1, 1)  # seq, batch, attributions.
            x_d = torch.cat([x_d, x_s], dim=-1)  # seq, batch, feature+attr
        else:
            pass

        lstm_output, (h_n, c_n) = self.lstm(input=x_d)

        # reshape to [batch_size, seq_length, n_hiddens]
        h_n = h_n.transpose(0, 1)
        c_n = c_n.transpose(0, 1)
        y_hat = self.dense(self.dropout(lstm_output.transpose(0, 1)))
        # print(y_hat.shape)
        y_hat = self.sigmoid(y_hat) # probability.
        return y_hat, h_n, c_n