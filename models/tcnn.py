import torch
from torch import nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):  # causal padding
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernal_size, stride, dilation, padding, dropout=0.4):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernal_size, stride=stride,
                                           padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernal_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None  # res connection
        self.relu = nn.ReLU()
        # self.init_weights()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)  # res connection

    def init_weights(self):
        self.conv1.weight.data.uniform_(-0.1, 0.1)
        self.conv2.weight.data.uniform_(-0.1, 0.1)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)


class TCNN(nn.Module):
    def __init__(self, kernal_size=7, num_levels=3, num_channels=20, input_size=14):
        super(TCNN, self).__init__()
        self.kernal_size = kernal_size
        self.num_levels = num_levels
        self.num_channels = num_channels
        self.dr_rate = 0.4
        self.input_size = input_size
        layers = []

        for i in range(self.num_levels):
            dilation_size = 2 ** (i + 1)  # dilation rate with layer number
            # dilation_size = 6 * (i + 1)
            in_channels = self.input_size if i == 0 else self.num_channels
            out_channels = self.num_channels
            layers += [
                TemporalBlock(in_channels, out_channels, padding=(self.kernal_size - 1) * dilation_size, stride=1,
                              dilation=dilation_size,
                              dropout=self.dr_rate, kernal_size=self.kernal_size)]

        self.tcnn = nn.Sequential(*layers)

        self.dropout = nn.Dropout(p=self.dr_rate)
        self.dense = nn.Linear(in_features=num_channels, out_features=1)

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor):
        # transpose to [seq_length, batch_size, n_features]
        x_d = x_d.transpose(0, 1)
        # original: batch_size, length, features
        # to [batch_size, n_features, seq_length]

        # concat all inputs
        if x_s.nelement() > 0:
            x_s = x_s.unsqueeze(0).repeat(x_d.shape[0], 1, 1)  # seq, batch, attributions.
            x_d = torch.cat([x_d, x_s], dim=-1)  # seq, batch, feature+attr
        else:
            pass
        ## convert to CNN inputs:
        x_d = x_d.transpose(0, 1)
        x_d = x_d.transpose(1, 2)
        tcnn_out = self.tcnn(input=x_d)  # N, Channel, Seq
        tcnn_out = tcnn_out.transpose(1, 2)
        y_hat = self.dense(tcnn_out)

        return y_hat, tcnn_out, x_d  # keep the same form with LSTM's other two outputs
