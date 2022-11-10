import torch
from torch.nn import L1Loss

class LossWithStd(torch.nn.Module):
    def __init__(self, eps=0.1):
        super(LossWithStd, self).__init__()

    def forward(self, y_hat, y, qstd):
        loss_with_std = torch.mean(qstd * (y_hat - y) ** 2)
        return loss_with_std


class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_hat, y):
        loss = torch.mean((y_hat - y) ** 2)
        return loss


class LossWithPenalty(torch.nn.Module):
    def __init__(self):
        super(LossWithPenalty, self).__init__()

    def forward(self, y_hat, y, ymin):
        loss = torch.mean((y_hat - y) ** 2)
        penalty = torch.mean(1 / (y - ymin + 0.1) * (y_hat - y) ** 2)
        return loss + penalty

class L12loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        l1loss = L1Loss()
    def forward(self, y_hat, y, ymin=0):
        ind = y>ymin
        L1 = torch.sum(torch.abs((y_hat-y)*(~ind)))
        L2 = torch.sum(torch.square((y_hat-y)*ind))
        loss = (L1+L2)/y_hat.nelement()
        return loss