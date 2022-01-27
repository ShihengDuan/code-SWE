import torch


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
