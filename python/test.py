import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def compute_average_loss(model: nn.Module, test_loader: DataLoader, loss_fcn: nn.modules.loss._Loss):
    model.eval()
    loss = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x)
            loss += loss_fcn(pred, y)
            total += 1

    return loss/total