import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def compute_average_loss(model: nn.Module, test_loader: DataLoader, loss_fcn: nn.modules.loss._Loss):
    """
    :param model: The model that shall be evaluated.
    :param test_loader: The test data the model shall be evaluated on.
    :param loss_fcn: The desired loss function for loss computation.
    :return: The average value of the loss of the mode predictions on the test data.
    """
    model.eval()
    loss = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x)
            loss += loss_fcn(pred, y)
            total += 1
    return loss/total