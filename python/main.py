import types

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np

import h5py
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tkinter as tk

import python.data as data

from python import data
from python.gui import DrawingApp
from python.model import MLP
from python.test import compute_average_loss
from python.train import train_network

train = True
save_model = True

# Detect device
device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

kernel = torch.tensor(np.array(h5py.File("data/kernel.mat")['PSF']))

# Create instance of NeuralNetwork model
imgx = 16 # Image size
model = MLP([imgx**2, imgx**2, imgx**2], [nn.ReLU()], lambda x: x.reshape(-1, 1, imgx, imgx)).to(device)



if __name__ == '__main__':
    # Load datasets
    training_data = data.MatlabDataset("data/DATA_Diff_16.mat")
    test_data = data.MatlabDataset("data/DATA_Diff_16.mat", train=False)

    train_loader = DataLoader(training_data, batch_size=130, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=100, shuffle=False, num_workers=0)

    if train:
        # Define objects for training
        options = types.SimpleNamespace(learning_rate=1e-3)
        options.validate = True
        options.validation_data = test_loader
        options.validate_after_iterations = 50
        options.validation_patience = 100
        options.epochs = 500

        opt_adam = optim.Adam(model.parameters(), lr=options.learning_rate)

        loss = nn.MSELoss()

        train_network(model, train_loader, opt_adam, loss, options)

        if save_model:
            torch.save(model.state_dict(), "mlp_model.pth")
    else:
        model.load_state_dict(torch.load("mlp_model.pth", map_location=device))
        root = tk.Tk()
        app = DrawingApp(root, model, kernel)
        root.mainloop()