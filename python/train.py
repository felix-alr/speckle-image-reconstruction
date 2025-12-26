import types

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from python.test import compute_average_loss

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def train_network(model: nn.Module, data_loader: DataLoader, optimizer: optim.Optimizer, loss_fcn: nn.modules.loss._Loss, options: types.SimpleNamespace):
    optimizer.zero_grad()

    options.epochs = getattr(options, "epochs", 20)
    options.learning_rate = getattr(options, "learning_rate", 1e-2)
    options.validate = getattr(options, "validate", False)
    options.validation_data = getattr(options, "validation_data", None)
    options.validation_frequency = getattr(options, "validation_frequency", data_loader.batch_size - 1)

    options.validation_patience = getattr(options, "validation_patience", -1)




    model.train()
    device = next(model.parameters()).device

    iteration = 0

    lowest_loss = -1
    validation_patience_count = 0

    iter_train = []
    iter_valid = []
    data_train = []
    data_valid = []

    for epoch in range(options.epochs):
        print(f"Epoch: {epoch}/{options.epochs}")

        # Break if validation patience is exceeded
        if not options.validation_patience == -1 and validation_patience_count >= options.validation_patience:
            print("Training has been stopped! Criterion met!")
            break

        for batch, (x, y) in enumerate(data_loader):
            # Validate if criteria are met
            if options.validate and options.validation_data is not None and iteration % options.validation_frequency == 0:
                loss = compute_average_loss(model, options.validation_data, loss_fcn)
                # Set new lowest loss if it has not yet been set or it's lower than the previously lowest loss
                if lowest_loss == -1 or loss < lowest_loss:
                    lowest_loss = loss
                else:
                    # If loss >= lowest_loss, add to validation_patience_count
                    validation_patience_count += 1

                data_valid.append(loss)
                iter_valid.append(iteration)
                print(f"Validation Loss: {loss.item()}")
                model.train()


            x,y = x.to(device), y.to(device)

            pred = model(x)

            loss = loss_fcn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            data_train.append(loss.item())
            iter_train.append(iteration)

            iteration += 1
        if epoch == options.epochs:
            print("Training stopped! Reached maximum epoch amount!")

    fig, ax = plt.subplots()
    ax.plot(iter_train, data_train, label="Training Loss", color="blue")
    if len(data_valid) > 0:
        ax.plot(iter_valid, data_valid, label="Validation Loss", color="orange")
    ax.set_title("Training Progress")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (MSE)")
    ax.legend()
    ax.grid(True)
    plt.draw()
    plt.show()