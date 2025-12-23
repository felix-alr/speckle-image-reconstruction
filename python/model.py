import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, layer_sizes, activation_functions, reshape_output_fcn = lambda x: x):
        super().__init__()
        self.reshape_output_fcn = reshape_output_fcn
        self.flatten = nn.Flatten()
        # Array to construct model structure from layer_sizes and activation_functions
        layers = []
        # Defining activation functions for each layer according to activation_functions
        activation = [None] * (len(layer_sizes)-1)

        match len(activation_functions):
            case 1: # the same activation function for each layer
                activation = [activation_functions[0]] * (len(layer_sizes)-1)
            case 2: # first activation function for each hidden layer, second one for final layer
                activation = [activation_functions[0]] * (len(layer_sizes)-2) + [activation_functions[1]]
            case n if n == (len(layer_sizes)-1): # one activation function for each layer as defined in activation_functions
                activation = activation_functions
            case _: # No activation function for any layer
                pass

        for i, size in enumerate(layer_sizes):
            if i < len(layer_sizes)-1: # Add linear layer of desired size (according to layer_sizes)
                layers.append(nn.Linear(size, layer_sizes[i+1]))
                if activation[i] is not None:
                    layers.append(activation[i])

        self.layer_stack = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.layer_stack.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.flatten(x)
        pred = self.layer_stack(x)
        return self.reshape_output_fcn(pred)