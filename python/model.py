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

class UNetRegression(nn.Module):
    def __init__(self, input_channels = 1, output_channels = 1, features=[64, 128, 256, 512]):
        super().__init__()
        self.features = features

        self.enc = nn.ModuleList()
        self.upsampling = nn.ModuleList()
        self.dec = nn.ModuleList()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._convolution_block(features[-1], features[-1]*2)

        self.final_convolution = nn.Conv2d(features[0], output_channels, kernel_size=1)

        # Encoder block
        c_in = input_channels
        for feature in features:
            self.enc.append(self._convolution_block(c_in, feature))
            c_in = feature

        # Decoder block
        for feature in reversed(features):
            self.upsampling.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.dec.append(self._convolution_block(feature*2, feature))

    def forward(self, x):
        skip = []

        # Encoder block
        prev_out = x
        for encoder in self.enc:#
            prev_out = encoder(prev_out)
            skip.append(prev_out)
            prev_out = self.maxpool(prev_out)

        # Bottleneck
        prev_out = self.bottleneck(prev_out)

        # Reverse skip connections
        skip = skip[::-1]

        # Decoder block
        for i in range(0, len(self.dec), 1):
            ups = self.upsampling[i](prev_out)

            # Resize upsampled image to match size of skip connection
            if ups.shape != skip[i].shape:
                ups = nn.functional.interpolate(ups, size=skip[i].shape[2:], mode="bilinear", align_corners=True)

            # Concatenating skip connection and upsampling and computing a decode step
            prev_out = torch.cat([ups, skip[i]], dim=1)
            prev_out = self.dec[i](prev_out)

        # Computing the final convolution
        return self.final_convolution(prev_out)

    def _convolution_block(self, c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )