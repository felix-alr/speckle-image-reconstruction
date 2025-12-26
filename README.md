# Handwritten Digit Scattered Image Reconstruction
In this university project I was provided with a dataset comprising images of handwritten digits as targets and their diffused versions as input features.
The diffusion was achieved by convolving the target images with a given point spread function.

The objective of this project is to train both a multilayer perceptron (MLP) and a U-Net to in order to reconstruct the digit images from their diffused counterpart.
The resulting loss curves and performance metrics are then analyzed and compared to evaluate and compare their reconstruction performance.

## Implementation
As can be seen in the project folders, I have implemented the tasks both in MATLAB and Python to fulfill the requirements of the course in MATLAB but also learn to use pytorch to achieve the same results.<br/>
After cloning the repository, both the MATLAB and the Python code can be used for reconstruction. The MATLAB models are provided under `matlab/data/training/trained-nets/`, the Python models can be accessed under `python/data/training/trained-nets`. Plots have only been exported after training in MATLAB as required in the university course.

## MATLAB
### Using trained models
In order to use the model trained using MATLAB, you may find the trained network parameters under matlab/data/training/trained-nets.
When imported using e.g. `net = load('matlab/data/training/trained-nets/unet-augmented-data-net.mat').net;` the network can immediately be used for reconstruction by using `predict(net, inputFeatures)`.
For the MLP, the following model structure was used:
```matlab
layers = [imageInputLayer([I_px I_px 1]),
            fullyConnectedLayer(I_px^2),
            reluLayer(),
            fullyConnectedLayer(O_px^2),
            reluLayer(),
            depthToSpace2dLayer([O_px O_px]),
            regressionLayer()
            ];
```
The U-Net has been implemented as follows:
```matlab
layers = unetLayers([I_px I_px 1], 2, 'encoderDepth',3);
    finalConvLayer = convolution2dLayer(1,1,'Padding','same','Stride',1,'Name', 'Final-ConvolutionLayer');
    layers = replaceLayer(layers, 'Final-ConvolutionLayer', finalConvLayer);
    layers = removeLayers(layers,'Softmax-Layer');
    regLayer = regressionLayer('Name','Reg-Layer');
    layers = replaceLayer(layers, 'Segmentation-Layer',regLayer);
    layers = connectLayers(layers, 'Final-ConvolutionLayer','Reg-Layer');
```
For further examples you may have a look at `matlab/evaluate_networks.m` and `matlab/functions/*.m`as those files comprise the code for evaluating and thus reconstructing images using the models.
### Executing the demo program
The code for the demo is located withing `matlab/functions/predictDrawing.m` and can be executed by runnin `matlab/evaluate_networks.m`. You may want to comment out the code for generating plots beforehand which can be done by commenting out or removing all the code after line 12. Running the script will the open a drawing tool with a button `Predict` and `Clear`. Clicking predict will then open a figure showing the drawn image, the diffused image that has been passed to the model as well as the image reconstructed by the model.

## Python
Using the Python-Model may be achieved as follows:
1. Open `main.py`in an IDE.
