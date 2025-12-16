% Training of neural network for image reconstruction of digits propagated 
% through a scattering medium

clearvars
clear

rng(0);

addpath("data\")
addpath("functions\")

taskNames = ["mlp-raw-data", "mlp-augmented-data", "unet-augmented-data"];
task = 2;



%% Load training data
if (task == 1)
    load("DATA_Diff_16.mat")
elseif (task == 2 || task == 3)
    load("DATA_Diff_16_aug.mat")
end

%% Create Neural Network Layergraph MLP
I_px = 16;
O_px = 16;

layers = [];

if (task <= 2)
    layers = [imageInputLayer([I_px I_px 1]),
            fullyConnectedLayer(I_px^2),
            reluLayer(),
            fullyConnectedLayer(O_px^2),
            reluLayer(),
            depthToSpace2dLayer([O_px O_px]),
            regressionLayer()
            ];
else
    layers = unetLayers([I_px I_px 1], 2, 'encoderDepth',3);
    finalConvLayer = convolution2dLayer(1,1,'Padding','same','Stride',1,'Name', 'Final-ConvolutionLayer');
    layers = replaceLayer(layers, 'Final-ConvolutionLayer', finalConvLayer);
    layers = removeLayers(layers,'Softmax-Layer');
    regLayer = regressionLayer('Name','Reg-Layer');
    layers = replaceLayer(layers, 'Segmentation-Layer',regLayer);
    layers = connectLayers(layers, 'Final-ConvolutionLayer','Reg-Layer');
end

%% Training network
% define "trainingOptions"
options = trainingOptions("adam",Plots="training-progress", ExecutionEnvironment="auto");
options.MaxEpochs = 500;
options.InitialLearnRate = 0.001;
options.MiniBatchSize = 130;
options.ValidationData = {XValid, YValid};
options.ValidationFrequency = 30;
options.ValidationPatience = 10;

lossFcn = "crossentropy";

% training using "trainNetwork"
tic;
[net, info] = trainNetwork(XTrain, YTrain, layers, options);
trainTime = toc;

save("data\training\"+taskNames(task)+"-info.mat", "info");
save("data\training\trained-nets\"+taskNames(task)+"-net.mat", "net");
save("data\training\"+taskNames(task)+"-train-time.mat", "trainTime");

%% Calculate Prediction 
% use command "predict"

%% Evaluate Network
% calculate RMSE, Correlation, SSIM, PSNR

%% Boxplots for step 6 of instructions

%% Step 7: create Neural Network Layergraph U-Net
% Layers = [];

%% Boxplots for step 8 of instructions
