% Training of neural network for image reconstruction of digits propagated 
% through a scattering medium

clearvars

%% Load training data
load("DATA_Diff_16.mat")

%% Create Neural Network Layergraph MLP
I_px = 16;
O_px = 16;

layers = [imageInputLayer([I_px I_px 1]),
fullyConnectedLayer(I_px^2),
reluLayer(),
fullyConnectedLayer(O_px^2),
reluLayer(),
depthToSpace2dLayer([O_px O_px]),
regressionLayer()
];

%% Training network
% define "trainingOptions"
options = trainingOptions("adam",Plots="training-progress", ExecutionEnvironment="auto");
options.MaxEpochs = 100;
options.InitialLearnRate = 0.01;
options.MiniBatchSize = 100;
options.ValidationData = {XValid, YValid};
options.ValidationPatience = 20;

lossFcn = "crossentropy";

% training using "trainNetwork"
[net, info] = trainNetwork(XTrain, YTrain, layers, options);

%% Calculate Prediction 
% use command "predict"

%% Evaluate Network
% calculate RMSE, Correlation, SSIM, PSNR

%% Boxplots for step 6 of instructions

%% Step 7: create Neural Network Layergraph U-Net
% Layers = [];

%% Boxplots for step 8 of instructions
