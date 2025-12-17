clearvars
clear

addpath("data\training\trained-nets\")
addpath("functions\")

load("data\Diff_PSF.mat")
load("data\DATA_Diff_16_aug.mat")

predictDrawing(load("unet-augmented-data-net.mat").net, PSF);
%predictDrawing(load("mlp-augmented-data-net.mat"), PSF);


%% Evaluating mlp with and without data augmentation by plotting the loss curve during training
%{
plotLoss("data\training\mlp-raw-data-info.mat");
plotLoss("data\training\mlp-augmented-data-info.mat");
plotLoss("data\training\unet-augmented-data-info.mat");


netMLPRaw = load("mlp-raw-data-net.mat");
netMLPAug = load("mlp-augmented-data-net.mat");
unetAug = load("unet-augmented-data-net.mat");
labelsMLP = {'Raw Data', 'Augmented Data'};
labelsMLPUNet = {'MLP', 'U-Net'};

compareMetrics(netMLPRaw, netMLPAug, labelsMLP, XValid, YValid);
compareMetrics(netMLPAug, unetAug, labelsMLPUNet, XValid, YValid);
%}