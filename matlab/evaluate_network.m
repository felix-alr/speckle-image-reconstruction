clearvars
clear

addpath("data\training\trained-nets\")
addpath("functions\")

load("data\DATA_Diff_16_aug.mat")

%% Evaluating mlp with and without data augmentation by plotting the loss curve during training
plotLoss("data\training\mlp-raw-data-info.mat");
%plotLoss("data\training\mlp-augmented-data-info.mat");

%% Evaluating mlp with and without data augmentation using the metrics RMSE, PSNR, SSIM, and cross correlation

labels = {'Raw Data', 'Augmented Data'};

netMLPRaw = load("mlp-raw-data-net.mat");
netMLPAug = load("mlp-augmented-data-net.mat");

predYMLPRaw = predict(netMLPRaw.net,XValid);
predYMLPAug = predict(netMLPAug.net,XValid);


rmseRaw = computeRMSE(predYMLPRaw, YValid);
rmseAug = computeRMSE(predYMLPAug, YValid);

subplot(2,2,1);
boxplot([rmseRaw(:), rmseAug(:)], 'Labels', labels);
ylabel("RMSE");
title("Comparison RMSE")


psnrRaw = computePSNR(predYMLPRaw, YValid);
psnrAug = computePSNR(predYMLPAug, YValid);

subplot(2,2,2);
boxplot([psnrRaw(:), psnrAug(:)], 'Labels', labels);
ylabel("PSNR");
title("Comparison PSNR")


ssimRaw = computeSSIM(predYMLPRaw, YValid);
ssimAug  = computeSSIM(predYMLPAug, YValid);

subplot(2,2,3);
boxplot([ssimRaw(:), ssimAug(:)], 'Labels', labels);
ylabel("SSIM");
title("Comparison SSIM")


corrRaw = computeCrossCorrelation(predYMLPRaw, YValid);
corrAug = computeCrossCorrelation(predYMLPAug, YValid);

subplot(2,2,4);
boxplot([corrRaw(:), corrAug(:)], 'Labels', labels);
ylabel("Cross Correlation");
title("Comparison Cross Correlation")
