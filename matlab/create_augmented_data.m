%% Load non augmented training data

addpath("data\")

load('DATA_Diff_16.mat');           % load data sets
load('Diff_PSF.mat');               % load diffuser PSF
N = size(XTrain,4);                 % number of training images
r = size(XTrain,1);                 % resolution of training images=[r,r]

%% New training data
XTrain_aug = zeros(r,r,1,2*N);
YTrain_aug = zeros(r,r,1,2*N);

%% Data Augmentation (2 new images per training image)
for i1=1:N
    original_image = YTrain(:,:,:,i1);
    
    % Data Augmentation 1 (you can use more than one way of augmenting on
    % one image)
    % aug_image = function(original_image); 
    aug_image = imtranslate(original_image, [randi([-8,8]), randi([-8,8])]);
    aug_image = imrotate(aug_image, randi([1,560]), 'nearest', 'crop');
    
    [XTrain_aug(:,:,:,i1), YTrain_aug(:,:,:,i1)] = deal(conv2(aug_image,PSF,'same'), aug_image);
    % Data Augmentation 2
    % aug_image = function(original_image);
    aug_image = circshift(original_image, [randi([-8,8]), randi([-8,8])]);
    aug_image = imrotate(aug_image, randi([1,360]), 'nearest', 'crop');
    
    [XTrain_aug(:,:,:,N+i1), YTrain_aug(:,:,:,N+i1)] = deal(conv2(aug_image,PSF,'same'), aug_image);
    
    disp([num2str(i1) '/' num2str(N)]);
end


%% Save Augmented Training Data
XTrain = cat(4,XTrain,XTrain_aug);
YTrain = cat(4,YTrain,YTrain_aug);
save('data\DATA_Diff_16_aug.mat','XTrain','YTrain','XValid','YValid');