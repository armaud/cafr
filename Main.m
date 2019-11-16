clc, close all, clear all

warning off parallel:gpu:device:DeviceLibsNeedsRecompiling
try
    gpuArray.eye(2)^2;
catch ME
end
try
    nnet.internal.cnngpu.reluForward(1);
catch ME
end

% digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
%     'nndatasets','DigitDataset');
imds = imageDatastore('DatasetVOC2012/', ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
% 
% imds = imageDatastore(digitDatasetPath, ...
%     'IncludeSubfolders',true,'LabelSource','foldernames');
%% resize images


%%
figure;
perm = randperm(10000,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end

labelCount = countEachLabel(imds);
img = readimage(imds,1);

% numTrainFiles = round(length(imds.Labels)*0.75/10);
[imdsTrain,imdsValidation test] = splitEachLabel(imds,0.3,0.05,0.65,'randomize');
%% describe model and error

model = vgg19;
% model = alexnet;

error = 2;
%% accuracy file

[filters accuracyDiffPercentage]= accuracyAlgorithm(model,imdsTrain,imdsValidation,error,0.65,0.05)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% calculation of parameters in convolutional layers
TotalParameters = 0;
load net
for i = 1:length(net.Layers)
if (isprop (net.Layers(i),'Weights') && isprop (net.Layers(i),'Bias'))
    TotalParameters = TotalParameters + prod(size(net.Layers(i).Weights))+prod(size(net.Layers(i).Bias));
end
end
ReducedParameters=0;
load netOpt
for i = 1:length(netOpt.Layers.Layers)
if (isprop (net.Layers.Layers(i),'Weights') && isprop (net.Layers.Layers(i),'Bias'))
    ReducedParameters = ReducedParameters + prod(size(net.Layers.Layers(i).Weights))+prod(size(net.Layers.Layers(i).Bias));
end
end
save 'TotalParameters.mat'
save 'Reducedparameters.mat'
save 'accuracyDiffPercentage.mat'
disp(['Size of convolutional learnable parameters are reduced to: ' num2str(100-(ReducedParameters/TotalParameters)*100,6) '%'])
disp(['of total learnable parameters in convolutional layers with'])
disp(['the percentage difference of: ' num2str(accuracyDiffPercentage,6) '%'])

