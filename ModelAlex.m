function accuracy=ModelAlex(InSeq,rgb,imdsTrain,imdsValidation,modeWrite)
% InSeq = [96 ,256 ,384 ,384 ,256]
layers = [
    imageInputLayer([227 227 rgb],'name','data')
    %% 1
    convolution2dLayer([11 11],InSeq(1),'Stride', [4 4],'Padding',[0 0 0 0])
    reluLayer
    crossChannelNormalizationLayer(5)
    maxPooling2dLayer([3 3],'Stride',[2 2],'Padding', [0 0 0 0])
    %% 2
    convolution2dLayer([5 5],InSeq(2),'Stride', [1 1],'Padding',[2 2 2 2])
    reluLayer
    crossChannelNormalizationLayer(5)
    maxPooling2dLayer([3 3],'Stride',[2 2],'Padding', [0 0 0 0])
    %% 3
    convolution2dLayer([3 3],InSeq(3),'Stride', [1 1],'Padding',[1 1 1 1])
    reluLayer
    %% 4
    convolution2dLayer([3 3],InSeq(3),'Stride', [1 1],'Padding',[1 1 1 1])
    reluLayer
    %% 5
    convolution2dLayer([3 3],InSeq(3),'Stride', [1 1],'Padding',[1 1 1 1])
    reluLayer
    maxPooling2dLayer([3 3],'Stride',[2 2],'Padding', [0 0 0 0])
    %% Fully connected
    fullyConnectedLayer(4096)
    reluLayer
    dropoutLayer(0.50)
    %%%%
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
    ];

%% training

% options = trainingOptions('sgdm', ...
%     'MaxEpochs',4, ...
%     'ValidationData',imdsValidation, ...
%     'ValidationFrequency',30, ...
%     'Verbose',false);

%% Resize images for training and validation
augimdsTrain = augmentedImageDatastore([227 227],imdsTrain);
augimdsValidation = augmentedImageDatastore([227 227],imdsValidation);
%%
options = trainingOptions('sgdm', ...
    'MaxEpochs',4, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(augimdsTrain,layers,options);
if modeWrite == 'save'
save net.mat
disp('Model saved')
else
    disp('Model not saved')
end
%% predictions
YPred = classify(net,augimdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation)
end