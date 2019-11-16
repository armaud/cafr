function accuracy=Model(InSeq,imdsTrain,imdsValidation,modeWrite)

layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,InSeq(1),'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,InSeq(2),'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,InSeq(3),'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,InSeq(4),'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

%% training
% options = trainingOptions('sgdm', ...
%     'MaxEpochs',4, ...
%     'ValidationData',imdsValidation, ...
%     'ValidationFrequency',30, ...
%     'Verbose',false, ...
%     'Plots','training-progress');
options = trainingOptions('sgdm', ...
    'MaxEpochs',5, ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',100, ...
    'Verbose',false);

net = trainNetwork(imdsTrain,layers,options);
if modeWrite == 'save'
save net.mat
disp('Model saved')
else
    disp('Model not saved')
end
%% predictions
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation)
end