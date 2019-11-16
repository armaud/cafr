function accuracy = callModel(inModel,imdsTrain,imdsValidation,mode)
%% transfer Learning alexnet algorithm
net = inModel;
net.Layers
inputSize = net.Layers(1).InputSize
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels))

%% last layer definition
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer]
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',64, ...
    'MaxEpochs',5, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',100, ...
    'ValidationPatience',Inf, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Training
net = trainNetwork(augimdsTrain,layers,options);
%% Save Model
if mode == 'save'
disp('saving model...!')
    save net
pause(5)
disp('model saved...!')
else
disp('model not saved...')
end
YPred = classify(net,augimdsValidation);
YTest = imdsValidation.Labels;
accuracy = sum(YPred == YTest)/numel(YTest);
end
