function accuracy = modifyModel(FullModel,convLayer,filterNum,imdsTrain,imdsValidation,modeWrite)
% FullModel = alexnet;
% convLayer = [2, 6];
% filterNum = [10 15];

netOpt = FullModel;

inputSize = netOpt.Layers(1).InputSize; 
numClasses = numel(categories(imdsTrain.Labels))
% numClasses = 10
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);




lgraph = layerGraph(netOpt.Layers);
%  figure
%  plot(lgraph)

%% read convolution layer

for i = 1:length(convLayer)
currentLayer = netOpt.Layers(convLayer(i));
lgraph=disconnectLayers(lgraph,netOpt.Layers(convLayer(i)-1).Name,netOpt.Layers(convLayer(i)).Name);
lgraph=disconnectLayers(lgraph,netOpt.Layers(convLayer(i)).Name,netOpt.Layers(convLayer(i)+1).Name);
% figure
% plot(lgraph)
lgraph = removeLayers(lgraph,netOpt.Layers(convLayer(i)).Name);
%% replace layer
numberFilt =filterNum(i);
newLayer = convolution2dLayer(currentLayer.FilterSize(1),numberFilt,'Stride',currentLayer.Stride(1),'Padding',currentLayer.PaddingSize(1),'Name',currentLayer.Name);
newLayer.Weights = currentLayer.Weights(:,:,:,1:numberFilt);
newLayer.Bias = currentLayer.Bias(:,:,1:numberFilt);
newLayer.WeightLearnRateFactor = currentLayer.WeightLearnRateFactor;
newLayer.WeightL2Factor = currentLayer.WeightL2Factor;
newLayer.BiasLearnRateFactor = currentLayer.BiasLearnRateFactor;
newLayer.BiasL2Factor = currentLayer.BiasL2Factor;
% newLayer
 
lgraph = addLayers(lgraph,newLayer);
% figure
% plot(lgraph)
lgraph=connectLayers(lgraph,netOpt.Layers(convLayer(i)-1).Name,newLayer.Name);
lgraph=connectLayers(lgraph,newLayer.Name,netOpt.Layers(convLayer(i)+1).Name); 
% figure
% plot(lgraph)

end
lgraph = removeLayers(lgraph,'output');
lgraph = removeLayers(lgraph,'prob');
lgraph = removeLayers(lgraph,'fc8');
% figure
% plot(lgraph)
newLayer = fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20,'Name','fc8');
lgraph = addLayers(lgraph,newLayer);
newLayer = softmaxLayer('Name','softmax');
lgraph = addLayers(lgraph,newLayer);
newLayer = classificationLayer('Name','classification');
lgraph = addLayers(lgraph,newLayer);
lgraph=connectLayers(lgraph,'fc8','softmax');
lgraph=connectLayers(lgraph,'softmax','classification');

% figure
% plot(lgraph)
lgraph=connectLayers(lgraph,'drop7','fc8');
% figure
% plot(lgraph)
%  net1 = SeriesNetwork(lgraph)
% retModel = lgraph;
% options = trainingOptions('sgdm', ...
%     'MaxEpochs',5, ...
%     'ValidationData',augimdsValidation, ...
%     'ValidationFrequency',30, ...
%     'Verbose',false);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',16, ...
    'MaxEpochs',5, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',100, ...
    'ValidationPatience',Inf, ...
    'Verbose',false);%, ...
   % 'Plots','training-progress');

netOpt = trainNetwork(augimdsTrain,lgraph,options);
if modeWrite == 'save'
save netOpt.mat
disp('Model saved')
else
    disp('Model not saved')
end
%% predictions
YPred = classify(netOpt,augimdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation)
end
