function accuracy = modifyModelwithClassesAlex(FullModel,convLayer,filterNum,imdsTrain,imdsValidation,modeWrite)
% FullModel = VGG16;
% convLayer = [2, 6];
% filterNum = [10 15];

netOpt = FullModel;

inputSize = netOpt.Layers(1).InputSize; 
numClasses = numel(categories(imdsTrain.Labels))
% numClasses = 10
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

%% modify


netOpt = FullModel;

inputSize = netOpt.Layers(1).InputSize; 
lgraph = layerGraph(netOpt.Layers);
%  figure
%  plot(lgraph)
%% read convolution layer

for i = 1:length(convLayer)
currentLayer = netOpt.Layers(convLayer(i));
lgraph=disconnectLayers(lgraph,netOpt.Layers(convLayer(i)-1).Name,netOpt.Layers(convLayer(i)).Name);
lgraph=disconnectLayers(lgraph,netOpt.Layers(convLayer(i)).Name,netOpt.Layers(convLayer(i)+1).Name);
lgraph = removeLayers(lgraph,netOpt.Layers(convLayer(i)).Name);
%% replace layer
numberFilt =filterNum(i);
if i>1
numberChannels = filterNum(i-1);
else
    numberChannels = 3;
end
newLayer = convolution2dLayer(currentLayer.FilterSize(1),numberFilt,'Stride',currentLayer.Stride(1),'Padding',currentLayer.PaddingSize(1),'Name',currentLayer.Name,'NumChannels',numberChannels);
newLayer.Weights = currentLayer.Weights(:,:,1:numberChannels,1:numberFilt);
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
%% fc layer
if i==length(convLayer)
    for j=1:length(netOpt.Layers)
    currentLayer = lgraph.Layers(j);
    tmpLayerName =char(["fc"+string(str2num(netOpt.Layers(convLayer(i)).Name(5))+1)]);
    if string(currentLayer.Name) == string(tmpLayerName)
        break
    end
    end
    upLayer = lgraph.Layers(j-1).Name;
    downLayer = lgraph.Layers(j+1).Name;
lgraph=disconnectLayers(lgraph,lgraph.Layers(j-1).Name,lgraph.Layers(j).Name);
lgraph=disconnectLayers(lgraph,lgraph.Layers(j).Name,lgraph.Layers(j+1).Name);
lgraph = removeLayers(lgraph,currentLayer.Name);

newLayer = fullyConnectedLayer(currentLayer.OutputSize,'Name',currentLayer.Name);
% InputSize=7*7*convLayer(i);
% newLayer.Weights = currentLayer.Weights(:,:,:,1:7*7*convLayer(i));
% newLayer.Bias = currentLayer.Bias;
newLayer.WeightLearnRateFactor = currentLayer.WeightLearnRateFactor;
newLayer.WeightL2Factor = currentLayer.WeightL2Factor;
newLayer.BiasLearnRateFactor = currentLayer.BiasLearnRateFactor;
newLayer.BiasL2Factor = currentLayer.BiasL2Factor;
lgraph = addLayers(lgraph,newLayer);
% figure
% plot(lgraph)
lgraph=connectLayers(lgraph,upLayer,newLayer.Name);
lgraph=connectLayers(lgraph,newLayer.Name,downLayer); 
% figure
% plot(lgraph)
end
end
%%

fcn = fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20,'Name','fc8');
sft = softmaxLayer('Name','softmax');
classi = classificationLayer('Name','Classification');

lgraph = removeLayers(lgraph,'fc8');
lgraph = removeLayers(lgraph,'prob');
lgraph = removeLayers(lgraph,'output');

lgraph = addLayers(lgraph,fcn);
lgraph = addLayers(lgraph,sft);
lgraph = addLayers(lgraph,classi);

lgraph=connectLayers(lgraph,'drop7','fc8');
lgraph=connectLayers(lgraph,'fc8','softmax');
lgraph=connectLayers(lgraph,'softmax','Classification');
% figure
% plot(lgraph)
options = trainingOptions('sgdm', ...
    'MiniBatchSize',64, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',50, ...
    'ValidationPatience',Inf, ...
    'Verbose',false, ...
    'Plots','training-progress');
disp('Retraining Model...!')
netOpt = trainNetwork(augimdsTrain,lgraph,options);
if modeWrite == 'save'
    disp('Saving Model...!')
save netOpt
pause(5)
disp('Model saved')
else
    disp('Model not saved')
end
%% predictions
YPred = classify(netOpt,augimdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation)
end
