function [result accuracyPersentage] = accuracyAlgorithm(modell,imdsTrain,imdsValidation,error,threshold,stepSize)
%% Call model
% % % accuracy = callModel(inModel,imdsTrain,imdsValidation,mode)
inModel = modell;
accFullModel=callModel(inModel,imdsTrain,imdsValidation,'save');
%accFullModel=0.8659;
%  accFullModel = 79.17;
%% Call optimization algo
% filters=optimizationAlgoAutowithWeights(threshold,stepSize,64); % 64 is the maximum channel depth for each layer

% load filters.mat
% load InSeq.mat
if ~exist('Retrainingfilters.mat')
    disp('Previous retraining filters not found...!')
    disp('Calling optimization function...!')
    filters = OptimizeWorking();
else
    disp('Previous Retraining Filters found...')
    disp('Loading Previous Retraining Filters...!')
load('Retrainingfilters.mat')    
    disp('Filters loaded...!')
end
% filters=[2 48;4 1;7 1; 9 3; 12 9; 14 3; 16 4; 19 12; 21 5; 23 20; 26 7; 28 26; 30 3]
InSeq=[];
%% calculate number of filters in orignal model
for xii=1:length(modell.Layers)
layer = modell.Layers(xii).Name;
    if length(layer)>3
        if layer(1:4) == 'conv'
            InSeq=[InSeq; sum(modell.Layers(xii).NumFilters)];
        end
    end
end
InSeq
filters
%%
pres =[];
%% call model with optimized Parameters
while(true)
% call network modeification algo


accOptModel=modifyModelwithClasses(inModel,filters(:,1),filters(:,2),imdsTrain,imdsValidation,'save')
accuracyDifferencePersentage=abs(accFullModel*100-accOptModel*100);
disp(['accurray % difference: ' string(accuracyDifferencePersentage)])
if accuracyDifferencePersentage <= error
    break
else
    
    %%

    for pre = 1:length(filters(:,2))
       ff1= double(filters(pre,2));
        ff2 = double(InSeq(pre));
    pres(pre) =  double((ff1/ff2)*100);
    end
    [valmin indmin]=min(pres)
    [valmax indmax]=max(pres);
    
    %%
    if valmin < 95
    filters(indmin,2) = filters(indmin,2)+ ceil(abs(InSeq(indmin)-filters(indmin,2))/40);
    end
%    if valmin > 80
%    filters(indmax,2) = filters(indmax,2)-1;
%    end
end
%%
disp(string(filters(:,2)'))
save('Retrainingfilters.mat','filters');
% nnet.trainParam.showWindow = 0;
%%
end
accuracyPersentage = accuracyDifferencePersentage;
result = [filters];

end
