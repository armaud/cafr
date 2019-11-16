function resultAll = OptimizeWorking()
%% optimized code
% clear all,close all,clc
array = [0:0.001:1];
array1=array'*1000;
threshold=0;
array1=sort(unique(sort(bin2dec(dec2bin(array1,10))/1000,'descend')'),'descend');
corrMat=[];
diffMat=[];
latch=0;

% [row column]=find(unique(sort(bin2dec(dec2bin(array1,10))/100,'ascend')')==x)

% function filts=optimizationAlgoAutowithWeightsMultiVal(threshold,stepSize,depth)
% threshold = 0.5
% stepSize = 0.05
%% optimization algorithm
minFilters=[];
% disp(['Initial threshold is taken as: ' num2str(threshold,6) ' with step size of: ' num2str(stepSize,6)])
newStep=0;
warning off
disp("Loading Network...")
load ('net.mat');
% clearvars -except net threshold stepSize newStep minFilters depth
disp("Network loaded!")
net.Layers;
if class(net) == 'SeriesNetwork'
    disp('Series Network Found!')
Ca=[];
meanCa=0;
% for ii=1:15
%% find number of convolutional layers
disp('Finding number of convolution layers...')
convCounter=0;
for xii=1:length(net.Layers)
layer = net.Layers(xii).Name;
    if length(layer)>3
        if layer(1:4) == 'conv'
            convCounter = convCounter+1;
        end
    end
end

corrMat=zeros(convCounter,length(array1));
layerNumbers=[];
disp([num2str(convCounter) ' Convolution layers found!'])
%% optimize threshold
disp('Optimizing Network...')
%%
if ~exist('corrMat.mat')
   istarter = 1;
   growR=0;
else
    load('corrMat.mat');
    load('layerNumbers.mat');
    istarter = layerNumbers(length(layerNumbers))+1
    growR = length(layerNumbers)
end
% while(threshold <= 1)
% for recursiveIterations = 1:5
    Ca=[];
    
    layerDiffCounter = -1;
for ii=istarter:length(net.Layers)
    layer = net.Layers(ii).Name;
    if length(layer)>3
        if layer(1:4) == 'conv'
        ii;
        growR=growR+1
        layerNumbers=[layerNumbers;ii]
%         numf =net.Layers(ii).NumFilters
        disp(['Loading Feature Maps from: '  layer '...!'])
%         I = deepDreamImage(net,layer,numf, ...
%         'PyramidLevels',1, ...
%         'Verbose',0);
%          if (depth>0 && depth<= sum(net.Layers(ii).NumFilters))
%             if (depth>0 && depth<= sum(net.Layers(ii).NumChannels))
%              I = net.Layers(ii).Weights(:,:,depth,depth);
%             else
%                 I = net.Layers(ii).Weights(:,:,:,depth);
%             end   
%          else
%             disp('in else')

            I = net.Layers(ii).Weights;
      
%          end
            [r c l f]=size(I);
        disp(['Computing Feature Maps from: '  layer '...!'])
        layerDiffCounter = layerDiffCounter + 1;
%         threshold * (convCounter/(convCounter+layerDiffCounter))
        val=[];
            for j=1:round(f/2)
                for i = 1:round(f/2)
                    if i~=j
             
                        for k=1:l
                            
                                val=[val abs([corr2(I(:,:,k,j),I(:,:,k,i))]) abs([corr2(I(:,:,k,abs(f-j)),I(:,:,k,abs(f-i)))]) abs([corr2(I(:,:,k,abs(f-j)),I(:,:,k,abs(i)))]) abs([corr2(I(:,:,k,abs(j)),I(:,:,k,abs(f-i)))])];
                            
                            [row column]=find(array1==bin2dec(dec2bin(max(val)*1000,10))/1000);
                            corrMat(growR,column)=corrMat(growR,column)+1;
                        end
                    end
                    val=[];
                end
            [j*2,i*2,k]    
            end
            save('corrMat.mat','corrMat');
            save('layerNumbers.mat','layerNumbers')
            disp('saved processed layers!')
            end
        val=[];
        I=[];
        end
        
    end    

% findUnique = unique(Ca);
% computeResults = [];
% for i = 1:length(findUnique)
%     filttval=rem(sum(net.Layers(findUnique(i)).NumFilters)-length(find(Ca==findUnique(i)))/2,sum(net.Layers(findUnique(i)).NumFilters));
%      computeResults = [computeResults; filttval];
% end
%%

%% total number to be changed into the network
corrMat=corrMat/2;
%%
% hold on
diffMat=[];
disp('Calculating differentials...!')
for i=1:length(layerNumbers)
% diffMat=[diffMat;(rem(sum(net.Layers(layerNumbers(i)).NumFilters)-corrMat(i,:),sum(net.Layers(layerNumbers(i)).NumFilters)))];
diffMat=[diffMat;sum(net.Layers(layerNumbers(i)).NumFilters)-corrMat(i,:)];%,sum(net.Layers(layerNumbers(i)).NumFilters)))];
end
size(diffMat)
diffMat=int16(diffMat);
% plot(diffMat)
% diffMat1=[];
% B = 1/3*ones(3,1);
% hold on
% for i=1:length(diffMat(:,1))
% diffMat1(i,:) = filter(B,1,diffMat(i,:));
% plot(diffMat1(i,:))
% end
% hold off

% diffMat1=int16(diffMat1);

result=diffMat(:,end);
C=zeros([length(diffMat(:,1)),1]);
for i=1:-0.001:0
[row column]=find(array1==bin2dec(dec2bin(max(i)*1000,10))/1000);
for j=1:length(diffMat(:,1))
if (diffMat(j,column)>0)
result(j) = min(result(j), diffMat(j,column));
% [rr,cc]=find(array1==diffMat(j,column),1)
% c(j)=cc
% pause(0.01)
end

% pause(0.1)
end

end
for i=1:size(result,1)
threshold =threshold + min(array1(find(diffMat(i,:)==result(i))));
end
threshold=threshold/i;
disp(['Optimal threshold found at: ' num2str(threshold)])
disp('Computed filters in each layer are:')
disp ([' layer  ' 'Filters'])
save('Filters.mat','result');
disp([layerNumbers result])
% filts=[findUnique minFilters];
resultAll = [layerNumbers result];
% save VggOpt
else 
    disp ('Model is not Series!')
    disp ('Cannot Optimize Network!')
end
end


