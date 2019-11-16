function filts=optimizationAlgoAuto(threshold,stepSize)
% threshold = 0.5
% stepSize = 0.05
%% optimization algorithm
minFilters=[];
disp(['Initial threshold is taken as: ' num2str(threshold,6) ' with step size of: ' num2str(stepSize,6)])
newStep=0;
warning off
disp("Loading Network...")
load ('net.mat');
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
disp([num2str(convCounter) ' Convolution layers found!'])
%% optimize threshold
disp('Optimizing Network...')
while(threshold <= 1)
for recursiveIterations = 1:5
    Ca=[];
    layerDiffCounter = -1;
for ii=1:length(net.Layers)
    layer = net.Layers(ii).Name;
    if length(layer)>3
        if layer(1:4) == 'conv'
        ii;
        numf = 1:net.Layers(ii).NumFilters;
        
%         if ~isempty (gpuDevice(1))
%             disp('Clearing GPU memory to load new feature maps...')
%         reset(gpuDevice(1))
%         end
        disp(['Loading Feature Maps from: '  layer '...!'])
        I = deepDreamImage(net,layer,numf, ...
        'PyramidLevels',1, ...
        'Verbose',0,'ExecutionEnvironment','cpu');
        [r c l f]=size(I);
        disp(['Computing Feature Maps from: '  layer '...!'])
        layerDiffCounter = layerDiffCounter + 1;
%         threshold * (convCounter/(convCounter+layerDiffCounter))
        val=[];
            for j=1:f
                for i = 1:f
                    if i~=j
                        for k=1:l
                        val=[val abs([corr2(I(:,:,k,j),I(:,:,k,i))])];
                        end
                    if abs(max(val)) >= threshold * (convCounter/(convCounter+layerDiffCounter))
                    Ca=[Ca;ii i j abs(max(val))];
                    end
                    val=[];
                    end
%                  [ i, j ]   
                end
                
            end
        val;
        val=[];
        end
        
    end
    
end
% if ~isempty (gpuDevice(1))
% reset(gpudev)
% end
%%
findUnique = unique(Ca(:,1));
computeResults = [];
for i = 1:length(findUnique)
    filttval=rem(sum(net.Layers(findUnique(i)).NumFilters)-length(find(Ca(:,1)==findUnique(i)))/2,sum(net.Layers(findUnique(i)).NumFilters));
     computeResults = [computeResults; filttval];
end


%% total number to be changed into the network


if (length(computeResults(computeResults<1))<=0)
if recursiveIterations ==1
   minFilters = computeResults;
else
    if (length(computeResults(computeResults<=2))<=0)
    for i=1:length(computeResults)
    if computeResults(i)>3
        minFilters(i) = min(minFilters(i),computeResults(i));
    end
    end
    end
end
else
    
    if length(computeResults(computeResults<=2)<=0)
    newStep  = ((length(computeResults(computeResults<=2))/convCounter))*stepSize;
    end
    threshold = threshold + newStep;
    recursiveIterations =1;
    disp(char(['Increasing threshold as: ' num2str(threshold,6) ' with step size: ' num2str(newStep,6)]));
    break
end
end
if  ~isempty(minFilters)
for cont =1:length(minFilters)
    if minFilters(cont)<=2 
        if net.Layers(findUnique(cont)).NumFilters <  minFilters(cont)
        minFilters(cont) = net.Layers(findUnique(cont)).NumFilters
        else
            minFilters(cont) = 3;
        end
    end
end
disp(['Optimal threshold found at: ' num2str(threshold)])
disp('Computed filters in each layer are:')
disp ([' layer     ' 'Filters'])
disp([findUnique minFilters])
filts=[findUnique minFilters];
threshold =1.1;
else
    filts=[];
end

end
else 
    disp ('Model is not Series!')
    disp ('Cannot Optimize Network!')
end
end