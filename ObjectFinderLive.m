function ObjectFinderLive(trainedClassifier,bag)
[fig, ax1, ax2] = figureSetup(trainedClassifier);

wcam = webcam;

while ishandle(fig)
    img = snapshot(wcam);
    grayimg = rgb2gray(img);
    
	imagefeatures = double(encode(bag,grayimg));
    
	[imagepred, probabilities] = predict(trainedClassifier,imagefeatures);
    
    try
        imshow(insertText(img,[640,1],upper(cellstr(imagepred)),...
            'AnchorPoint','RightTop','FontSize',50,'BoxColor','Green',...
            'BoxOpacity',0.4),'Parent',ax1);    
        ax2.Children.YData = probabilities;
        ax2.YLim = [0 1];
    catch err
    end
    drawnow
end 

function cname = getClassifierName(trainedClassifier)
cname = class(trainedClassifier);
if isa(trainedClassifier,'ClassificationECOC')
    cname = 'SVM';
end
pos = strfind(cname,'.');
if ~isempty(pos)
  cname = cname(pos(end)+1:end);
end

function [fig, ax1, ax2] = figureSetup(trainedClassifier)
warning('off','images:imshow:magnificationMustBeFitForDockedFigure')
set(0,'defaultfigurewindowstyle','docked')
fig = figure('Name','Object Finder Go!','NumberTitle','off');
ax1 = subplot(2,1,1);
ax2 = subplot(2,1,2);
bar(ax2,zeros(1,numel(trainedClassifier.ClassNames)),'FaceColor',[0.2 0.6 0.8])
set(ax2,'XTickLabel',cellstr(trainedClassifier.ClassNames));
title(getClassifierName(trainedClassifier)), ylabel('Probability')
set(0,'defaultfigurewindowstyle','normal')
