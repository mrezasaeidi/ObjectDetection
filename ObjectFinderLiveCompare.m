function ObjectFinderLiveCompare(trainedClassifier1,trainedClassifier2,bag)

[fig, ax] = figureSetup;

wcam = webcam;

while ishandle(fig)
    img = snapshot(wcam);
    grayimg = rgb2gray(img);
    
	imagefeatures = double(encode(bag,grayimg));
    
    imagepred1 = predict(trainedClassifier1,imagefeatures);
	imagepred2 = predict(trainedClassifier2,imagefeatures);
    
    try
        PredName1 = [getClassifierName(trainedClassifier1),':',upper(char(imagepred1))];
        PredName2 = [getClassifierName(trainedClassifier2),':',upper(char(imagepred2))];
        
        im = insertText(img,[640,480],PredName1,...
            'AnchorPoint','RightBottom','FontSize',30,'BoxColor','Green',...
            'BoxOpacity',0.4);  
        imshow(insertText(im,[1,1],PredName2,...
            'AnchorPoint','LeftTop','FontSize',30,'BoxColor','Red',...
            'BoxOpacity',0.4),'Parent',ax);  
        title('Compare Classifiers')
    catch err
    end
    drawnow
end 

function cname = getClassifierName(trainedClassifier)
    cname = class(trainedClassifier);
	if isa(trainedClassifier,'ClassificationECOC')
        cname = 'SVM';
    end
    if isa(trainedClassifier,'ClassificationKNN')
        cname = 'KNN';
    end
    pos = strfind(cname,'.');
    if ~isempty(pos)
      cname = cname(pos(end)+1:end);
    end

function [fig, ax] = figureSetup
    warning('off','images:imshow:magnificationMustBeFitForDockedFigure')
    fig = figure('Name','Object Finder Go!','NumberTitle','off');
    ax = axes;
