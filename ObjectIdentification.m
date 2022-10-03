%% Capture Training Images

captureTrainingImages('trainingImages', 'book')
captureTrainingImages('trainingImages', 'money')
captureTrainingImages('trainingImages', 'plate')

%% Load image data
imset = imageSet('trainingImages','recursive'); 

%% Pre-process Training Data: *Feature Extraction*
bag = bagOfFeatures(imset,'VocabularySize',200,...
    'PointSelection','Detector');

imagefeatures = encode(bag,imset);

%% Create a Table using the encoded features
Data         = array2table(imagefeatures);
Data.objectNmae = getImageLabels(imset);

%% Use the new features to train a model and assess its performance
classificationLearner

%% Test Trained Model
ObjectFinderLive(trainedModel.ClassificationKNN,bag)

%% Compare multiple classifiers
ObjectFinderLiveCompare(trainedModel.ClassificationKNN,trainedModel1.ClassificationSVM,bag)
