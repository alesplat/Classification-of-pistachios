%% Loading images
path_images = "C:\Users\xseri\Downloads\Pistachio_Image_Dataset\Pistachio_Image_Dataset\Pistachio_Image_Dataset";
image_datastore = imageDatastore(path_images, "IncludeSubfolders",true,"LabelSource","foldernames");

%% Splitting dataset
[train, validation, testset] = splitEachLabel(image_datastore,0.6, 0.20, 0.20, 'randomized');

%% Testing phase and confusion matrix
aug_test = augmentedImageDatastore([224 224 3], testset);
[pred_test_labels, pred_test_scores] = classify(trainedNetwork_1, aug_test);
true_test_labels = testset.Labels;
accuracy_test = mean(true_test_labels == pred_test_labels);
plotconfusion(true_test_labels, pred_test_labels)


%% Cross-validation
[traincv, testcv] = splitEachLabel(image_datastore,0.7, 0.30, 'randomized');
numFolds = 5;
cv = cvpartition(traincv.Labels,'Kfold',numFolds);
network_architecture;
imageAugmenter = imageDataAugmenter(...
    'RandRotation', [-30,30], ...
    'RandXReflection', 1);
augmented_train = augmentedImageDatastore([224 224 3], traincv,  'DataAugmentation', imageAugmenter);
augmented_test = augmentedImageDatastore([224 224 3], testcv);

nets = [];
accuracies = [];
for i=1:numFolds
    idx_train = training(cv,i);
    idx_valid = test(cv,i);
    train = subset(augmented_train,idx_train);
    valid = subset(augmented_train,idx_valid);
    valid_label = subset(traincv,idx_valid);
    opts = trainingOptions("sgdm",...
        "ExecutionEnvironment","auto",...
        "InitialLearnRate",0.01,...
        "MaxEpochs",2,...
        "MiniBatchSize",64,...
        "Shuffle","every-epoch",...
        "ValidationFrequency",50,...
        "Plots","training-progress",...
        "Momentum",0.9,...
        "ValidationData",valid);
    [network, traininfo] = trainNetwork(train,lgraph,opts);
    true_valid_labels =  valid_label.Labels; 
    pred_valid_labels = classify(network, valid);
    accuracy = mean(true_valid_labels==pred_valid_labels);
    accuracies = [accuracies, accuracy];
    nets = [nets, network];
end

%% test cross-validation
avg = sum(accuracies)/length(accuracies);
[max_acc,indx_max] = max(accuracies);
cv_net = nets(indx_max);
true_test_labels = testcv.Labels;
pred_test_labels = classify(cv_net,augmented_test);
accuracy_test_cv = mean(true_test_labels == pred_test_labels);
plotconfusion(true_test_labels, pred_test_labels);

%% REAL MISCLASSIFICATION ERRORS ON KIRMIZI
wronglyPredicted = find(pred_test_labels~= test.Labels);
test.Files{wronglyPredicted}
 h = [];
 s = [];
 h(1) = subplot(2,2,1);
 h(2) = subplot(2,2,2);
 h(3) = subplot(2,2,3);

 s(1) = subplot(5,3,1);
 s(2) = subplot(5,3,2);
 s(3) = subplot(5,3,3);
 s(4) = subplot(5,3,4);
 s(5) = subplot(5,3,5);
 s(6) = subplot(5,3,6);
 s(7) = subplot(5,3,7);
 s(8) = subplot(5,3,8);
 s(9) = subplot(5,3,9);
 s(10) = subplot(5,3,10);
 s(11) = subplot(5,3,11);
 s(12) = subplot(5,3,12);
 s(13) = subplot(5,3,13);
 s(14) = subplot(5,3,14);
 s(15) = subplot(5,3,15);


for h1 = 1:3
    C = imread(test.Files{wronglyPredicted(h1)});
    image(C,'Parent',s(h1));
    image(C,'Parent',s(h1));
    image(C,'Parent',s(h1));


   
end


%% Kirmizi most like Kirmizi
chosenClass = "Kirmizi_Pistachio";
classIdx = find(trainedNetwork_1.Layers(end).Classes == chosenClass);

numImgsToShow = 9;

[sortedScores,imgIdx] = findMaxActivatingImages(test,chosenClass,pred_test_scores,numImgsToShow);

figure
plotImages(test,imgIdx,sortedScores,pred_test_labels,numImgsToShow)


%% Activation map Kirmizi
%imageNumber = imread(test.Files{wronglyPredicted(1)});
imageNumber = 1;

observation = aug_test.readByIndex(imgIdx(imageNumber));
img = observation.input{1};

label = pred_test_labels(imgIdx(imageNumber));
score = sortedScores(imageNumber);

gradcamMap = gradCAM(trainedNetwork_1,img,label);

figure
alpha = 0.5;
plotGradCAM(img,gradcamMap,alpha);
sgtitle(string(label)+" (score: "+ max(score)+")")

%% Siirt most like Siirt
chosenClass = "Siirt_Pistachio";
classIdx = find(trainedNetwork_1.Layers(end).Classes == chosenClass);

numImgsToShow = 9;

[sortedScores,imgIdx] = findMaxActivatingImages(test,chosenClass,pred_test_scores,numImgsToShow);

figure
plotImages(test,imgIdx,sortedScores,pred_test_labels,numImgsToShow)

%% Activation map Siirt
imageNumber = 5;

observation = aug_test.readByIndex(imgIdx(imageNumber));
img = observation.input{1};

label = pred_test_labels(imgIdx(imageNumber));
score = sortedScores(imageNumber);

gradcamMap = gradCAM(trainedNetwork_1,img,label);

figure
alpha = 0.5;
plotGradCAM(img,gradcamMap,alpha);
sgtitle(string(label)+" (score: "+ max(score)+")")



%% Function used to search Images with max prediction score
function [sortedScores,imgIdx] = findMaxActivatingImages(imds,className,predictedScores,numImgsToShow)
% Find the predicted scores of the chosen class on all the images of the chosen class
% (e.g. predicted scores for sushi on all the images of sushi)
[scoresForChosenClass,imgsOfClassIdxs] = findScoresForChosenClass(imds,className,predictedScores);

% Sort the scores in descending order
[sortedScores,idx] = sort(scoresForChosenClass,'descend');

% Return the indices of only the first few
imgIdx = imgsOfClassIdxs(idx(1:numImgsToShow));

end
%% Function used to search Images with min prediction score

function [sortedScores,imgIdx] = findMinActivatingImages(imds,className,predictedScores,numImgsToShow)
% Find the predicted scores of the chosen class on all the images of the chosen class
% (e.g. predicted scores for sushi on all the images of sushi)
[scoresForChosenClass,imgsOfClassIdxs] = findScoresForChosenClass(imds,className,predictedScores);

% Sort the scores in ascending order
[sortedScores,idx] = sort(scoresForChosenClass,'ascend');

% Return the indices of only the first few
imgIdx = imgsOfClassIdxs(idx(1:numImgsToShow));

end

%% Other functions
function [scoresForChosenClass,imgsOfClassIdxs] = findScoresForChosenClass(imds,className,predictedScores)
% Find the index of className (e.g. "sushi" is the 9th class)
uniqueClasses = unique(imds.Labels);
chosenClassIdx = string(uniqueClasses) == className;

% Find the indices in imageDatastore that are images of label "className"
% (e.g. find all images of class sushi)
imgsOfClassIdxs = find(imds.Labels == className);

% Find the predicted scores of the chosen class on all the images of the
% chosen class
% (e.g. predicted scores for sushi on all the images of sushi)
scoresForChosenClass = predictedScores(imgsOfClassIdxs,chosenClassIdx);
end

function plotImages(imds,imgIdx,sortedScores,predictedClasses,numImgsToShow)

for i=1:numImgsToShow
    score = sortedScores(i);
    sortedImgIdx = imgIdx(i);
    predClass = predictedClasses(sortedImgIdx); 
    correctClass = imds.Labels(sortedImgIdx);
        
    imgPath = imds.Files{sortedImgIdx};
    
    if predClass == correctClass
        color = "\color{green}";
    else
        color = "\color{red}";
    end
    
    predClassTitle = strrep(string(predClass),'_',' ');
    correctClassTitle = strrep(string(correctClass),'_',' ');
    
    subplot(3,ceil(numImgsToShow./3),i)
    imshow(imread(imgPath));
    title("Predicted: " + color + predClassTitle + "\newline\color{black}Score: " + num2str(score) + "\newlineGround truth: " + correctClassTitle);
end

end

%% GradCam
function plotGradCAM(img,gradcamMap,alpha)

subplot(1,2,1)
imshow(img);

h = subplot(1,2,2);
imshow(img)
hold on;
imagesc(gradcamMap,'AlphaData',alpha);

originalSize2 = get(h,'Position');

colormap jet
colorbar

set(h,'Position',originalSize2);
hold off;
end