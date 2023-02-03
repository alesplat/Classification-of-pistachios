# Classification of pistachio images with MATLAB

The aim of the following project is to realize a deep learning architecture able to classify correctly two different species of pistachios with different characteristics that address different market types. MATLAB is the software used.

## Data exploration

Our dataset includes 2148 images of two different classes, 1232 for *Kirmizi* pistachios and 916 for *Siirt* pistachios. All images are coloured and have dimensions 600x600x3. Below there is a short preview.

**Kirmizi Pistachio**

<img src="file:///C:/Users/xseri/Downloads/Pistachio_Image_Dataset/Pistachio_Image_Dataset/Immagini%20report/Kirmizi_Pistachio.jpg" title="" alt="Kirmizi_Pistachio" width="500">

**Siirt Pistachio**

<img src="file:///C:/Users/xseri/Downloads/Pistachio_Image_Dataset/Pistachio_Image_Dataset/Immagini%20report/Siirt_Pistachio.jpg" title="" alt="Siirt_Pistachio" width="503">

As we can see, the images are very similar and even a non-expert may not be able to distinguish the two classes.

## Problem designing

Our goal is to realize a network that is able to classify correctly two types of pistachio, Kirmizi and Siirt. To deal with this problem, the best approach is to use a CNN (Convolutional Neural Network).

A **Convolutional Neural Network (ConvNet/CNN)** is a Deep Learning algorithm that can take in an input image, assign some weights and biases to various aspects/objects in the image, and be able to differentiate one from the other. Its built-in convolutional layer reduces the high dimensionality of images without losing its information. That is why CNNs are especially suited for this use case.

For our purposes, rather than working on a CNN from scratch it could be better to adopt a pre-trained CNN and apply the so-called *transfer learning*. Transfer learning is an advanced method of machine learning, in which a model pre-developed to perform a specific task is reused as a starting point for the development of a model intended to perform a second, different task. The intuition behind transfer learning, especially in the task of image classification, is that if a model is trained on a sufficiently large and general dataset, it will effectively act as a generic model of the visual world; it will then be possible to exploit the general features learnt without having to train a new neural network model from scratch, wasting resources and time in the process of training the neural network on sufficiently large datasets to return an optimal result.

<img title="" src="https://data-science-blog.com/wp-content/uploads/2022/04/re-use-of-pre-trained-models-in-transfer-machine-learning.png" alt="Re-use of pre-trained machine learning models in transfer learning" width="516">

There exist different pre-trained models that can be suitable for our needs. Each model has its own peculiar characteristics in terms of depth (number of layers), number of parameters, size and input size accepted. Some of them have been realized to compete on the _ImageNet Large Scale Visual Recognition Challenge (ILSVRC)_, an annual competition that took place between 2010 and 2017 in which the aim was to both promote the development of better computer vision techniques and to benchmark the state of the art.

![](C:\Users\xseri\AppData\Roaming\marktext\images\2023-01-19-10-42-18-image.png)

For my task I tried several networks with different hyperparameters configurations. _GoogleNet_, _SqueezeNet_ and _ResNet18_ did not were able to correctly classify the images, giving a value of accuracy of around 60% in all the cases. The best option turned up to be _EfficientNetb0_.

## Architecture settings

*EfficientNetb0* is a baseline network built up by Mingxing Tan and Quoc V. Le. by performing a neural architecture search using the AutoML MNAS framework, which optimizes both accuracy and efficiency (FLOPS). The resulting architecture uses mobile inverted bottleneck convolution (MBConv). 

<img title="" src="https://www.researchgate.net/profile/Tashin-Ahmed/publication/344410350/figure/fig4/AS:1022373302128641@1620764198841/Architecture-of-EfficientNet-B0-with-MBConv-as-Basic-building-blocks.png" alt="Architecture of EfficientNet-B0 with MBConv as Basic building blocks." width="527">

| Stage | Operator                  | Resolution | #Output Feature Maps (=channels) | #Layers |
|:-----:|:-------------------------:|:----------:|:--------------------------------:| ------- |
| 1     | Conv 3 × 3                | 224x224x3  | 32                               | 1       |
| 2     | MBConv1, k3 × 3           | 112x112x32 | 16                               | 1       |
| 3     | MBConv6, k3 × 3           | 112x112x16 | 24                               | 2       |
| 4     | MBConv6, k5 × 5           | 56x56x24   | 40                               | 2       |
| 5     | MBConv6, k3 × 3           | 28x28x40   | 80                               | 3       |
| 6     | MBConv6, k5 × 5           | 14x14x80   | 112                              | 3       |
| 7     | MBConv6, k5 × 5           | 14x14x112  | 192                              | 4       |
| 8     | MBConv6, k3 × 3           | 7x7x192    | 320                              | 1       |
| 9     | Conv 1 × 1 & Pooling & FC | 7x7x320    | 1280                             | 1       |

A brief workflow of the MBConv1 k3x3 and MBConv6 k3x3 is shown below. MBConv1 uses depthwise convolution (a type of convolution where we apply a single convolutional filter for each input channel) which integrates a kernel size of 3 × 3 with the stride size of *s* and the input feature maps of *M*. Batch Normalization, Swish activation function, Squeeze-excitation (a block that improves channel interdependency by letting the network taking into account how each feature map weights), Convolution with kernel 1x1 and again Batch Normalization follows. 

MbConv6 k3x3 shares the same structure but is preceded by a Convolution with kernel 1x1 and return 6 times the number of input feature maps (extension of feature map); then Batch Normalization and Swish activation function follows. MBConv6 k5x5 is the same with kernel size 5x5.

![](file:///C:/Users/xseri/AppData/Roaming/marktext/images/2023-01-20-11-29-56-image.png)

The main idea behind EfficientNet was to create a baseline network that could be efficiently exploited by applying a new concept of scaling dimensions. Typically, model scaling is done to arbitrarily increase the CNN depth (for instance *ResNet* can be scaled up from *ResNet18* to *ResNet200*) or width, or to use larger input image resolution for training and evaluation. While these methods do improve accuracy, they usually lead to suboptimal performance. Deeper ConvNet can capture richer and more complex features but are also more difficult to train due to the vanishing gradient problem. Whereas, wider networks tend to be able to capture more fine-grained features and are easier to train but shallow networks tend to have difficulties in capturing higher level features.

Mingxing Tan and Quoc V. Le proposed a novel model scaling method that uses a compound coefficient to scale up CNNs. Rather than scaling arbitrarily network dimensions (width, depth, resolution), this method uniformly scales each dimension with a fixed set of scaling coefficients.

![](C:\Users\xseri\AppData\Roaming\marktext\images\2023-01-20-10-20-30-image.png) 

The first step in the compound scaling method is to perform a grid search, as to find relationships between different scaling dimensions of the baseline network under a fixed resource constrain. This determines the appropriate scaling coefficient for each of the dimensions mentioned above. We then apply those coefficients to scale up the baseline network to the desired target model size or computational budget. This compound scaling method consistently improves model accuracy and efficiency for scaling up existing models such as *MobileNet* (+1.4% imagenet accuracy), and *ResNet*(+0.7%), compared to conventional scaling methods.

The effectiveness of model scaling also relies heavily on the baseline network and for this reason *EfficentNetb0* was born.

To solve our classification task, I applied *transfer learning*. I replaced the last FC layer with a new one that gives 2 as output (as the number of classes we have). Then, I applied a sigmoid layer and a final classification layer.

![](C:\Users\xseri\AppData\Roaming\marktext\images\2023-01-18-12-37-46-image.png)

## Data pre-processing and hyperparameter optimization

First of all, we want to load our data. To move accordingly, we first define a variable that contain the path to our images.

```matlab
path_images = "C:\Users\xseri\Downloads\Pistachio_Image_Dataset\Pistachio_Image_Dataset\Pistachio_Image_Dataset"
```

Then, through the function *ImageDataStore* we manage the collection of images files. We include the files contained in the subfolders and associate for each image the name of the folders in which it is contained.

```matlab
image_datastore = imageDatastore(path_images, "IncludeSubfolders",true,"LabelSource","foldernames")
```

At this point we can split our dataset in training, validation and test set.

- Training set -> 60% of data

- Validation set -> 20% of data

- Test set -> 20% of data

We are going to do this split in a randomized way, namely shuffling the data we have.

```matlab
[train, validation, test] = splitEachLabel(image_datastore,0.6, 0.20, 0.20, 'randomized')
```

![](C:\Users\xseri\AppData\Roaming\marktext\images\2023-01-10-17-30-37-image.png)

![](C:\Users\xseri\AppData\Roaming\marktext\images\2023-01-10-17-31-17-image.png)

![](C:\Users\xseri\AppData\Roaming\marktext\images\2023-01-10-17-31-39-image.png)

After having split the dataset, some operations of data augmentation have been performed. I worked with the *Deep Network Designer* app. 

Data augmentation is a process through which we can increase the amount of data by generating new data point composed of slightly modified copies of already existing data. Common transformations are flipping, translation, rotation or the addition of some noise. In my case, I decided to apply a random reflection on X axis and a random rotation in the range -30° and 30°.

![](C:\Users\xseri\AppData\Roaming\marktext\images\2023-01-12-19-00-14-image.png)

After these operations, I set the hyperparameters.

<img src="file:///C:/Users/xseri/AppData/Roaming/marktext/images/2023-01-19-19-23-23-image.png" title="" alt="" width="495">

I adopted the *Stochastic Gradient Descent with momentum (sgdm)* to update the gradient descent, with a *learning rate* (a parameter that influences how far the gradient needs to move after each iteration) equals to 0.01.

10 *epochs* refers to how many times the network have to face all the training images. *MiniBatchSize* equals to 64 means that after having considered 64 images the gradient can be updated, so we associate the concept of iteration to each time a minibatch passes through the network. 

Other interesting hyperparameters are *L2Regularization* (equals to 0.0001) that controls the intensity of regularization of the network (to avoid overfitting in case of large weights that can have a higher influence) and Momentum (equals to 0.9), an hyperparameter involved in causing the gradient descent to reach convergence (the minimum point) faster.

## Training phase

After having finished the phase of data pre-processing and hyperparameter optimization, the net has been trained. As shown in the figure below, to perform 10 epoch it lasted almost 2 hours, reaching a sort of equilibrium in terms of accuracy from the third epoch onwards. The validation accuracy obtained was 93.02%, much higher than the results obtained with the other networks citied at the end of the *problem design* paragraph.

![](C:\Users\xseri\AppData\Roaming\marktext\images\2023-01-18-15-27-50-image.png)

## Testing phase and confusion matrix

Then, I tested my network on the test set.

```matlab
aug_test = augmentedImageDatastore([224 224 3], test)
[pred_test_labels, pred_test_scores] = classify(trainedNetwork_1, aug_test)
true_test_labels = test.Labels
accuracy_test = mean(true_test_labels == pred_test_labels)
```

![](C:\Users\xseri\AppData\Roaming\marktext\images\2023-01-19-15-52-12-image.png)

The model performed quite good, obtaining a relevant result in terms of accuracy. 

Below, the confusion matrix is plotted.

```matlab
 plotconfusion(true_test_labels, pred_test_labels)
```

<img title="" src="file:///C:/Users/xseri/AppData/Roaming/marktext/images/2023-01-18-15-36-38-image.png" alt="" width="445">

In this figure, the first two diagonal cells show the number and percentage of correct classifications by the trained network. For instance, 243 Kirmizi pistachios have been correctly classified. This corresponds to the 56.6% of all the images of the test set (429). Similarly, 171 Siirt pistachios have been correctly classified, corresponding the 39.9% of all the images. 2.8% of Siirt pistachios have been classified wrongly as Kirmizi (12 images), while 0.7% of Kirmizi pistachios have been labelled as Siirt (just 3 images).

Out of 255 Kirmizi predictions, 95.3% are correct while 4.7% are wrong. Out of 174 Siirt predictions, 98.3% are correct and 1.7% are wrong. 

Out of 246 Kirmizi images, 98.8% are correctly predicted while 1.2% are wrongly predicted. Out of 183 Siirt images, 93.4% are correctly predicted while 6.6% are wrongly predicted. 

As we can see the model works quite well. It has just some minor problems when dealing with Siirt images that are more frequently mistaken with Kirmizi images.

## Investigate classifications

We want to visualize some images, especially those that most strongly activate the network for each class, giving a look at the patterns that have the strongest impact in the decision process. Having glanced at the dataset we can expect that some differences are not appreciable for our eyes.

**Kirmizi**

```matlab
%% Kirmizi most like Kirmizi
chosenClass = "Kirmizi_Pistachio";
classIdx = find(trainedNetwork_1.Layers(end).Classes == chosenClass);

numImgsToShow = 9;

[sortedScores,imgIdx] = findMaxActivatingImages(test,chosenClass,pred_test_scores,numImgsToShow);

figure
plotImages(test,imgIdx,sortedScores,pred_test_labels,numImgsToShow)
```

<img src="file:///C:/Users/xseri/AppData/Roaming/marktext/images/2023-01-19-16-46-27-image.png" title="" alt="" width="632">

Let's visualize the activation map for one of those images.

<img src="file:///C:/Users/xseri/AppData/Roaming/marktext/images/2023-01-19-16-53-35-image.png" title="" alt="" width="479">

As we can see, there are some patterns in the shell of the pistachio that the algorithm finds in common with other images of the same class.  

**Siirt**

```matlab
%% Siirt most like Siirt
chosenClass = "Siirt_Pistachio";
classIdx = find(trainedNetwork_1.Layers(end).Classes == chosenClass);

numImgsToShow = 9;

[sortedScores,imgIdx] = findMaxActivatingImages(test,chosenClass,pred_test_scores,numImgsToShow);

figure
plotImages(test,imgIdx,sortedScores,pred_test_labels,numImgsToShow)
```

<img title="" src="file:///C:/Users/xseri/AppData/Roaming/marktext/images/2023-01-19-16-54-51-image.png" alt="" width="611">

We can notice that even though we are looking at the images that have most strongly activated the network for the Siirt class, there is an image that has been misclassified, meaning that a score even higher has been assigned to the predicted class (despite the score for Siirt it's still very relevant being 0.99994).

Let's visualize the activation map of an image correctly classified and the activation map of the image indicated above as misclassified.

<img title="" src="file:///C:/Users/xseri/AppData/Roaming/marktext/images/2023-01-19-17-00-23-image.png" alt="" width="447">

<img title="" src="file:///C:/Users/xseri/AppData/Roaming/marktext/images/2023-01-19-18-09-30-image.png" alt="" width="448">

As stated at the beginning of the paragraph, it's not always feasible for us to understand clearly what the network sees. In this case it's very difficult to state why the algorithm misclassified the pistachio according to our eyes. We can just affirm that the network finds some elements in the upper part of the shell that resemble those seen in Kirmizi and makes this mistake. 

## Comments on results

The network applied to our dataset showed relevant performance in terms of accuracy.   *EfficientNetb0* was the only tested architecture that was able to predict with low misclassification errors our classes, even though the low number of parameters adopted (the general structure has almost 5.2 mln vs 7 mln of *GoogleNet* and 11.7 of *ResNet18*, just SqueezeNet had fewer parameters, 1.24). 

Below are plotted the misclassified images of Kirmizi pistachios...

<img src="file:///C:/Users/xseri/AppData/Roaming/marktext/images/2023-01-19-19-05-44-image.png" title="" alt="" width="612">

... and the misclassified images of Siirt pistachios.

![](C:\Users\xseri\AppData\Roaming\marktext\images\2023-01-19-18-51-14-image.png)

They are difficult to distinguish, aren't they?

## Bonus: Cross-validation approach

A second way to realize a network that could be able to solve our task is through cross-validation. Generally speaking, with cross-validation we can train the model on all the data we have and in different conditions so to face problems of bias or overfitting (as can happen when working on static data such as a pre-defined training set), making the network more trustworthy. I applied cross-validation by splitting the data in k-fold (with k=5) so to train and test it with different subsets. 

I splitted the data in training and test (I'll leabe the test set as a final set in which I test the results extracted by CV).

```matlab
[traincv, testcv] = splitEachLabel(image_datastore,0.7, 0.30, 'randomized'); 
```

Then, i partitioned my training set in 5 folds and iteratively I trained and tested 5 times the network to see the results obtained in terms of accuracy score. 

```matlab
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
```

I performed the same operations of data augmentation that have been performed previously, namely I let the images randomly reflect on the X axis and made them randomly rotate between -30° and 30°. After having performed the training phase, I showed the accuracies obtained by the 5 iterations and calculated the average.

![](C:\Users\xseri\AppData\Roaming\marktext\images\2023-02-01-18-13-59-image.png)

```matlab
avg = sum(accuracies)/length(accuracies);
```

![](C:\Users\xseri\AppData\Roaming\marktext\images\2023-02-01-18-17-55-image.png)

As we can see, the results obtained were very relevant. On average I obtained an accuracy of 0.95, leading me to think that  the network behaves well. To test the network with the test set, I chose as final network the one that had the higher accuracy score.

```matlab
[max_acc,indx_max] = max(accuracies);
cv_net = nets(indx_max);
```

![](C:\Users\xseri\AppData\Roaming\marktext\images\2023-02-01-18-15-30-image.png)

Then, I tested it with the test set.

```matlab
true_test_labels = testcv.Labels;
pred_test_labels = classify(cv_net,augmented_test);
accuracy_test_cv = mean(true_test_labels == pred_test_labels);
```

<img title="" src="file:///C:/Users/xseri/AppData/Roaming/marktext/images/2023-02-01-18-26-24-image.png" alt="" width="234">

As we can see, the results obtained confirm that the network is very solid and does not have problems of overfitting. We can conclude that the model is able to generalize in a good way, being able to distinguish the two classes of pistachios with consistency and reliability. 
