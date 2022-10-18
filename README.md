# Skin Cancer Classification Using Convolutional Neural Networks

## Table of Contents
* [Problem Statement](#problem-statement)
* [Step 1: Importing, and Understanding the Data](#step-1-importing-and-understanding-the-data)
* [Step 2: Data Preparation](#step-2-data-preparation)
* [Step 3: Model Building](#step-3-model-building)
* [Models Summary](#models-summary)
* [Next Steps](#next-steps)
* [Acknowledgements](#acknowledgements)

## Problem Statement
In this case study, we have been given a set of images for 9 different skin cancer types.

The objective is to develop a Convolutional Neural Network (CNN) to accurately predict the class corresponding to each image.

## Step 1: Importing, and Understanding the Data
The dataset in each Train and Test directory, contains images in 9 folders corresponding to different cancer types. The images in each folder are of the type with the name of the folder.

Once the images were imported and resized to 180 x 180, a random sample image from each cancer type is shown below.

<img src="/images/cancers_images.png" width = 1200>

The number of images from each cancer type is plotted below. As can be seen:
- the 'pigmented benign keratosis' and 'melanoma' are the prevalent cancer types, which contain 462 and 438 images, respectively.
- On the other hand, 'seborrheic keratosis' cancer type has the least number of images, i.e. 77.

<img src="/images/freq1.png" width = 500>

This means that the dataset is imbalanced, which will be dealt with, using 2 different types of data augmentation techniques below, later:
- Keras Layers
- Augmentor library

## Step 2: Data Preparation
The function train_val_data_prep is used to:
- divide the values of the array of x_train by 255 to have values within (0, 1).
- the labels are first label encoded.
- then train_test_split is used to get the train and validation sets for CNN.
- then the train and validation label encoded y's are one hot encoded to be considered as the output of the CNN.

x_test and y_test are also processed in a similar manner to abovementioned.

## Step 3: Model Building
Three models were developed:
- **Base model**: which uses the original train images for training the CNN.
- **Keras augmented model**: which uses keras.layers functions for data augmentation for training the CNN.
- **Augmented model**: which uses the newly created images with the Augmentor library, 'along with' the original train images for training the CNN.

### Base model
The structure of the base model is shown in the figure below. ALL the models were compiled using the 'categorical_crossentropy' loss function, 'adam' optimiser, and 'accuracy' as the metric, and were trained with batch size of 32 through 30 epochs.

<img src="/images/cnn-structure.png" width = 400>

The plots below show the performance of the Base model in terms of comparison of the train and validation accuracy and losses during the epochs.

<img src="/images/model1-plots.png" width = 600>

A sample Base model after 30 epochs has shown the metrics below, including test set accuracy:

<img src="/images/model1-summary.png" width = 500>

As can be seen, there is a huge gap between training accuracy and validation accuracy (printed above). This means the model is overfitted, which seems to be due to class imbalance and structure of the CNN. The class imbalance is further dealt with in 2 ways below in the 2 later models developed:
- the functions within keras.layers are utilised in order to perform data augmentation.
- the Augmentor library is used in order to create new set of images to be added to the original training images.

### Keras augmented model
This model is very similar to the Base model, exceppt that it utilises RandomFlip, RandomRotation, and RandomZoom from keras.layers.

The plots below show the performance of the Keras augmented model in terms of comparison of the train and validation accuracy and losses during the epochs.

<img src="/images/model2-plots.png" width = 600>

A sample Keras augmented model after 30 epochs has shown the metrics below, including test set accuracy:

<img src="/images/model2-summary.png" width = 500>

As can be seen, the difference between training accuracy and validation accuracy is still high (as printed above). This means the model is still overfit due to this gap and the structure of the CNN.

### Augmented model
The Augmentor library were used to create 500 additional new images from each class to be added to the original train set. This new set of training images (with the number of images in each class shown below) were fed to the CNN with the same structure as the Base model.

<img src="/images/freq2.png" width = 600>

The plots below show the performance of the Augmented model in terms of comparison of the train and validation accuracy and losses during the epochs.

<img src="/images/model3-plots.png" width = 600>

A sample Augmented model after 30 epochs has shown the metrics below, including test set accuracy:

<img src="/images/model3-summary.png" width = 500>

As can be seen, the difference between training accuracy and validation accuracy has reduced to the value printed above. So this implies that the extent of overfit of the model is less than the previous 2 models. This means that using Augmentor to produce new images and adding them to the original train images has improved the performance of the CNN.

## Models Summary
The performance of the developed models are compared and ranked as below.

<img src="/images/models-summary.png" width = 700>

As can be seen, the **Augmented model**, created using the Augmentor library, has:
- the highest train accuracy
- the lowest difference between train and validation accuracy:
- the highest test set accuracy

This is due to the fact that this model has been trained based on more balanced set of data from these 9 classes, and hence has shown a better performance during training, validation, and test. Therefore, this model can be considered for production.

## Next Steps
- To further improve the structure of the CNN
- To further increase the number of samples produced by Augmentor

# Acknowledgements
- I would like to acknowledge the feedback, support and dataset provision by [upGrad](https://www.upgrad.com/gb) and The [International Institute of Information Technology (IIIT), Bangalore](https://www.iiitb.ac.in/).  
- Also, I would like to express my gratitude to [Nishan Ali](https://www.linkedin.com/in/nishan-ali-826552166/) for providing clarification and guidance to carry out this project.   
- Furthermore, the valuable feedback from [Dr Tayeb Jamali](https://www.linkedin.com/in/tayeb-jamali-b1a10937/) is highly appreciated.
