# Introduction

In this problem we are tasked with recognizing spoken digits (0-9) in English. To solve this problem, we turn to Deep Learning and Mel-Frequency Cepstral Coefficients extraction. In this project, we use `Python`, with `Keras` for the model building, `Scipy` for audio file processing and `Pydub` for cutting and joining of audio files. `Numpy` is also used for some matrix operations.

# Before You Start

Before you begin, you need to unzip `spoken_numbers.7z`. It contains all the data we will use, which comes from [Pannous.net](pannous.net/spoken_numbers.zip). Also, if you don't want to pre-process the data yourself, you must unzip `X_train.7z` from the `np_mfcc` folder. It contains the data in `numpy` arrays, needed for the training.

# Overview

The steps for this task of Automatic Speech Recognition are the following:

1. Reading Dataset
2. Data Processing (Padding and Normalization)
3. Feature Extraction (MFCC)
4. Split of Dataset into Training/Validation
5. Training of a Convolutional Neural Network
6. Validation Score analysis
7. Automatic Speech Recognition on New Samples

# Dataset

The dataset contains recordings from 15 speakers (note that one user speaks German, so we removed her), sampled at a rate of 22050 and with one channel (mono). The recordings are available in 19 different speeds. In total, there are about 2500 recordings, from both genders. Because, though, some recordings are too fast/slow, we remove them to have a smoother learning process. Finally, we hold one speaker for validation (Bruce).

Firstly, we read our data using `Scipy` and we filter them using a Butterworth filter (bidirectional). The bidirectional filtering results in no phase created in the signals.

Afterwards, we want our signals to be of equal length and for this reason we use padding for shorter signals, while we cut longer ones to the desired length. We will also normalize the signals, so that they lie in the `[0, 1]` range. Finally, we center the signals by using padding on the left and right of each signal.

Because we do not have much data, we are using three different filters one each signal, thus creating three new signals, tripling the data at hand.

After this pre-processing step, we are going to extract the MFCC features of each signal. These features can be represented as images and therefore our CNN will be able to learn differences between the different digits (in MFCC image form).

The targets of each datum is converted to one-hot encoding.

# Neural Network

The neural network we built uses a series of Convolutional Layers before a fully connected Dense layer which produces an output of 10 values (one for each digit). The activation function is a ReLU for all layers except the last one, which uses the categorical cross-entropy loss function (since the problem at hand is multi-class classification). The optimizer of the network is ADAM.

The basic idea behind this architecture is that the Convolutional Layers will extract features from the imput images and then the Dense layers will learn to classify items based on these features.

After the Convolutional Layers, we have two more layers:

a) Max Pooling: Passes over "windows" of the previous Convolutional Layer's output and replaces the values with their maximum, passing this adjusted matrix to the next layer.

b) Spatial Dropout: In 2D (in this cases images), this layer "kills" (sets to 0) part of the input randomly, to avoid overfitting. The difference between Spatial and the simple Dropout is that in images a Dropout layer kills individual pixels, while Spatial Dropout kills neighbourhoods of pixels. Since in images neighbourhoods of pixels hold more information together than random pixels, Spatial Dropout works best in this case.

# Training/Validation Sets:

For validation, we used the recordings of a single speaker, Bruce. His recordings were removed from the training set, so that the validation set would be separate from the training one, and therefore representative of outside/new data.

# Digit Recognition

For the recognition of digits in new recordings, the following process is followed:

1. Resample Audio File
2. Split into Words
3. Pre-process data in the same way training data was pre-processed
4. Use the model for recognition of these individual digits
5. Concatenate results together and print

# Results












*NOTE: This project is for an undergraduate assignment I completed.*
