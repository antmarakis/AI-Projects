# Introduction

In this problem we are tasked with recognizing spoken digits (0-9) in English. To solve this problem, we turn to Deep Learning and Mel-Frequency Cepstral Coefficients extraction. In this project, we use `Python`, with `Keras` for the model building, `Scipy` for audio file processing and `Pydub` for cutting and joining of audio files. `Numpy` is also used for some matrix operations.

# Before You Start

Before you begin, you need to unzip `spoken_numbers.7z`. It contains all the data we will use, which comes from [Pannous.net](pannous.net/spoken_numbers.zip) and [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset). Also, if you don't want to pre-process the data yourself, you must unzip `X_train.7z` from the `np_mfcc` folder. It contains the data in `numpy` arrays, needed for the training.

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

The dataset contains recordings from speakers sampled at a rate of 22050 and with one channel (mono). The recordings are available in different speeds and include audio files from both genders. From this collection of recordings, we hold one speaker for validation (Bruce).

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

The neural network is trained for 10 epochs, with data passed in one-by-one (batch size is 1). The training accuracy score reaches around 95%, while the validation set usually reaches around 85%.

The model was tested on new data, which can be found in the 'to_recognize' folder. All recordings of Bruce (which is a concatenation of some random single digit recordings of his) can be recognized in great accuracy. For our own recordings, the accuracy is found below for the model weights we have stored in the 'model_weights' folder:

‘recording_9’: 7/9  
‘recording_10a’: 7/10  
‘recording_10b’: 8/10  
‘recording_10c’: 10/10

# Project Structure

Documentation for files and folders follows:

* `process_dataset.py`: Creates the training and validation sets, with the necessary features.

* `train_model.py`: Neural network training.

* `evaluate.py`: Performance analysis for validation set. Shows how well the model performed for each digit.

* `create_audio.py`: Creates recordings from the dataset files, by concatenating random single digit recordings.

* `recognize.py`: Takes as input an audio file and outputs its digits.

* `utils.py`: Contains utility functions used in the rest of the scripts.

* `model_weights`: A folder where model weights will be saved after training.

* `np_mfcc`: `Numpy` arrays are stored here for the processed dataset.

* `spoken_numbers`: Contains the dataset of speakers (not that I recorded myself some digits to add to this set).

* `to_recognize`: A folder containing some sample recordings for recognition.

* `Words`: When we try to recognize digits in a recording, we first split it into separate words. These words are stored in this folder. After the recognition, we do not delete this folder for debugging purposes.

# Instructions

* To check the accuracy of a network on the validation set, run this in the command line:

`python evaluate.py model_weights.h5`

or just

`python evaluate.py`

* To recognize audio files (either your own or those in the 'to_recognize' folder):

`python recognize.py file_name.wav model_weights.h5`

or simply

`python recognize.py file_name.wav`

Note that the files need to be `.wav` files, with a space of about 0.2 seconds between each digit.

* TO create an audio file from the dataset, use:

`python create_audio.py n Boolean`

where `n` is the number of words and `Boolean` a boolean value (as recognized by Python, that is, `True`/`False`, `0`/`1` etc.).

Alternatively

`python create_audio.py`

With `create_audio.py` a new file named `audio_sample.wav` will be created in the file directory.

* To train a model from scratch you can run the scripts `process_dataset.py` and `train_model.py`. The `train_model.py` script can take as input the number of epochs.

```
python process_dataset.py
python train_model.py n
```

where `n` is the number of epochs.

---

NOTE: More information on execution of scripts can be found in the header of each script.

# Pre-requisites

For the development of this project, `Python 3.6.4` was used. To run the whole code, the following libraries are used:

* `Keras`
* `Numpy`
* `Pandas`
* `Scipy`
* [`Pydub`](http://pydub.com/)
* `Matplotlib` (for signal visualization, something which can be skipped)
* [`python_speech_features`](https://python-speech-features.readthedocs.io/en/latest/)

Also, you are going to need [`Sox`](http://sox.sourceforge.net/) for the audio file resampling.

# Final Note

*This project is for an undergraduate assignment I completed. Some of the test recordings came from some friends of mine.*
