"""
Contains all the code used in the various scripts of the project.
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict

PATH = 'spoken_numbers'
extreme_speeds = ['40', '60', '80', '320', '340', '360', '380', '400']



def CreateAudio(n=10, use_training_set=False):
    """
    Create random audio file by concatenating n random files from path p.
    By default, speakers from the training set will not be used. To use such
    speakers, set use_training_set to True.
    """
    from pydub import AudioSegment
    from random import shuffle
    
    fnames = os.listdir(PATH)
    files = []
    for f in fnames:
        if not f.endswith('.wav'):
            continue
        
        digit, speaker, speed = f[:-4].split('_')
        
        if speaker == 'Steffi':
            # Steffi speaks German
            continue
        
        if speed in extreme_speeds:
            # Extreme speeds
            continue
        
        # Either use files exclusively of Bruce, or don't include
        # Bruce at all, depending on use_training_set.
        if not use_training_set and speaker != 'Bruce':
            continue
        elif use_training_set and speaker == 'Bruce':
            continue
        
        files.append(f)
    
    shuffle(files)
    files = files[:n]
    
    total = AudioSegment.empty()
    silence = AudioSegment.silent(500)
    
    for f in files:
        seg = AudioSegment.from_wav(PATH + '/' + f)
        total += seg + silence
    
    total.export('audio_sample.wav', format='wav')



def SplitAudio(fname):
    """
    Splits given audio file on silence and exports
    the separated chunks. 
    """
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
    from scipy.io import wavfile
    from scipy import signal
    
    silence = AudioSegment.silent(500)
    audio = AudioSegment.from_wav(fname)
    audio = silence + audio + silence
    chunks = split_on_silence(audio,
                              min_silence_len=175,
                              silence_thresh=-50)

    wavs = defaultdict(list)
    b, a = signal.butter(4, 0.1, analog=False)
    
    for i, c in enumerate(chunks):
        fname = 'Words/word{}.wav'.format(i)
        c.export(fname, format='wav')
        
        rate, data = wavfile.read(fname)
        if len(data.shape) == 2:
            # Two channels were found (stereo), but we need only one
            data = data[:, 0]
        
        data = signal.filtfilt(b, a, data) # Filter signal
        
        wavs['rate'].append(rate)
        wavs['data'].append(data)
    
    return pd.DataFrame(wavs)



def ReadAudioFile(fname, p=PATH):
    """
    Reads audio.
    
    When reading, we will first pad the file with some silence
    and then trim it. That way all the audio files in our system
    will be similarly positioned.
    """
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
    from scipy.io import wavfile
    
    silence = AudioSegment.silent(500)
    audio = AudioSegment.from_wav(p + '/' + fname)
    audio = silence + audio + silence
    chunk = split_on_silence(audio,
                             min_silence_len=175,
                             silence_thresh=-50)
    
    c = chunk[0]
    c.export(fname, format='wav')
    f = ConvertSampleRate(fname)
    rate, data = wavfile.read(f)
    
    ### CLEAN-UP ###
    os.remove(fname)
    if f != fname:
        # A new file was also created. Delete it too.
        os.remove(f)
    
    return rate, data



def ReadFiles(p=PATH):
    """
    Reads files from PATH, filtering the signal and storing
    audio file information on a pandas DataFrame.
    
    We store the audio files of a particular speaker (Bruce)
    for testing purposes.
    """
    from scipy import signal

    fnames = os.listdir(p)
    wavs = defaultdict(list)

    # Create Butterworth Filtering parameters
    b, a = [0]*3, [0]*3
    b[0], a[0] = signal.butter(4, 0.1, analog=False)
    b[1], a[1] = signal.butter(4, 0.7, analog=False)
    b[2], a[2] = signal.butter(8, 0.5, analog=False)

    for f in fnames:
        if not f.endswith('.wav'):
            continue
        
        digit, speaker, speed = f[:-4].split('_')
        
        if speaker == 'Steffi':
            # Steffi speaks German
            continue
        
        if speed in ['40', '60', '80', '320', '340', '360', '380', '400']:
            # Extreme speeds
            continue
        
        # rate, data = wavfile.read(p + '/' + f)
        rate, data = ReadAudioFile(f)
        if len(data.shape) == 2:
            # Two channels were found (stereo), convert it to one (mono)
            # Normally, we should be averaging the two channels, but here
            # we assume the two channels have equal values and we can
            # simply drop one of them. That is to speed up computation.
            data = data[:, 0]
        
        for i in range(len(b)):
            d_filtered = signal.filtfilt(b[i], a[i], data) # Filter signal
        
            wavs['digit'].append(digit)
            wavs['speaker'].append(speaker)
            wavs['speed'].append(speed)
            wavs['rate'].append(rate)
            wavs['data'].append(d_filtered)

    return pd.DataFrame(wavs)



def Padding(df, max_length=45000, hard_set=False):
    """
    Pad signals up to max_length. If a signal is longer, then cut it down to max_length.
    This ensures that all signals are of the same size, which is necessary for the network.
    
    With the argument hard_set we can make sure the padding length will be exactly max_length.
    """
    m = -1
    for _, d in df['data'].iteritems():
        l = len(d)
        if l > m:
            m = l
    
    if m < max_length and not hard_set:
        max_length = m
    
    for i, d in df['data'].iteritems():
        if len(d) <= max_length:
            df['data'][i] = np.pad(d, (0, max_length - len(d)), 'constant', constant_values=(0))
        else:
            df['data'][i] = df['data'][i][:max_length]



def Normalize(df):
    """
    Normalizes signal data, so that all values are in [0, 1].
    """
    for i, d in df['data'].iteritems():
        df['data'][i] = MinMax(d)



def MinMax(to_normalize):
    """
    Given an array to normalize, use the min_max normalization method.
    """
    total_max, total_min = np.max(to_normalize), np.min(to_normalize)
    return (to_normalize - total_min) / (total_max - total_min)



def Features(df):
    """
    Calculate the features of the audio files.
    Namely, the Mel-Frequency Cepstral Coefficients (mfcc). 
    """
    from python_speech_features import mfcc, logfbank
    from scipy.io import wavfile

    d_series, r_series = df['data'], df['rate']
    m_features = []
    for (_, d), (_, r) in zip(d_series.iteritems(), r_series.iteritems()):
        m_features.append(mfcc(d, r, nfft=2048))

    df['mfcc'] = m_features



def PrepareDataset(df):
    """
    Prepare dataset for the Neural Network.
    Specifically, reshape and one-hot encode targets.
    """
    inp_length, inp_width = df['mfcc'][0].shape
    
    dataset = df[['digit', 'mfcc']]
    dataset = dataset.sample(frac=1).reset_index(drop=True) # Shuffle

    size = len(dataset)
    
    # Split dataset to training and validation   
    X_train = dataset['mfcc'].reset_index(drop=True)
    Y_train = dataset['digit'].reset_index(drop=True)

    # Reshape dataset into N x L x W numpy arrays, where:
    # N: Size of set
    # L: Input length
    # W: Input width
    # Each item is basically treated as an image of L x W dimensions
    X_train = np.concatenate(X_train, axis=0)
    X_train = X_train.reshape([size, inp_length, inp_width, 1])

    # One-Hot encode target digits
    from sklearn.preprocessing import OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse=False)
    Y_train = onehot_encoder.fit_transform(Y_train.reshape(-1, 1))
    
    return X_train, Y_train, inp_length, inp_width



def BuildModel(X_train, Y_train, X_val, Y_val, inp_length, inp_width, epochs=5, weights='model_weights/model_92.h5', load=True):
    """
    Builds Convolutional Neural Network model.
    """
    from keras.models import Model, Sequential
    from keras.layers import Dense, Dropout, SpatialDropout2D, Flatten, Conv2D, MaxPooling2D, concatenate

    input_shape = (inp_length, inp_width, 1)

    model = Sequential()
    
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(16, (1, 1), activation='relu'))
    model.add(SpatialDropout2D(0.25))
    
    model.add(Flatten())
    
    model.add(Dense(32, activation='relu'))
    
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    if load:
        model.load_weights(weights)
    else:
        model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=1, verbose=1)
        model.save_weights("model.h5")

    return model



def EvaluatePredictions(model, X_test, Y_test):
    """
    Checks the predictions for a given set of inputs against the
    real targets of the set and returns accuracy in percentile form
    for all the different classes/digits.
    
    Since the output of predict_classes is a single number, we check
    if the corresponding value in the one-hot encoded Y_test is 1 or 0.
    If it is 1, the prediction is correct.
    """
    results = model.predict_classes(X_test, batch_size=1, verbose=1)
    correct = {d: 0 for d in range(10)}
    total_correct = 0
    
    counts = {d: 0 for d in range(10)}
    for t in Y_test:
        counts[list(t).index(1)] += 1

    for i, r in enumerate(results):
        if Y_test[i][r] == 1:
            total_correct += 1
            correct[r] += 1

    for k, v in correct.items():
        print('{} Accuracy:'.format(k), correct[k] / counts[k] * 100, '%')
    
    print('Total Accuracy:', total_correct/len(results) * 100, '%')



def Recognize(fname, weights='model_weights/model_92.h5'):
    """
    Recognize digits from fname, given the model by weights.
    
    The file will first be converted to 22050 sample rates
    and then broken into separate words by SplitAudio,
    before being put through the same process as the training
    and validation sets. Finally, a model will be built using
    the given weights file to predict the classes of the audio.
    """
    f = ConvertSampleRate(fname)
    
    df = SplitAudio(f)
    padding_length = int(np.load('np_mfcc/padding_length.npy'))

    Padding(df, max_length=padding_length, hard_set=True)
    Normalize(df)
    Features(df)

    inp_length, inp_width = df['mfcc'][0].shape
    
    X = df['mfcc']
    X = np.concatenate(X, axis=0)
    X = X.reshape([len(df), inp_length, inp_width, 1])
    
    model = BuildModel(None, None, None, None, inp_length, inp_width, weights=weights)
    predictions = model.predict_classes(X, batch_size=1, verbose=1)
    PrintSentence(predictions)
    
    ### CLEAN-UP ###
    if f != fname:
        os.remove(f)



def PrintSentence(output):
    """
    Take as input the output of the model and print it out
    with words replacing the corresponding digits.
    """
    digits = ['zero', 'one', 'two', 'three', 'four',
              'five', 'six', 'seven', 'eight', 'nine']

    to_print = []
    for o in output:
        to_print.append(digits[o])
    
    print(' '.join(to_print))



def ConvertSampleRate(fname, rate=22050):
    """
    Converts sample rates of audio files to given rate using sox.
    WARNING: If given a value to convert to other than the default 22050,
    issues may arise. This has not been tested properly.
    
    Returns the name of the processed file. If no file was created,
    returns the original argument.
    """
    from scipy.io import wavfile
    r, _ = wavfile.read(fname)
    if r == rate:
        return fname
    
    fout = fname[:-4] + '_temp.wav'
    
    from subprocess import call, check_output
    
    command = 'sox "{0}" -r {1} "{2}"'.format(fname, rate, fout)
    call(command, shell=True)
    
    return fout



def Plots(df, i=0):
    """
    A couple of visualizations for signal data and mfcc features.
    """
    import matplotlib.pyplot as plt
    
    plt.plot(df.iloc[i]['data'], '-', ); plt.show();
    plt.imshow(df.iloc[i]['mfcc'], cmap='hot', interpolation='nearest'); plt.show();
