"""
Builds, trains and evaluates a model for ASR.
The epochs are given by the user, defaulting at 10.
"""

from utils import BuildModel, EvaluatePredictions
import numpy as np
import sys

if len(sys.argv) > 1:
    epochs = int(sys.argv[1])
else:
    epochs = 10

if __name__ == "__main__":
    X_train = np.load('np_mfcc/X_train.npy')
    Y_train = np.load('np_mfcc/Y_train.npy')
    X_test = np.load('np_mfcc/X_test.npy')
    Y_test = np.load('np_mfcc/Y_test.npy')
    inp_length = np.load('np_mfcc/inp_length.npy')
    inp_width = np.load('np_mfcc/inp_width.npy')

    model = BuildModel(X_train, Y_train, X_test, Y_test, inp_length, inp_width, epochs=epochs, load=False)

    EvaluatePredictions(model, X_test, Y_test)
