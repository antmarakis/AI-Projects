"""
Takes as input model weights and runs the Bruce validation set
on the corresponding model. Call with:

python evaluate.py model_weights.h5
"""

from utils import EvaluatePredictions, BuildModel
import numpy as np
import sys

if len(sys.argv) > 1:
    weights = sys.argv[1]
else:
    weights = 'model_weights/model_92.h5'

if __name__ == "__main__":
    X_test = np.load('np_mfcc/X_test.npy')
    Y_test = np.load('np_mfcc/Y_test.npy')
    inp_length = np.load('np_mfcc/inp_length.npy')
    inp_width = np.load('np_mfcc/inp_width.npy')
    
    model = BuildModel(None, None, None, None, inp_length, inp_width, weights=weights)
    EvaluatePredictions(model, X_test, Y_test)
    