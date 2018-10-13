"""
Takes as input an audio file name and model weights
and prints the list of recognized digits. Call with:

python recognize.py file_name.wav model_weights.h5
"""

from utils import Recognize
import sys

fname = sys.argv[1]
if len(sys.argv) > 2:
    weights = sys.argv[2]
else:
    weights = 'model_weights/model_92.h5'

if __name__ == "__main__":
    Recognize(fname, weights)
    