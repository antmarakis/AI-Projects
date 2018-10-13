"""
Creates a sample audio file from
the Spoken Numbers dataset.
"""

from utils import CreateAudio
import numpy as np
import sys

if len(sys.argv) > 2:
    n = int(sys.argv[1])
    use_training_set = sys.argv[2] in ['True', '1']
elif len(sys.argv) > 1:
    n = int(sys.argv[1])
    use_training_set = True
else:
    n = 10
    use_training_set = True

if __name__ == "__main__":
    CreateAudio(n, use_training_set)
    