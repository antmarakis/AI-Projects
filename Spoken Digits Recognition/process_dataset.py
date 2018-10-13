"""
Processes the Spoken Number dataset and stores
the training and validation sets, along with other
necessary parameters.

The features in the sets are the MFCC of each signal.
"""

from utils import *
import numpy as np

if __name__ == "__main__":
    print("\nReading...\n")
    df = ReadFiles()

    print("\nPadding...\n")
    Padding(df)
    print("\nNormalizing...\n")
    Normalize(df)
    print("\nExtracting features...\n")
    Features(df)

    print("\nPrepare dataset...\n")
    # Remove Bruce from df and add him to validation set
    df_val = df[df['speaker'] == 'Bruce'].reset_index(drop=True)
    df = df[df['speaker'] != 'Bruce'].reset_index(drop=True)

    X_train, Y_train, inp_length, inp_width = PrepareDataset(df)
    X_test, Y_test, inp_length, inp_width = PrepareDataset(df_val)

    print("\nSaving...")
    np.save('np_mfcc/X_train', X_train)
    np.save('np_mfcc/Y_train', Y_train)
    np.save('np_mfcc/X_test', X_test)
    np.save('np_mfcc/Y_test', Y_test)
    np.save('np_mfcc/inp_length', inp_length)
    np.save('np_mfcc/inp_width', inp_width)
    np.save('np_mfcc/padding_length', len(df.iloc[0]['data']))
