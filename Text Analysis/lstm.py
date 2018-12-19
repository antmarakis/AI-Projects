import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

import re
from utils import *


max_features = 2000
validation_size = 1500


gop = pd.read_csv('Data/gop.csv')
data = gop[['text','sentiment']]

# Balance Negative - Positive tweets
data[data['sentiment'] == 'Negative'] = data[data['sentiment'] == 'Negative'][:2236]
data[data['sentiment'] == 'Neutral'] = data[data['sentiment'] == 'Neutral'][:2236]
data = data.dropna()

data['sentiment'].value_counts() #Negative: 8493; Neutral: 3142; Positive: 2236
X, Y = format_data(data, max_features)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)
X_validate, Y_validate = X_test[-validation_size:], Y_test[-validation_size:]
X_test, Y_test = X_test[:-validation_size], Y_test[:-validation_size]

model = Sequential()
model.add(Embedding(max_features, 125, input_length=X.shape[1]))
model.add(Dropout(0.1))
model.add(LSTM(75, dropout_U=0.2, dropout_W=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['accuracy'])

model.load_weights('Weights/base.h5')
# model.fit(X_train, Y_train, epochs=5, batch_size=32, verbose=1)
# model.save_weights('Weights/base.h5')


score,acc = model.evaluate(X_test, Y_test, verbose=1, batch_size=32)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

results = model.predict_classes(X_validate, batch_size=1, verbose=1)
run_test(results, Y_validate)
