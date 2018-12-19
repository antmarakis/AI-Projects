import numpy as np
import pandas as pd
from keras.layers import Input, Dense, Bidirectional, GRU, Embedding, Dropout, Reshape, Conv2D, Flatten
from keras.layers import Concatenate, SpatialDropout1D, GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPool2D
from keras.models import Model
from sklearn.model_selection import train_test_split
from utils import *


maxlen = 150
max_features = 2500
emb_size = 150
num_filters = 16
filter_sizes = [1, 2, 3, 5]


gop = pd.read_csv('Data/gop.csv')
data = gop[['text','sentiment']]

# Balance Negative - Positive tweets
data[data['sentiment'] == 'Negative'] = data[data['sentiment'] == 'Negative'][:2236]
data[data['sentiment'] == 'Neutral'] = data[data['sentiment'] == 'Neutral'][:2236]
data = data.dropna()

data['sentiment'].value_counts() #Negative: 8493; Neutral: 3142; Positive: 2236
X, Y = format_data(data, max_features, maxlen)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)

# Input shape
inp = Input(shape=(maxlen,))

# Embedding and GRU
x = Embedding(max_features, emb_size)(inp)
x = SpatialDropout1D(0.25)(x)
x = Bidirectional(GRU(75, return_sequences=True))(x)
x = Reshape((maxlen, emb_size, 1))(x)
    
conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], emb_size),
                kernel_initializer='normal', activation='elu')(x)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], emb_size),
                kernel_initializer='normal', activation='elu')(x)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], emb_size),
                kernel_initializer='normal', activation='elu')(x)
conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], emb_size),
                kernel_initializer='normal', activation='elu')(x)
    
maxpool_0 = MaxPool2D(pool_size=(maxlen - filter_sizes[0] + 1, 1))(conv_0)
maxpool_1 = MaxPool2D(pool_size=(maxlen - filter_sizes[1] + 1, 1))(conv_1)
maxpool_2 = MaxPool2D(pool_size=(maxlen - filter_sizes[2] + 1, 1))(conv_2)
maxpool_3 = MaxPool2D(pool_size=(maxlen - filter_sizes[3] + 1, 1))(conv_3)
        
conc = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])   
z = Flatten()(conc)
z = Dropout(0.1)(z)

# Output layer
output = Dense(1, activation='sigmoid')(z)

model = Model(inputs=inp, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# model.load_weights('Weights/gru5_3.h5')
model.fit(X_train, Y_train, epochs=3, batch_size=32, verbose=1)
# model.save_weights('Weights/gru3.h5')

results = model.predict(X_test, batch_size=1, verbose=1)
run_test(results, Y_test)
