from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
import numpy as np
import pandas as pd
from utils import *

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 



train_x = pd.read_csv("train.csv", sep=',')
test_x = pd.read_csv("test.csv", sep=',')

train_y = train_x['SalePrice']
stored_id = test_x.copy(True)

train_x = engineer(train_x)
test_x = engineer(test_x)


# Create model
model = Sequential()

model.add(Dense(25, activation='relu', input_shape=(10,)))
model.add(Dense(15, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

model.compile(optimizer = 'rmsprop',
              loss = root_mean_squared_error,
              metrics = ['mse'])

model.fit(train_x, train_y, epochs=100, batch_size=1, verbose=1)
model.save_weights('Weights/w100.h5')

# Make predictions
predictions = model.predict(test_x).flatten()
write_to_csv(stored_id, predictions)