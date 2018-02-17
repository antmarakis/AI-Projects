from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import utils



train_x = pd.read_csv("train.csv", sep=',')
test_x = pd.read_csv("test.csv", sep=',')

# Modify train
train_x['Title'] = utils.create_title(train_x)
train_x, train_y = utils.create_train(train_x)

# Create model
model = Sequential()

model.add(Dense(16, activation='relu', input_shape=(3,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# model.fit(train_x, train_y, epochs=50, batch_size=1, verbose=1)
# model.save_weights('Weights/titles.h5')

model.load_weights('Weights/titles.h5')

# Modify test
to_test = test_x.copy(True)

to_test['Title'] = utils.create_title(to_test)

to_test = to_test.drop(['SibSp', 'Parch', 'Ticket', 'Embarked', 'Name', 'Cabin', 'PassengerId', 'Fare', 'Age'], axis=1)
_, to_test['Sex'] = np.unique(test_x['Sex'], return_inverse=True)


# Make predictions
predictions = model.predict_classes(to_test).flatten()
utils.write_to_csv(test_x, predictions)