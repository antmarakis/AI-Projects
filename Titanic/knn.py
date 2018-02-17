import numpy as np
import pandas as pd
import utils
from sklearn.neighbors import KNeighborsClassifier



train_x = pd.read_csv("train.csv", sep=',')
test_x = pd.read_csv("test.csv", sep=',')

# Modify train
train_x['Title'] = utils.create_title(train_x)
train_x, train_y = utils.create_train(train_x)


# Modify test
to_test = test_x.copy(True)

to_test['Title'] = utils.create_title(to_test)

to_test = to_test.drop(['SibSp', 'Parch', 'Ticket', 'Embarked', 'Name', 'Cabin', 'PassengerId', 'Fare', 'Age'], axis=1)
_, to_test['Sex'] = np.unique(test_x['Sex'], return_inverse=True)


# Make predictions
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_x, train_y)
predictions = knn.predict(to_test)
utils.write_to_csv(test_x, predictions)