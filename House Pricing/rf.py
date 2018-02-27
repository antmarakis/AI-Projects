import numpy as np
import pandas as pd
from utils import *
from sklearn.ensemble import RandomForestRegressor



train_x = pd.read_csv("train.csv", sep=',')
test_x = pd.read_csv("test.csv", sep=',')

train_y = train_x['SalePrice']
stored_id = test_x.copy(True)

train_x = engineer(train_x)
test_x = engineer(test_x)

rf = RandomForestRegressor()
rf.fit(train_x, train_y)

predictions = rf.predict(test_x)
write_to_csv(stored_id, predictions)