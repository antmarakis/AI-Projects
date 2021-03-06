import pandas as pd
import csv
from utils import *


train_x = pd.read_csv("train.csv", sep=',')
test_x = pd.read_csv("test.csv", sep=',')

EAP, HPL, MWS = parse_text(train_x)
c_eap, c_hpl, c_mws = create_dist(EAP), create_dist(HPL), create_dist(MWS)

dist = {'EAP': c_eap, 'HPL': c_hpl, 'MWS': c_mws}
nBS = NaiveBayes(dist)

submission = predictions(test_x, nBS)
submission.to_csv('submission.csv', index=False)
