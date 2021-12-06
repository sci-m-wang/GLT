import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from DataSets.DataLoader import Climate
from joblib import load

climate = Climate()
PATH = os.path.join("/data/wm/GLT/Climate/")
CFs = pd.read_csv(PATH+"ClimateCFs-GLT.csv",header=0)
CFs = pd.DataFrame(CFs)
climate_CFs = np.array(CFs)

X = climate.data
y = climate.target
X_CFs = climate_CFs[:,:-1]
y_CFs = climate_CFs[:,-1]

clf = load("clf.joblib")


print(clf.score(X,y))

print(clf.score(X_CFs,y_CFs))