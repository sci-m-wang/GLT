from joblib import dump
from sklearn.neural_network import MLPClassifier
from DataSets.DataLoader import Climate

climate = Climate()
X = climate.data
y = climate.target

clf = MLPClassifier((100,2),max_iter=1000)

clf.fit(X,y)

dump(clf,"clf.joblib")