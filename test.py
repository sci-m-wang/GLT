import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
import random
import matplotlib.pyplot as plt
from DataSets.DataLoader import MnistData

mnist = MnistData()
X = mnist.data
y = mnist.target

CFs = pd.read_csv("/data/wm/GLT/MnistCounterfactuals.csv")
CFs = np.array(CFs)
X_cf = CFs[:,1:-1]
y_cf = CFs[:,-1]

clf = MLPClassifier((100,10),max_iter=1000)
clf.fit(X,y)

true_CFs = []
true_CF_labels = []
for index in range(len(X_cf)):
    if clf.predict(X_cf[index].reshape(1,-1)) == y_cf[index]:
        true_CFs.append(X_cf[index])
        true_CF_labels.append(y_cf[index])
        pass
    pass

for i in range(10):
    index = random.randint(0,1000)
    print(true_CF_labels[index])
    image = true_CFs[index].reshape((28,28))
    plt.imshow(image,cmap="Greys",interpolation="None")
    plt.show()
    pass
