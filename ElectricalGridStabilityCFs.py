import numpy as np
import pandas as pd
from DataSets.DataLoader import ElectricalGridStabilityData
from link_tree import LinkTree
from sklearn.neural_network import MLPClassifier
from generate_tree import Node

elec = ElectricalGridStabilityData()

X = elec.data
y = elec.target

stable_samples = []
unstable_samples = []

for index in range(len(X)):
    if y[index] == 1:
        stable_samples.append(X[index])
        pass
    else:
        unstable_samples.append(X[index])
        pass
    pass

clf = MLPClassifier((100,2),max_iter=1000)
clf.fit(X,y)

StableNodes = []
UnStableNodes = []
StableProtoNodes = []
UnStableProtoNodes = []

for each in stable_samples:
    sample = []
    proto = []
    for value in each:
        sample.append(Node(value,"sample"))
        proto.append(Node(value,"proto"))
        pass
    StableNodes.append(sample)
    StableProtoNodes.append(proto)
    pass
for each in unstable_samples:
    sample = []
    proto = []
    for value in each:
        sample.append(Node(value,"sample"))
        proto.append(Node(value,"proto"))
        pass
    UnStableNodes.append(sample)
    UnStableProtoNodes.append(proto)
    pass

unstable_CFs = []
for each in StableNodes:
    for unstable_proto in UnStableProtoNodes:
        tree = LinkTree(12,6,unstable_proto,each)
        tree.gen_link()
        counterfactual = tree.create_CF()
        if clf.predict(np.array(counterfactual).reshape(1,-1)) == 0:
            unstable_CFs.append(counterfactual)
            pass
        pass
    pass
stable_CFs = []
for each in UnStableNodes:
    for stable_proto in StableProtoNodes:
        tree = LinkTree(12,6,stable_proto,each)
        tree.gen_link()
        counterfactual = tree.create_CF()
        if clf.predict(np.array(counterfactual).reshape(1,-1)) == 1:
            stable_CFs.append(counterfactual)
            pass
        pass
    pass

CFs = stable_CFs+unstable_CFs
ElectricalGridStabilityCounterfactuals = pd.DataFrame(CFs,columns=elec.elec_data.columns.values)
ElectricalGridStabilityCounterfactuals.to_csv("ElectricalGridStabilityCounterfactuals.csv")