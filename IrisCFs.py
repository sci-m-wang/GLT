import numpy as np
import pandas as pd
from DataSets.DataLoader import IrisData
from extract_prototypes import ProtoTypes
from link_tree import LinkTree
from sklearn.neighbors import KNeighborsClassifier
from generate_tree import Node
from time import time

n_neighbors = 14
## 导入数据
iris = IrisData()
X = iris.data
y = iris.target

## 训练区分模型
clf = KNeighborsClassifier(n_neighbors,weights="uniform")
clf.fit(X,y)

## 划分不同类别样本
setosa_samples,versicolor_samples,virginica_samples = [],[],[]
non_setosa_samples,non_versicolor_samples,non_virginica_samples = [],[],[]
for index in range(len(X)):
    if y[index] == 1:
        setosa_samples.append(X[index])
        non_versicolor_samples.append(X[index])
        non_virginica_samples.append(X[index])
        pass
    elif y[index] == 2:
        non_setosa_samples.append(X[index])
        versicolor_samples.append(X[index])
        non_virginica_samples.append(X[index])
        pass
    else:
        non_setosa_samples.append(X[index])
        non_versicolor_samples.append(X[index])
        virginica_samples.append(X[index])
        pass
    pass

## 获取原型，这里数据样例较少，省略此步，直接使用对应的样本集作为原型
SetosaProto = setosa_samples.copy()
VersicolorProto = versicolor_samples.copy()
VirginicaProto = virginica_samples.copy()

## 转化为生成链接树需要的节点类型
# 样本节点序列
NonSetosaNodes,NonVersicolorNodes,NonVirginicaNodes = [],[],[]
for each in non_setosa_samples:
    sample_node = []
    for value in each:
        sample_node.append(Node(value,"sample"))
        pass
    NonSetosaNodes.append(sample_node)
    pass
for each in non_versicolor_samples:
    sample_node = []
    for value in each:
        sample_node.append(Node(value,"sample"))
        pass
    NonVersicolorNodes.append(sample_node)
    pass
for each in non_virginica_samples:
    sample_node = []
    for value in each:
        sample_node.append(Node(value,"sample"))
        pass
    NonVirginicaNodes.append(sample_node)
    pass
# 原型节点生成
SetosaProtoNodes,VersicolorProtoNodes,VirginicaProtoNodes = [],[],[]
for item in SetosaProto:
    proto_node = []
    for e in item:
        proto_node.append(Node(e,"proto"))
        pass
    SetosaProtoNodes.append(proto_node)
    pass
for item in VersicolorProto:
    proto_node = []
    for e in item:
        proto_node.append(Node(e,"proto"))
        pass
    VersicolorProtoNodes.append(proto_node)
    pass
for item in VirginicaProto:
    proto_node = []
    for e in item:
        proto_node.append(Node(e,"proto"))
        pass
    VirginicaProtoNodes.append(proto_node)
    pass

## 生成反事实
start = time()
setosaCFs,versicolorCFs,virginicaCFs = [],[],[]
# 对非setosa类别的数据生成setosa类别的数据
for each in NonSetosaNodes:
    for proto_node in SetosaProtoNodes:
        tree = LinkTree(4,1,proto_node,each)
        tree.gen_link()
        counterfactual = tree.create_CF()
        if clf.predict(np.array(counterfactual).reshape(1,-1)) == 1:
            setosaCFs.append(tuple(counterfactual+[1]))
            pass
        pass
    pass
# 与setosa类同理
for each in NonVersicolorNodes:
    for proto_node in VersicolorProtoNodes:
        tree = LinkTree(4,1,proto_node,each)
        tree.gen_link()
        counterfactual = tree.create_CF()
        if clf.predict(np.array(counterfactual).reshape(1,-1)) == 2:
            versicolorCFs.append(tuple(counterfactual+[2]))
            pass
        pass
    pass
for each in NonVirginicaNodes:
    for proto_node in VirginicaProtoNodes:
        tree = LinkTree(4,1,proto_node,each)
        tree.gen_link()
        counterfactual = tree.create_CF()
        if clf.predict(np.array(counterfactual).reshape(1,-1)) == 3:
            virginicaCFs.append(tuple(counterfactual+[3]))
            pass
        pass
    pass
end = time()
print(end-start)

## 保存反事实
CFs = list(set(setosaCFs+versicolorCFs+virginicaCFs))
IrisCounterfactuals = pd.DataFrame(CFs,columns=iris.iris_data.columns.values)
IrisCounterfactuals.to_csv("IrisCounterfactuals.csv",index=False)