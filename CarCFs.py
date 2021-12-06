import numpy as np
import pandas as pd
from DataSets.DataLoader import CarData
from extract_prototypes import ProtoTypes
from link_tree import LinkTree
from sklearn.neural_network import MLPClassifier
from generate_tree import Node
from sklearn.model_selection import train_test_split

car = CarData()
car.load_data()

X = car.data
y = car.target

## 划分数据集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

## 分类模型质量测试
# clf1 = MLPClassifier((240,2),max_iter=1000)
#
# clf1.fit(X_train,y_train)

## 选择需要转换类型的数据
acc_samples = []
unacc_samples = []
# 根据分类选择数据，注意接受类型分类为1，不接受为0
for index in range(len(X)):
    if y[index] == 1:
        acc_samples.append(X[index])
        pass
    else:
        unacc_samples.append(X[index])
        pass
    pass

## 构建分类器，用于判断生成的反事实是否符合要求
clf2 = MLPClassifier((100,2),max_iter=1000)
clf2.fit(X,y)

## 通过聚类中心点获取原型
AccProto = ProtoTypes(acc_samples,1)
acc_proto,acc_indecies = AccProto.get_prototypes()

UnaccProto = ProtoTypes(unacc_samples,1)
unacc_proto,unacc_indecies = AccProto.get_prototypes()

print(acc_proto,unacc_proto)

## 生成需要的类型
ACCNodes = []         # 样本节点序列
for each in acc_samples:
    sample_node = []
    for value in each:
        sample_node.append(Node(value, "sample"))
        pass
    ACCNodes.append(sample_node)
    pass
UNACCNodes = []
for each in unacc_samples:
    sample_node = []
    for value in each:
        sample_node.append(Node(value, "sample"))
        pass
    UNACCNodes.append(sample_node)
    pass

unacc_proto_nodes = []
unacc_proto_node = []         # 使用不接受原型unacc_proto
for item in unacc_proto:
    for e in item:
        unacc_proto_node.append(Node(e,"proto"))
        pass
    unacc_proto_nodes.append(unacc_proto_node)
    pass
acc_proto_nodes = []
acc_proto_node = []
for item in acc_proto:
    for e in item:
        acc_proto_node.append(Node(e,"proto"))
        pass
    acc_proto_nodes.append(acc_proto_node)
    pass

## 生成反事实
unacc_CFs = []
for each in ACCNodes:
    for unacc_proto_node in unacc_proto_nodes:
        tree = LinkTree(6,2,unacc_proto_node,each)
        tree.gen_link()
        counterfactual = tree.create_CF()
        if clf2.predict(np.array(counterfactual).reshape(1,-1)) == 0:
            unacc_CFs.append(counterfactual)
            pass
        pass
    pass
acc_CFs = []
for each in UNACCNodes:
    for acc_proto_node in acc_proto_nodes:
        tree = LinkTree(6,2,acc_proto_node,each)
        tree.gen_link()
        counterfactual = tree.create_CF()
        if clf2.predict(np.array(counterfactual).reshape(1,-1)) == 1:
            acc_CFs.append(counterfactual)
            pass
        pass
    pass

## 保存反事实库
CFs = acc_CFs.append(unacc_CFs)
CarCounterfactuals = pd.DataFrame(CFs,columns=car.columns)
CarCounterfactuals.to_csv("CarCounterfactuals.csv")

