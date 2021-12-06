import numpy as np
import pandas as pd

from DataSets.DataLoader import InternetAdData
from extract_prototypes import ProtoTypes
from link_tree import LinkTree
from sklearn.neural_network import MLPClassifier
from generate_tree import Node
from sklearn.model_selection import train_test_split

ad = InternetAdData()

X = ad.data
y = ad.target
## 划分数据集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

## 分类模型质量测试
# clf1 = MLPClassifier((240,2),max_iter=1000)
#
# clf1.fit(X_train,y_train)

## 选择需要转换类型的数据
ad_samples = []
nonad_samples = []
# 根据分类选择数据，注意接受类型分类为1，不接受为0
for index in range(len(X)):
    if y[index] == 1:
        ad_samples.append(X[index])
        pass
    else:
        nonad_samples.append(X[index])
        pass
    pass

## 构建分类器，用于判断生成的反事实是否符合要求
clf2 = MLPClassifier((100,2),max_iter=1000)
clf2.fit(X,y)

## 通过聚类中心点获取原型
AdProto = ProtoTypes(ad_samples,5)
ad_proto,ad_indecies = AdProto.get_prototypes()

NonadProto = ProtoTypes(nonad_samples,5)
nonad_proto,nonad_indecies = NonadProto.get_prototypes()

# print(pd.Series(ad_proto)==pd.Series(nonad_proto))

## 生成需要的类型
AdNodes = []         # 样本节点序列
for each in ad_samples:
    sample_node = []
    for value in each:
        sample_node.append(Node(value, "sample"))
        pass
    AdNodes.append(sample_node)
    pass
NonadNodes = []
for each in nonad_samples:
    sample_node = []
    for value in each:
        sample_node.append(Node(value, "sample"))
        pass
    NonadNodes.append(sample_node)
    pass

nonad_proto_nodes = []
nonad_proto_node = []         # 使用不接受原型unacc_proto
for item in nonad_proto:
    for e in item:
        nonad_proto_node.append(Node(e,"proto"))
        pass
    nonad_proto_nodes.append(nonad_proto_node)
    pass
ad_proto_nodes = []
ad_proto_node = []
for item in ad_proto:
    for e in item:
        ad_proto_node.append(Node(e,"proto"))
        pass
    ad_proto_nodes.append(ad_proto_node)
    pass
## 生成反事实
nonad_CFs = []
for each in AdNodes:
    for nonad_proto_node in nonad_proto_nodes:
        tree = LinkTree(1555,5,nonad_proto_node,each)
        tree.gen_link()
        counterfactual = tree.create_CF()
        if clf2.predict(np.array(counterfactual).reshape(1,-1)) == 0:
            nonad_CFs.append(counterfactual.append(0))
            pass
        pass
    pass

ad_CFs = []
for each in NonadNodes:
    for ad_proto_node in ad_proto_nodes:
        tree = LinkTree(1555,5,ad_proto_node,each)
        tree.gen_link()
        counterfactual = tree.create_CF()
        if clf2.predict(np.array(counterfactual).reshape(1,-1)) == 1:
            ad_CFs.append(counterfactual.append(1))
            pass
        pass
    pass

column = ["A"+str(i) for i in range(1555)]
column.append("if-ad")

counterfactual_array = ad_CFs.append(nonad_CFs)
print(len(counterfactual_array))
AdCounterfactuals = pd.DataFrame(counterfactual_array,columns=column)
AdCounterfactuals.to_csv("AdCounterfactuals.csv")