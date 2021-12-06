import numpy as np
import pandas as pd
from DataSets.DataLoader import CoverTypeData
from extract_prototypes import ProtoTypes
from link_tree import LinkTree
from sklearn.neural_network import MLPClassifier
from generate_tree import Node

## 导入数据
cover = CoverTypeData()
X = cover.data
y = cover.target

## 划分不同类样本
one_samples,two_samples,three_samples,four_samples,five_samples,six_samples,seven_samples = [],[],[],[],[],[],[]          # 每一类的样本
non_one_samples,non_two_samples,non_three_samples,non_four_samples,non_five_samples,non_six_samples,non_seven_samples = [],[],[],[],[],[],[]
# 划分为对应类的样本和非对应类的样本
for index in range(len(X)):
    if y[index] == 1:
        one_samples.append(X[index])
        non_two_samples.append(X[index])
        non_three_samples.append(X[index])
        non_four_samples.append(X[index])
        non_five_samples.append(X[index])
        non_six_samples.append(X[index])
        non_seven_samples.append(X[index])
        pass
    elif y[index] == 2:
        non_one_samples.append(X[index])
        two_samples.append(X[index])
        non_three_samples.append(X[index])
        non_four_samples.append(X[index])
        non_five_samples.append(X[index])
        non_six_samples.append(X[index])
        non_seven_samples.append(X[index])
        pass
    elif y[index] == 3:
        non_one_samples.append(X[index])
        non_two_samples.append(X[index])
        three_samples.append(X[index])
        non_four_samples.append(X[index])
        non_five_samples.append(X[index])
        non_six_samples.append(X[index])
        non_seven_samples.append(X[index])
        pass
    elif y[index] == 4:
        non_one_samples.append(X[index])
        non_two_samples.append(X[index])
        non_three_samples.append(X[index])
        four_samples.append(X[index])
        non_five_samples.append(X[index])
        non_six_samples.append(X[index])
        non_seven_samples.append(X[index])
        pass
    elif y[index] == 5:
        non_one_samples.append(X[index])
        non_two_samples.append(X[index])
        non_three_samples.append(X[index])
        non_four_samples.append(X[index])
        five_samples.append(X[index])
        non_six_samples.append(X[index])
        non_seven_samples.append(X[index])
        pass
    elif y[index] == 6:
        non_one_samples.append(X[index])
        non_two_samples.append(X[index])
        non_three_samples.append(X[index])
        non_four_samples.append(X[index])
        non_five_samples.append(X[index])
        six_samples.append(X[index])
        non_seven_samples.append(X[index])
        pass
    elif y[index] == 7:
        non_one_samples.append(X[index])
        non_two_samples.append(X[index])
        non_three_samples.append(X[index])
        non_four_samples.append(X[index])
        non_five_samples.append(X[index])
        non_six_samples.append(X[index])
        seven_samples.append(X[index])
        pass
    pass

## 构建分类器，判断反事实类别是否符合要求
clf2 = MLPClassifier((100,7),max_iter=1000)
clf2.fit(X,y)           # 使用全部数据进行训练，测试时需要划分训练集和测试集

## 通过聚类中心点获取原型
OneProto = ProtoTypes(one_samples,2)
TwoProto = ProtoTypes(two_samples,2)
ThreeProto = ProtoTypes(three_samples,2)
FourProto = ProtoTypes(four_samples,2)
FiveProto = ProtoTypes(five_samples,2)
SixProto = ProtoTypes(six_samples,2)
SevenProto = ProtoTypes(seven_samples,2)

one_proto,_ = OneProto.get_prototypes()
two_proto,_ = TwoProto.get_prototypes()
three_proto,_ = ThreeProto.get_prototypes()
four_proto,_ = FourProto.get_prototypes()
five_proto,_ = FiveProto.get_prototypes()
six_proto,_ = SixProto.get_prototypes()
seven_proto,_ = SevenProto.get_prototypes()

# 转化为节点序列，便于使用生成链接树生成反事实
# 样本节点序列生成
OneNodes,TwoNodes,ThreeNodes,FourNodes,FiveNodes,SixNodes,SevenNodes = [],[],[],[],[],[],[]
NonOneNodes,NonTwoNodes,NonThreeNodes,NonFourNodes,NonFiveNodes,NonSixNodes,NonSevenNodes = [],[],[],[],[],[],[]
for each in one_samples:
    sample_node = []
    for value in each:
        sample_node.append(Node(value,"sample"))
        pass
    OneNodes.append(sample_node)
    pass
for each in two_samples:
    sample_node = []
    for value in each:
        sample_node.append(Node(value,"sample"))
        pass
    TwoNodes.append(sample_node)
    pass
for each in three_samples:
    sample_node = []
    for value in each:
        sample_node.append(Node(value,"sample"))
        pass
    ThreeNodes.append(sample_node)
    pass
for each in four_samples:
    sample_node = []
    for value in each:
        sample_node.append(Node(value,"sample"))
        pass
    FourNodes.append(sample_node)
    pass
for each in five_samples:
    sample_node = []
    for value in each:
        sample_node.append(Node(value,"sample"))
        pass
    FiveNodes.append(sample_node)
    pass
for each in six_samples:
    sample_node = []
    for value in each:
        sample_node.append(Node(value,"sample"))
        pass
    SixNodes.append(sample_node)
    pass
for each in seven_samples:
    sample_node = []
    for value in each:
        sample_node.append(Node(value,"sample"))
        pass
    SevenNodes.append(sample_node)
    pass
# 非某类样本节点生成
for each in non_one_samples:
    sample_node = []
    for value in each:
        sample_node.append(Node(value,"sample"))
        pass
    NonOneNodes.append(sample_node)
    pass
for each in non_two_samples:
    sample_node = []
    for value in each:
        sample_node.append(Node(value,"sample"))
        pass
    NonTwoNodes.append(sample_node)
    pass
for each in non_three_samples:
    sample_node = []
    for value in each:
        sample_node.append(Node(value,"sample"))
        pass
    NonThreeNodes.append(sample_node)
    pass
for each in non_four_samples:
    sample_node = []
    for value in each:
        sample_node.append(Node(value,"sample"))
        pass
    NonFourNodes.append(sample_node)
    pass
for each in non_five_samples:
    sample_node = []
    for value in each:
        sample_node.append(Node(value,"sample"))
        pass
    NonFiveNodes.append(sample_node)
    pass
for each in non_six_samples:
    sample_node = []
    for value in each:
        sample_node.append(Node(value,"sample"))
        pass
    NonSixNodes.append(sample_node)
    pass
for each in non_seven_samples:
    sample_node = []
    for value in each:
        sample_node.append(Node(value,"sample"))
        pass
    NonSevenNodes.append(sample_node)
    pass

# 原型节点序列生成
one_proto_nodes,two_proto_nodes,three_proto_nodes,four_proto_nodes,five_proto_nodes,six_proto_nodes,seven_proto_nodes = [],[],[],[],[],[],[]
one_proto_node,two_proto_node,three_proto_node,four_proto_node,five_proto_node,six_proto_node,seven_proto_node = [],[],[],[],[],[],[]
for item in one_proto:
    for e in item:
        one_proto_node.append(Node(e,"proto"))
        pass
    one_proto_nodes.append(one_proto_node)
    pass
for item in two_proto:
    for e in item:
        two_proto_node.append(Node(e,"proto"))
        pass
    two_proto_nodes.append(one_proto_node)
    pass
for item in three_proto:
    for e in item:
        three_proto_node.append(Node(e,"proto"))
        pass
    three_proto_nodes.append(one_proto_node)
    pass
for item in four_proto:
    for e in item:
        four_proto_node.append(Node(e,"proto"))
        pass
    four_proto_nodes.append(one_proto_node)
    pass
for item in five_proto:
    for e in item:
        five_proto_node.append(Node(e,"proto"))
        pass
    five_proto_nodes.append(one_proto_node)
    pass
for item in six_proto:
    for e in item:
        six_proto_node.append(Node(e,"proto"))
        pass
    six_proto_nodes.append(one_proto_node)
    pass
for item in seven_proto:
    for e in item:
        seven_proto_node.append(Node(e,"proto"))
        pass
    seven_proto_nodes.append(one_proto_node)
    pass

## 生成反事实
one_CFs,two_CFs,three_CFs,four_CFs,five_CFs,six_CFs,seven_CFs = [],[],[],[],[],[],[]
# 对类别非1的数据生成类别为1的反事实
for each in NonOneNodes:
    for one_proto_node in one_proto_nodes:
        tree = LinkTree(54,9,one_proto_node,each)
        tree.gen_link()
        counterfactual = tree.create_CF()
        if clf2.predict(np.array(counterfactual).reshape(1,-1)) == 1:
            one_CFs.append(counterfactual+[1])
            pass
        pass
    pass

# 与类别1同理
for each in NonTwoNodes:
    for two_proto_node in two_proto_nodes:
        tree = LinkTree(54,9,two_proto_node,each)
        tree.gen_link()
        counterfactual = tree.create_CF()
        if clf2.predict(np.array(counterfactual).reshape(1,-1)) == 2:
            two_CFs.append(counterfactual+[2])
            pass
        pass
    pass
for each in NonThreeNodes:
    for three_proto_node in three_proto_nodes:
        tree = LinkTree(54,9,three_proto_node,each)
        tree.gen_link()
        counterfactual = tree.create_CF()
        if clf2.predict(np.array(counterfactual).reshape(1,-1)) == 3:
            three_CFs.append(counterfactual+[3])
            pass
        pass
    pass
for each in NonFourNodes:
    for four_proto_node in four_proto_nodes:
        tree = LinkTree(54,9,four_proto_node,each)
        tree.gen_link()
        counterfactual = tree.create_CF()
        if clf2.predict(np.array(counterfactual).reshape(1,-1)) == 4:
            four_CFs.append(counterfactual+[4])
            pass
        pass
    pass
for each in NonFiveNodes:
    for five_proto_node in five_proto_nodes:
        tree = LinkTree(54,9,five_proto_node,each)
        tree.gen_link()
        counterfactual = tree.create_CF()
        if clf2.predict(np.array(counterfactual).reshape(1,-1)) == 5:
            five_CFs.append(counterfactual+[5])
            pass
        pass
    pass
for each in NonSixNodes:
    for six_proto_node in six_proto_nodes:
        tree = LinkTree(54,9,six_proto_node,each)
        tree.gen_link()
        counterfactual = tree.create_CF()
        if clf2.predict(np.array(counterfactual).reshape(1,-1)) == 6:
            six_CFs.append(counterfactual+[6])
            pass
        pass
    pass
for each in NonSevenNodes:
    for seven_proto_node in seven_proto_nodes:
        tree = LinkTree(54,9,seven_proto_node,each)
        tree.gen_link()
        counterfactual = tree.create_CF()
        if clf2.predict(np.array(counterfactual).reshape(1,-1)) == 7:
            seven_CFs.append(counterfactual+[7])
            pass
        pass
    pass

## 保存反事实
CFs = one_CFs+two_CFs+three_CFs+four_CFs+five_CFs+six_CFs+seven_CFs
CoverCounterfactuals = pd.DataFrame(CFs,columns=cover.columns)
CoverCounterfactuals.to_csv("CoverCounterfactuals.csv")