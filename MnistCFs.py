import numpy as np
import pandas as pd
from DataSets.DataLoader import MnistData
from extract_prototypes import ProtoTypes
from link_tree import LinkTree
from sklearn.neural_network import MLPClassifier
from generate_tree import Node

## 导入数据
mnist = MnistData()
X = mnist.data
y = mnist.target

## 划分不同类样本
zero_samples,one_samples,two_samples,three_samples,four_samples,five_samples,six_samples,seven_samples,eight_samples,nine_samples = [],[],[],[],[],[],[],[],[],[]          # 每一类的样本
non_zero_samples,non_one_samples,non_two_samples,non_three_samples,non_four_samples,non_five_samples,non_six_samples,non_seven_samples,non_eight_samples,non_nine_samples = [],[],[],[],[],[],[],[],[],[]
# 划分为对应类的样本和非对应类的样本
for index in range(len(X)):
    if y[index] == 0:
        zero_samples.append(X[index])
        non_one_samples.append(X[index])
        non_two_samples.append(X[index])
        non_three_samples.append(X[index])
        non_four_samples.append(X[index])
        non_five_samples.append(X[index])
        non_six_samples.append(X[index])
        non_seven_samples.append(X[index])
        non_eight_samples.append(X[index])
        non_nine_samples.append(X[index])
        pass
    elif y[index] == 1:
        non_zero_samples.append(X[index])
        one_samples.append(X[index])
        non_two_samples.append(X[index])
        non_three_samples.append(X[index])
        non_four_samples.append(X[index])
        non_five_samples.append(X[index])
        non_six_samples.append(X[index])
        non_seven_samples.append(X[index])
        non_eight_samples.append(X[index])
        non_nine_samples.append(X[index])
        pass
    elif y[index] == 2:
        non_zero_samples.append(X[index])
        non_one_samples.append(X[index])
        two_samples.append(X[index])
        non_three_samples.append(X[index])
        non_four_samples.append(X[index])
        non_five_samples.append(X[index])
        non_six_samples.append(X[index])
        non_seven_samples.append(X[index])
        non_eight_samples.append(X[index])
        non_nine_samples.append(X[index])
        pass
    elif y[index] == 3:
        non_zero_samples.append(X[index])
        non_one_samples.append(X[index])
        non_two_samples.append(X[index])
        three_samples.append(X[index])
        non_four_samples.append(X[index])
        non_five_samples.append(X[index])
        non_six_samples.append(X[index])
        non_seven_samples.append(X[index])
        non_eight_samples.append(X[index])
        non_nine_samples.append(X[index])
        pass
    elif y[index] == 4:
        non_zero_samples.append(X[index])
        non_one_samples.append(X[index])
        non_two_samples.append(X[index])
        non_three_samples.append(X[index])
        four_samples.append(X[index])
        non_five_samples.append(X[index])
        non_six_samples.append(X[index])
        non_seven_samples.append(X[index])
        non_eight_samples.append(X[index])
        non_nine_samples.append(X[index])
        pass
    elif y[index] == 5:
        non_zero_samples.append(X[index])
        non_one_samples.append(X[index])
        non_two_samples.append(X[index])
        non_three_samples.append(X[index])
        non_four_samples.append(X[index])
        five_samples.append(X[index])
        non_six_samples.append(X[index])
        non_seven_samples.append(X[index])
        non_eight_samples.append(X[index])
        non_nine_samples.append(X[index])
        pass
    elif y[index] == 6:
        non_zero_samples.append(X[index])
        non_one_samples.append(X[index])
        non_two_samples.append(X[index])
        non_three_samples.append(X[index])
        non_four_samples.append(X[index])
        non_five_samples.append(X[index])
        six_samples.append(X[index])
        non_seven_samples.append(X[index])
        non_eight_samples.append(X[index])
        non_nine_samples.append(X[index])
        pass
    elif y[index] == 7:
        non_zero_samples.append(X[index])
        non_one_samples.append(X[index])
        non_two_samples.append(X[index])
        non_three_samples.append(X[index])
        non_four_samples.append(X[index])
        non_five_samples.append(X[index])
        non_six_samples.append(X[index])
        seven_samples.append(X[index])
        non_eight_samples.append(X[index])
        non_nine_samples.append(X[index])
        pass
    elif y[index] == 8:
        non_zero_samples.append(X[index])
        non_one_samples.append(X[index])
        non_two_samples.append(X[index])
        non_three_samples.append(X[index])
        non_four_samples.append(X[index])
        non_five_samples.append(X[index])
        non_six_samples.append(X[index])
        non_seven_samples.append(X[index])
        eight_samples.append(X[index])
        non_nine_samples.append(X[index])
        pass
    elif y[index] == 9:
        non_zero_samples.append(X[index])
        non_one_samples.append(X[index])
        non_two_samples.append(X[index])
        non_three_samples.append(X[index])
        non_four_samples.append(X[index])
        non_five_samples.append(X[index])
        non_six_samples.append(X[index])
        non_seven_samples.append(X[index])
        non_eight_samples.append(X[index])
        nine_samples.append(X[index])
        pass
    pass

## 构建分类器，判断反事实类别是否符合要求
clf2 = MLPClassifier((100,10),max_iter=1000)
clf2.fit(X,y)           # 使用全部数据进行训练，测试时需要划分训练集和测试集

## 通过聚类中心点获取原型
ZeroProto = ProtoTypes(zero_samples,2)
OneProto = ProtoTypes(one_samples,2)
TwoProto = ProtoTypes(two_samples,2)
ThreeProto = ProtoTypes(three_samples,2)
FourProto = ProtoTypes(four_samples,2)
FiveProto = ProtoTypes(five_samples,2)
SixProto = ProtoTypes(six_samples,2)
SevenProto = ProtoTypes(seven_samples,2)
EightProto = ProtoTypes(eight_samples,2)
NineProto = ProtoTypes(nine_samples,2)

zero_proto,_ = ZeroProto.get_prototypes()
one_proto,_ = OneProto.get_prototypes()
two_proto,_ = TwoProto.get_prototypes()
three_proto,_ = ThreeProto.get_prototypes()
four_proto,_ = FourProto.get_prototypes()
five_proto,_ = FiveProto.get_prototypes()
six_proto,_ = SixProto.get_prototypes()
seven_proto,_ = SevenProto.get_prototypes()
eight_proto,_ = EightProto.get_prototypes()
nine_proto,_ = NineProto.get_prototypes()

# 转化为节点序列，便于使用生成链接树生成反事实
# 样本节点序列生成
ZeroNodes,OneNodes,TwoNodes,ThreeNodes,FourNodes,FiveNodes,SixNodes,SevenNodes,EightNodes,NineNodes = [],[],[],[],[],[],[],[],[],[]
NonZeroNodes,NonOneNodes,NonTwoNodes,NonThreeNodes,NonFourNodes,NonFiveNodes,NonSixNodes,NonSevenNodes,NonEightNodes,NonNineNodes = [],[],[],[],[],[],[],[],[],[]
for each in zero_samples:
    sample_node = []
    for value in each:
        sample_node.append(Node(value,"sample"))
        pass
    ZeroNodes.append(sample_node)
    pass
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
for each in eight_samples:
    sample_node = []
    for value in each:
        sample_node.append(Node(value,"sample"))
        pass
    EightNodes.append(sample_node)
    pass
for each in nine_samples:
    sample_node = []
    for value in each:
        sample_node.append(Node(value,"sample"))
        pass
    NineNodes.append(sample_node)
    pass
# 非某类样本节点生成
for each in non_zero_samples:
    sample_node = []
    for value in each:
        sample_node.append(Node(value,"sample"))
        pass
    NonZeroNodes.append(sample_node)
    pass
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
for each in non_eight_samples:
    sample_node = []
    for value in each:
        sample_node.append(Node(value,"sample"))
        pass
    NonEightNodes.append(sample_node)
    pass
for each in non_nine_samples:
    sample_node = []
    for value in each:
        sample_node.append(Node(value,"sample"))
        pass
    NonNineNodes.append(sample_node)
    pass

# 原型节点序列生成
zero_proto_nodes,one_proto_nodes,two_proto_nodes,three_proto_nodes,four_proto_nodes,five_proto_nodes,six_proto_nodes,seven_proto_nodes,eight_proto_nodes,nine_proto_nodes = [],[],[],[],[],[],[],[],[],[]
zero_proto_node,one_proto_node,two_proto_node,three_proto_node,four_proto_node,five_proto_node,six_proto_node,seven_proto_node,eight_proto_node,nine_proto_node = [],[],[],[],[],[],[],[],[],[]
for item in zero_proto:
    for e in item:
        zero_proto_node.append(Node(e,"proto"))
        pass
    zero_proto_nodes.append(zero_proto_node)
    pass
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
for item in eight_proto:
    for e in item:
        eight_proto_node.append(Node(e,"proto"))
        pass
    eight_proto_nodes.append(eight_proto_node)
    pass
for item in nine_proto:
    for e in item:
        nine_proto_node.append(Node(e,"proto"))
        pass
    nine_proto_nodes.append(nine_proto_node)
    pass

## 生成反事实
zero_CFs,one_CFs,two_CFs,three_CFs,four_CFs,five_CFs,six_CFs,seven_CFs,eight_CFs,nine_CFs = [],[],[],[],[],[],[],[],[],[]
# 对类别非0的数据生成类别为0的反事实
for each in NonZeroNodes:
    for zero_proto_node in zero_proto_nodes:
        tree = LinkTree(784,4,zero_proto_node,each)
        tree.gen_link()
        counterfactual = tree.create_CF()
        if clf2.predict(np.array(counterfactual).reshape(1,-1)) == 1:
            zero_CFs.append(counterfactual+[0])
            pass
        pass
    pass

# 与类别0同理
for each in NonOneNodes:
    for one_proto_node in one_proto_nodes:
        tree = LinkTree(784,4,one_proto_node,each)
        tree.gen_link()
        counterfactual = tree.create_CF()
        if clf2.predict(np.array(counterfactual).reshape(1,-1)) == 1:
            one_CFs.append(counterfactual+[1])
            pass
        pass
    pass
for each in NonTwoNodes:
    for two_proto_node in two_proto_nodes:
        tree = LinkTree(784,4,two_proto_node,each)
        tree.gen_link()
        counterfactual = tree.create_CF()
        if clf2.predict(np.array(counterfactual).reshape(1,-1)) == 2:
            two_CFs.append(counterfactual+[2])
            pass
        pass
    pass
for each in NonThreeNodes:
    for three_proto_node in three_proto_nodes:
        tree = LinkTree(784,4,three_proto_node,each)
        tree.gen_link()
        counterfactual = tree.create_CF()
        if clf2.predict(np.array(counterfactual).reshape(1,-1)) == 3:
            three_CFs.append(counterfactual+[3])
            pass
        pass
    pass
for each in NonFourNodes:
    for four_proto_node in four_proto_nodes:
        tree = LinkTree(784,4,four_proto_node,each)
        tree.gen_link()
        counterfactual = tree.create_CF()
        if clf2.predict(np.array(counterfactual).reshape(1,-1)) == 4:
            four_CFs.append(counterfactual+[4])
            pass
        pass
    pass
for each in NonFiveNodes:
    for five_proto_node in five_proto_nodes:
        tree = LinkTree(784,4,five_proto_node,each)
        tree.gen_link()
        counterfactual = tree.create_CF()
        if clf2.predict(np.array(counterfactual).reshape(1,-1)) == 5:
            five_CFs.append(counterfactual+[5])
            pass
        pass
    pass
for each in NonSixNodes:
    for six_proto_node in six_proto_nodes:
        tree = LinkTree(784,4,six_proto_node,each)
        tree.gen_link()
        counterfactual = tree.create_CF()
        if clf2.predict(np.array(counterfactual).reshape(1,-1)) == 6:
            six_CFs.append(counterfactual+[6])
            pass
        pass
    pass
for each in NonSevenNodes:
    for seven_proto_node in seven_proto_nodes:
        tree = LinkTree(784,4,seven_proto_node,each)
        tree.gen_link()
        counterfactual = tree.create_CF()
        if clf2.predict(np.array(counterfactual).reshape(1,-1)) == 7:
            seven_CFs.append(counterfactual+[7])
            pass
        pass
    pass
for each in NonEightNodes:
    for eight_proto_node in eight_proto_nodes:
        tree = LinkTree(784,4,eight_proto_node,each)
        tree.gen_link()
        counterfactual = tree.create_CF()
        if clf2.predict(np.array(counterfactual).reshape(1,-1)) == 6:
            eight_CFs.append(counterfactual+[8])
            pass
        pass
    pass
for each in NonNineNodes:
    for nine_proto_node in nine_proto_nodes:
        tree = LinkTree(784,4,seven_proto_node,each)
        tree.gen_link()
        counterfactual = tree.create_CF()
        if clf2.predict(np.array(counterfactual).reshape(1,-1)) == 7:
            nine_CFs.append(counterfactual+[9])
            pass
        pass
    pass

## 保存反事实
CFs = zero_CFs+one_CFs+two_CFs+three_CFs+four_CFs+five_CFs+six_CFs+seven_CFs+eight_CFs+nine_CFs
MnistCounterfactuals = pd.DataFrame(CFs,columns=mnist.mnist_data.columns.values)
MnistCounterfactuals.to_csv("MnistCounterfactuals.csv")