import pandas as pd
from DataSets.DataLoader import WallRobotSensor
from CreateCFsWithGLT import Creator
from sklearn.neural_network import MLPClassifier

## 导入数据
wallrobot = WallRobotSensor()
X = wallrobot.data
y = wallrobot.target

## 按类别划分样本
forward_samples,right_samples,sright_samples,left_samples = [],[],[],[]
non_forward_samples,non_right_samples,non_sright_samples,non_left_samples = [],[],[],[]
# 根据类标签划分对应类和非对应类的样本
for index in range(len(X)):
    if y[index] == 0:
        forward_samples.append(X[index])
        non_right_samples.append(X[index])
        non_sright_samples.append(X[index])
        non_left_samples.append(X[index])
        pass
    elif y[index] == 1:
        non_forward_samples.append(X[index])
        right_samples.append(X[index])
        non_sright_samples.append(X[index])
        non_left_samples.append(X[index])
        pass
    elif y[index] == 2:
        non_forward_samples.append(X[index])
        non_right_samples.append(X[index])
        sright_samples.append(X[index])
        non_left_samples.append(X[index])
        pass
    else:
        non_forward_samples.append(X[index])
        non_right_samples.append(X[index])
        non_sright_samples.append(X[index])
        left_samples.append(X[index])
        pass

## 构建分类器，判断反事实类别是否符合要求
clf = MLPClassifier((100,4),max_iter=1000)
clf.fit(X,y)

## 生成反事实
forwardCFs,rightCFs,srightCFs,leftCFs = [],[],[],[]
# 初始化反事实生成器
forwardCreator = Creator(clf,forward_samples,non_forward_samples,24,6,0,5)          # 设定为每个样本生成5个反事实解释
rightCreator = Creator(clf,right_samples,non_right_samples,24,6,1,5)
srightCreator = Creator(clf,sright_samples,non_sright_samples,24,6,2,5)
leftCreator = Creator(clf,left_samples,non_left_samples,24,6,3,5)

forwardCFs = forwardCreator.createCFs()
rightCFs = rightCreator.createCFs()
srightCFs = srightCreator.createCFs()
leftCFs = leftCreator.createCFs()

## 保存反事实
CFs = list(set(tuple(forwardCFs)+tuple(rightCFs)+tuple(srightCFs)+tuple(leftCFs)))
WallRobotCFs = pd.DataFrame(CFs,columns=wallrobot.sensor_data.columns.values)
WallRobotCFs.to_csv("WallRobotCFs-GLT.csv",index=False)