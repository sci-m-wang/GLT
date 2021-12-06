import pandas as pd
from DataSets.DataLoader import Fertility
from CreateCFsWithGLT import Creator
from sklearn.neural_network import MLPClassifier

## 导入数据
fertility = Fertility()
X = fertility.data
y = fertility.target

## 按类别划分样本
normal_samples,altered_samples = [],[]
for index in range(len(X)):
    if y[index] == 0:
        normal_samples.append(X[index])
        pass
    else:
        altered_samples.append(X[index])
        pass
    pass

## 构建分类器，验证反事实
clf = MLPClassifier((100,2),max_iter=1000)
clf.fit(X,y)

## 生成反事实
normalCFs,alteredCFs = [],[]
# 初始化反事实生成器
normalCreator = Creator(clf,normal_samples,altered_samples,9,3,0,1)
alteredCreator = Creator(clf,altered_samples,normal_samples,9,3,1,1)

normalCFs = normalCreator.createCFs()
alteredCFs = alteredCreator.createCFs()

CFs = list(set(tuple(normalCFs)+tuple(alteredCFs)))
FertilityCFs = pd.DataFrame(CFs,columns=fertility.fertility_data.columns.values)
FertilityCFs.to_csv("FertilityCFs-GLT.csv",index=True)