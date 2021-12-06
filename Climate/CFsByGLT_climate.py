import pandas as pd
from DataSets.DataLoader import Climate
from CreateCFsWithGLT import Creator
from joblib import load

## 导入数据
climate = Climate()
X = climate.data
y = climate.target

## 按类别划分样本
failure_samples,success_samples = [],[]
for index in range(len(X)):
    if y[index] == 0:
        failure_samples.append(X[index])
        pass
    else:
        success_samples.append(X[index])
        pass
    pass

## 导入分类器，验证反事实
clf = load("clf.joblib")

## 生成反事实
failureCFs,successCFs = [],[]
# 初始化反事实生成器
failureCreator = Creator(clf,failure_samples,success_samples,18,6,0,2)
successCreator = Creator(clf,success_samples,failure_samples,18,6,1,2)

failureCFs = failureCreator.createCFs()
successCFs = successCreator.createCFs()

CFs = list(set(tuple(failureCFs)+tuple(successCFs)))
ClimateCFs = pd.DataFrame(CFs,columns=climate.climate_data.columns.values)
ClimateCFs.to_csv("ClimateCFs-GLT.csv",index=True)