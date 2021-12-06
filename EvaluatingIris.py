from random import randint
import dice_ml
import numpy as np
from dice_ml import Dice
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from DataSets.DataLoader import IrisData
import pandas as pd
import os
from time import time

PATH = os.path.join("/data/wm/GLT/")

## 读取数据
iris = IrisData()
df_iris = iris.load_df()
X = iris.data
y = iris.target
# 划分不同类别样本
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

## 使用基于遗传算法生成反事实
start_time = time()
outcome_name = "class"
continuous_features_iris = df_iris.drop(outcome_name,axis=1).columns.tolist()
target = df_iris[outcome_name]

datasetX = df_iris.drop(outcome_name, axis=1)
x_train, x_test, y_train, y_test = train_test_split(datasetX,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=target)

categorical_features = x_train.columns.difference(continuous_features_iris)

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

transformations = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, continuous_features_iris),
        ('cat', categorical_transformer, categorical_features)])


clf_iris = Pipeline(steps=[('preprocessor', transformations),
                           ('classifier', KNeighborsClassifier())])
model_iris = clf_iris.fit(x_train, y_train)

d_iris = dice_ml.Data(dataframe=df_iris,
                      continuous_features=continuous_features_iris,
                      outcome_name=outcome_name)
m_iris = dice_ml.Model(model=model_iris,
                       backend="sklearn",
                       model_type="classifier")

exp_genetic_iris = Dice(d_iris,m_iris,method="genetic")

n_cfs = 1

index = randint(0,99)
query_instances_setosa = pd.DataFrame(np.array(non_setosa_samples[index]).reshape(1,-1),columns=df_iris.columns.values[:-1])
index = randint(0,99)
query_instances_versicolor = pd.DataFrame(np.array(non_versicolor_samples[index]).reshape(1,-1),columns=df_iris.columns.values[:-1])
index = randint(0,99)
query_instances_virginica = pd.DataFrame(np.array(non_virginica_samples[index]).reshape(1,-1),columns=df_iris.columns.values[:-1])
# 单个输入样例的测试
# 非1类生成1类
genetic_setosa = exp_genetic_iris.generate_counterfactuals(query_instances_setosa,total_CFs=n_cfs,desired_class=0)
final_setosa_cfs_di = [None]*n_cfs
for i in range(n_cfs):
    final_setosa_cfs_di[i] = genetic_setosa.cf_examples_list[0].final_cfs_df.values[i]
    pass
# 非2类生成2类
genetic_versicolor = exp_genetic_iris.generate_counterfactuals(query_instances_versicolor,total_CFs=n_cfs,desired_class=1)
final_versicolor_cfs_di = [None]*n_cfs
for i in range(n_cfs):
    final_versicolor_cfs_di[i] = genetic_versicolor.cf_examples_list[0].final_cfs_df.values[i]
    pass
# 非3类生成3类
genetic_virginica = exp_genetic_iris.generate_counterfactuals(query_instances_virginica,total_CFs=n_cfs,desired_class=2)
final_virginica_cfs_di = [None]*n_cfs
for i in range(n_cfs):
    final_virginica_cfs_di[i] = genetic_virginica.cf_examples_list[0].final_cfs_df.values[i]
    pass

end_time = time()
print("genetic_time:",end_time-start_time)
## 使用基于KD树的方法生成反事实
start_time = time()
exp_kdtree_iris = Dice(d_iris,m_iris,method="kdtree")
kdtree_setosa = exp_kdtree_iris.generate_counterfactuals(query_instances_setosa,total_CFs=n_cfs,desired_class=0)
final_setosa_cfs_kd = [None]*n_cfs
for i in range(n_cfs):
    final_setosa_cfs_kd[i] = kdtree_setosa.cf_examples_list[0].final_cfs_df.values[i]
    pass
kdtree_versicolor = exp_kdtree_iris.generate_counterfactuals(query_instances_versicolor,total_CFs=n_cfs,desired_class=0)
final_versicolor_cfs_kd = [None]*n_cfs
for i in range(n_cfs):
    final_versicolor_cfs_kd[i] = kdtree_versicolor.cf_examples_list[0].final_cfs_df.values[i]
    pass
kdtree_virginica = exp_kdtree_iris.generate_counterfactuals(query_instances_virginica,total_CFs=n_cfs,desired_class=0)
final_virginica_cfs_kd = [None]*n_cfs
for i in range(n_cfs):
    final_virginica_cfs_kd[i] = kdtree_virginica.cf_examples_list[0].final_cfs_df.values[i]
    pass

end_time = time()
print("ketree_time:",end_time-start_time)

## 使用随机样本生成反事实
start_time = time()
exp_random_iris = Dice(d_iris,m_iris,method="random")
random_setosa = exp_random_iris.generate_counterfactuals(query_instances_setosa,total_CFs=n_cfs,desired_class=0)
final_setosa_cfs_ra = [None]*n_cfs
for i in range(n_cfs):
    final_setosa_cfs_ra[i] = random_setosa.cf_examples_list[0].final_cfs_df.values[i]
    pass
random_versicolor = exp_random_iris.generate_counterfactuals(query_instances_versicolor,total_CFs=n_cfs,desired_class=0)
final_versicolor_cfs_ra = [None]*n_cfs
for i in range(n_cfs):
    final_versicolor_cfs_ra[i] = random_versicolor.cf_examples_list[0].final_cfs_df.values[i]
    pass
random_virginica = exp_random_iris.generate_counterfactuals(query_instances_virginica,total_CFs=n_cfs,desired_class=0)
final_virginica_cfs_ra = [None]*n_cfs
for i in range(n_cfs):
    final_virginica_cfs_ra[i] = random_virginica.cf_examples_list[0].final_cfs_df.values[i]
    pass

end_time = time()
print("ketree_time:",end_time-start_time)

## 使用GLT-searching方法生成反事实
start_time = time()
GLT_iris_cfs = pd.read_csv(PATH+"IrisCounterfactuals.csv")          # 读取使用GLT方法生成的反事实库
# 划分不同类的反事实
setosa_cfs = np.array(GLT_iris_cfs[GLT_iris_cfs["class"] == 1].drop("class",axis=1))
versicolor_cfs = np.array(GLT_iris_cfs[GLT_iris_cfs["class"] == 2].drop("class",axis=1))
virginica_cfs = np.array(GLT_iris_cfs[GLT_iris_cfs["class"] == 3].drop("class",axis=1))
# 非1类生成1类
max_similarity = [0]*n_cfs
final_setosa_cfs = [None]*n_cfs
for each in setosa_cfs:
    query = np.array(query_instances_setosa)
    target = np.array(each)
    cost = 1/(1+np.exp(np.sqrt(np.sum((query-target) ** 2))/np.sqrt(np.sqrt(np.sum(query ** 2)))*np.sqrt(np.sqrt(np.sum(target ** 2)))))
    simi = np.exp(-np.sqrt(np.sum((query-target) ** 2))/np.sqrt(np.sqrt(np.sum(query ** 2)))*np.sqrt(np.sqrt(np.sum(target ** 2))))
    relative_similarity = simi/cost
    for i in range(len(max_similarity)):
        if relative_similarity >= max_similarity[i]:
            max_similarity[i + 1:] = max_similarity[i:-1]
            max_similarity[i] = relative_similarity
            final_setosa_cfs[i + 1:] = final_setosa_cfs[i:-1]
            final_setosa_cfs[i] = each
            break
            pass
        pass
    pass

# 非2类生成2类
max_similarity = [0]*n_cfs
final_versicolor_cfs = [None]*n_cfs
for each in versicolor_cfs:
    query = np.array(query_instances_versicolor)
    target = np.array(each)
    # if model_iris.predict(target.reshape(1,-1)) == 2:
    cost = 1/(1+np.exp(np.sqrt(np.sum((query-target) ** 2))/np.sqrt(np.sqrt(np.sum(query ** 2)))*np.sqrt(np.sqrt(np.sum(target ** 2)))))
    simi = np.exp(-np.sqrt(np.sum((query-target) ** 2))/np.sqrt(np.sqrt(np.sum(query ** 2)))*np.sqrt(np.sqrt(np.sum(target ** 2))))
    relative_similarity = simi/cost
    for i in range(len(max_similarity)):
        if relative_similarity >= max_similarity[i]:
            max_similarity[i + 1:] = max_similarity[i:-1]
            max_similarity[i] = relative_similarity
            final_versicolor_cfs[i + 1:] = final_versicolor_cfs[i:-1]
            final_versicolor_cfs[i] = each
            break
            pass
        pass
    pass
    # pass
# 非3类生成3类
max_similarity = [0]*n_cfs
final_virginica_cfs = [None]*n_cfs
for each in virginica_cfs:
    query = np.array(query_instances_virginica)
    target = np.array(each)
    # if model_iris.predict(target.reshape(1,-1)) == 3:
    cost = 1/(1+np.exp(np.sqrt(np.sum((query-target) ** 2))/np.sqrt(np.sqrt(np.sum(query ** 2)))*np.sqrt(np.sqrt(np.sum(target ** 2)))))
    simi = np.exp(-np.sqrt(np.sum((query-target) ** 2))/np.sqrt(np.sqrt(np.sum(query ** 2)))*np.sqrt(np.sqrt(np.sum(target ** 2))))
    relative_similarity = simi/cost
    for i in range(len(max_similarity)):
        if relative_similarity >= max_similarity[i]:
            max_similarity[i + 1:] = max_similarity[i:-1]
            max_similarity[i] = relative_similarity
            final_virginica_cfs[i + 1:] = final_virginica_cfs[i:-1]
            final_virginica_cfs[i] = each
            break
            pass
        pass
    pass
    # pass

end_time = time()
print(end_time-start_time)

## 反事实距离对比
setosa_di_distance,versicolor_di_distance,virginica_di_distance = 0,0,0
setosa_distance,versicolor_distance,virginica_distance = 0,0,0

for each in final_setosa_cfs_di:
    setosa_di_distance += np.sqrt(np.sum((np.array(query_instances_setosa)-np.array(each)[:-1]) ** 2))
    pass
setosa_di_distance /= n_cfs
for each in final_versicolor_cfs_di:
    versicolor_di_distance += np.sqrt(np.sum((np.array(query_instances_versicolor)-np.array(each)[:-1]) ** 2))
    pass
versicolor_di_distance /= n_cfs
for each in final_virginica_cfs_di:
    virginica_di_distance += np.sqrt(np.sum((np.array(query_instances_virginica)-np.array(each)[:-1]) ** 2))
    pass
virginica_di_distance /= n_cfs
distance_di = (setosa_di_distance+versicolor_di_distance+virginica_di_distance)/3

for each in final_setosa_cfs:
    setosa_distance += np.sqrt(np.sum((np.array(query_instances_setosa)-np.array(each)) ** 2))
    pass
setosa_distance /= n_cfs
for each in final_versicolor_cfs:
    versicolor_distance += np.sqrt(np.sum((np.array(query_instances_versicolor)-np.array(each)) ** 2))
    pass
versicolor_distance /= n_cfs
for each in final_virginica_cfs:
    virginica_distance += np.sqrt(np.sum((np.array(query_instances_virginica)-np.array(each)) ** 2))
    pass
virginica_distance /= n_cfs
distance = (setosa_distance+versicolor_distance+virginica_distance)/3

print(distance_di,distance)