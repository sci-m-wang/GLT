import dice_ml
import numpy as np
from dice_ml import Dice
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from joblib import load
from DataSets.DataLoader import Climate
import pandas as pd
import os

PATH = os.path.join("/data/wm/GLT")

## 读取数据
climate = Climate()
df_climate = climate.load_df()
X = climate.data
y = climate.target

## 按照类别划分样本
failure_samples,success_samples = [],[]
for index in range(len(X)):
    if y[index] == 0:
        failure_samples.append(X[index])
        pass
    else:
        success_samples.append(X[index])
        pass
    pass

## 使用遗传算法生成反事实
outcome_name = "outcome"
continuous_features_climate = df_climate.drop(outcome_name,axis=1).columns.tolist()
target = df_climate[outcome_name]
datasetX = df_climate.drop(outcome_name,axis=1)
categorical_features = datasetX.columns.difference(continuous_features_climate)
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

transformations = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, continuous_features_climate),
        ('cat', categorical_transformer, categorical_features)])
transformations.fit(datasetX,target)
clf_climate = Pipeline(steps=[('preprocessor', transformations),
                           ('classifier', load("clf.joblib"))])

d_climate = dice_ml.Data(dataframe=df_climate,
                      continuous_features=continuous_features_climate,
                      outcome_name=outcome_name)
m_climate = dice_ml.Model(model=clf_climate,
                       backend="sklearn",
                       model_type="classifier")
exp_kdtree_climate = Dice(d_climate,m_climate,method="kdtree")

n_cfs = 2
query_instances_failure = pd.DataFrame(np.array(success_samples),columns=df_climate.columns.values[:-1])
query_instances_success = pd.DataFrame(np.array(failure_samples),columns=df_climate.columns.values[:-1])

generate_failure = exp_kdtree_climate.generate_counterfactuals(query_instances_failure,total_CFs=100,desired_class=0)
final_failure_cfs = [None]*n_cfs
for i in range(n_cfs):
    final_failure_cfs[i] = tuple(generate_failure.cf_examples_list[0].final_cfs_df.values[i])
    pass
generate_success = exp_kdtree_climate.generate_counterfactuals(query_instances_success,total_CFs=100,desired_class=1)
final_success_cfs = [None]*n_cfs
for i in range(n_cfs):
    final_success_cfs[i] = tuple(generate_success.cf_examples_list[0].final_cfs_df.values[i])
    pass

CFs = list(set(tuple(final_success_cfs)+tuple(final_failure_cfs)))
ClimateCFs = pd.DataFrame(CFs,columns=climate.climate_data.columns.values)
ClimateCFs.to_csv("ClimateCFs-KDTree.csv",index=True)