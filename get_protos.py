from DataSets.DataLoader import InternetAdData
import numpy as np

ad = InternetAdData()

X = ad.data
y = ad.target

ad_samples = []
nonad_samples = []

# 根据分类选择数据，是广告为1，不是广告为0
for index in range(len(X)):
    if y[index] == 1:
        ad_samples.append(X[index])
        pass
    else:
        nonad_samples.append(X[index])
        pass
    pass
count = 0
for each in ad_samples:
    for item in nonad_samples:
        if np.sum(1-(each == item)) <= 0:
            count += 1
            pass
        pass
    pass
print(count)
print(count/len(ad_samples)/len(nonad_samples))