import numpy as np
import pandas as pd
import os

PATH = os.path.join("/data/wm/GLT/DataSets/")

class Fertility():
    """
    Instances: 100
    Attributes: 9
    - 季节: winter2fall(-1, -0.33, 0.33, 1)
    - 年龄: 18-36(0,1)
    - 幼儿疾病: yes/no(0,1)
    - 严重创伤: yes/no(0,1)
    - 手术干预: yes/no(0,1)
    - 过去一年高烧: 3-/3+/no(-1,0,1)
    - 饮酒频率: servel per day 2 hardly ever or never(0,1)
    - 吸烟习惯: never 2 daily(-1,0,1)
    - 每天坐的时长: (0,1)
    Classes: 2
    - 诊断: 正常/替换
    """
    columns = ['season','age','childish_diseases','accident','surgical','high_fever','alcohol','smoking','sitting','class']
    fertility_data = pd.read_csv(PATH+'fertility_Diagnosis.csv',names=columns)
    fertility_data['class'].replace("N",0,inplace=True)
    fertility_data['class'].replace("O",1,inplace=True)
    fertility_data = pd.DataFrame(fertility_data)
    fertility_array = np.array(fertility_data)
    def __init__(self):
        self.data = self.fertility_array[:,:-1]
        self.target = self.fertility_array[:,-1]
        pass
    def load_data(self):
        return self.fertility_array
    def load_df(self):
        return self.fertility_data

class Climate():
    """
    Instances: 540
    Attributes: 18
    - 18个气象模型的参数[0,1]
    Classes: 2
    - 模拟结果: 成功1/失败0
    """
    climate_data = pd.read_csv(PATH+'pop_failures.csv',header=0)
    climate_data = pd.DataFrame(climate_data)
    climate_array = np.array(climate_data)
    def __init__(self):
        self.data = self.climate_array[:,:-1]
        self.target = self.climate_array[:,-1]
        pass
    def load_data(self):
        return self.climate_array
    def load_df(self):
        return self.climate_data

class WallRobotSensor():
    """
    Instances: 5456
    Attributes: 24
    - 24个传感器的值
    Classes: 4
    - 向前走: Move-Forward -> 0
    - 小幅右转: Slight-Right-Turn -> 1
    - 急速右转: Sharp-Right-Turn -> 2
    - 小幅左转: Slight-Left-Turn -> 3
    """
    columns = ['S'+str(i) for i in range(1,25)]+['class']
    sensor_data = pd.read_csv(PATH+"sensor_readings_24.csv",names=columns)
    sensor_data["class"].replace("Move-Forward",0,inplace=True)
    sensor_data["class"].replace("Slight-Right-Turn", 1,inplace=True)
    sensor_data["class"].replace("Sharp-Right-Turn", 2,inplace=True)
    sensor_data["class"].replace("Slight-Left-Turn", 3,inplace=True)
    sensor_data = pd.DataFrame(sensor_data)
    sensor_array = np.array(sensor_data)
    def __init__(self):
        self.data = self.sensor_array[:,:-1]
        self.target = self.sensor_array[:,-1]
        pass
    def load_data(self):
        return self.sensor_array
    def load_df(self):
        return self.sensor_data
    pass

class CarData():
    """
    Instances: 1728
    Attributes: 6
    - 购买价格: v-high, high, med, low
    - 维修价格: v-high, high, med, low
    - 车门数:   2, 3, 4, 5-more
    - 核载人数: 2, 4, more
    - 后备箱:   small, med, big
    - 安全性能: low, med, high
    Classes：4->2
    - unacc  -> unacc(70.023%)
    - acc    -> acc(22.222%)
    - good   -> acc(3.993%)
    - v-good -> acc(3.762%)
    """
    columns = ["buy", "maint", "doors", "persons", "lug_boot", "safety", "if_acc"]
    car_data = pd.read_csv(PATH + "car.csv", names=columns)
    car_data["if_acc"].replace("good", "acc", inplace=True)
    car_data["if_acc"].replace("vgood", "acc", inplace=True)

    # 对分类特征进行编码
    car_data["buy"].replace("vhigh", 7, inplace=True)
    car_data["buy"].replace("high", 5, inplace=True)
    car_data["buy"].replace("med", 3, inplace=True)
    car_data["buy"].replace("low", 1, inplace=True)
    car_data["maint"].replace("vhigh", 7, inplace=True)
    car_data["maint"].replace("high", 5, inplace=True)
    car_data["maint"].replace("med", 3, inplace=True)
    car_data["maint"].replace("low", 1, inplace=True)
    car_data["doors"].replace("5more", 5, inplace=True)
    car_data["doors"].replace("4", 4, inplace=True)
    car_data["doors"].replace("3", 3, inplace=True)
    car_data["doors"].replace("2", 2, inplace=True)
    car_data["persons"].replace("more", 6, inplace=True)
    car_data["persons"].replace("4", 4, inplace=True)
    car_data["persons"].replace("2", 2, inplace=True)
    car_data["lug_boot"].replace("big", 5, inplace=True)
    car_data["lug_boot"].replace("med", 3, inplace=True)
    car_data["lug_boot"].replace("small", 1, inplace=True)
    car_data["safety"].replace("high", 5, inplace=True)
    car_data["safety"].replace("med", 3, inplace=True)
    car_data["safety"].replace("low", 1, inplace=True)

    # 对类别进行编码
    car_data["if_acc"].replace("acc", 1, inplace=True)
    car_data["if_acc"].replace("unacc", 0, inplace=True)

    car_data = pd.DataFrame(car_data)
    car_data = np.array(car_data)
    def __init__(self):
        self.data = self.car_data[:,:-1]
        self.target = self.car_data[:,-1]
        pass
    def load_original_data(self):
        return self.car_data
    def load_data(self):
        return self.car_data

    pass

class InternetAdData():
    """
    Instances: 3279
    Attributes: 1555
    - 457 features from url terms
    - 495 features from original url terms
    - 472 features from ancurl terms
    - 111 features from alt terms
    - 19 features from caption terms
    Classes: 2
    - nonads: 2821
    - ads:458
    """
    ad_data = pd.read_csv(PATH+"Internet_ad.csv",header=0)
    ad_data = pd.DataFrame(ad_data)
    ad_array = np.array(ad_data)
    def __init__(self):
        self.data = self.ad_array[:,:-1]
        self.target = self.ad_array[:,-1]
        pass
    def load_data(self):
        return self.ad_array
    def load_df(self):
        return self.ad_data
    pass

class CoverTypeData():
    """
    Instances: 581012
    Attributes: 54
    - 海拔: 定量，米
    - 方位角: 定量，方位角
    - 斜率: 定量，角度数
    - 水平最近地表水距离: 定量，米
    - 垂直最近地表水距离: 定量，米
    - 水平最近道路距离: 定量，米
    - 上午九点山影指数: 定量，0-255
    - 中午山影指数: 定量，0-255
    - 下午三点山影指数: 定量，0-255
    - 水平最近野外生火点距离: 定量，米
    - 野外区域: 4位2进制
    - 土壤类型: 40位2进制
    Classes: 7
    - 森林覆盖类型: 整数，1-7表示7种不同的覆盖率
    """
    wild_area = ["W1","W2","W3","W4"]
    soil_type = ["S"+str(i) for i in range(1,41)]
    columns = ["Elevation","Aspect","Slope","Horiz_Dist_to_Hydro","Verti_Dist_to_Hydro","Horiz_Dist_to_Road","Hillshade_9am","Hillshade_noon","Hillshade_3pm","Horiz_Dist_to_Firepoints"]+wild_area+soil_type+["CoverType"]
    cover_data = pd.read_csv(PATH+"covtype.csv",names=columns)
    cover_data = pd.DataFrame(cover_data)
    cover_array = np.array(cover_data)
    def __init__(self):
        self.data = self.cover_array[:,:-1]
        self.target = self.cover_array[:,-1]
        pass
    def load_data(self):
        return self.cover_array
    def load_df(self):
        return self.cover_data
    pass
class MnistData():
    """
    Instances: 10001
    Attributes: 784
    - 像素点: 0-255
    Classes: 10
    - 数字: 0-9
    """
    mnist_data = pd.read_csv(PATH+"mnist_test.csv",header=0)
    mnist_data = pd.DataFrame(mnist_data)
    mnist_array = np.array(mnist_data)
    def __init__(self):
        self.data = self.mnist_array[:,1:]
        self.target = self.mnist_array[:,0]
        pass
    def load_data(self):
        return self.mnist_array
    def load_df(self):
        return self.mnist_data
    pass
class ElectricalGridStabilityData():
    """
    Instances: 10000
    Attributes: 12
    - tau,参与者反应时间
    - p,供应负载均衡能力
    - g,弹性价格比例系数
    Classes: 2
    - stabf: stable,unstable
    """
    elec_data = pd.read_csv(PATH+"Data_for_UCI_named.csv",header=0)
    elec_data = pd.DataFrame(elec_data)
    elec_data["stabf"].replace("unstable",0,inplace=True)
    elec_data["stabf"].replace("stable",1,inplace=True)
    elec_array = np.array(elec_data)
    def __init__(self):
        self.data = self.elec_array[:,:-1]
        self.target = self.elec_array[:,-1]
        pass
    def load_data(self):
        return self.elec_array
    def load_df(self):
        return self.elec_data
    pass

class IrisData():
    """
    Instances: 150
    Attributes: 4
    - sepal length,萼片长
    - sepal width,萼片宽
    - petal length,花瓣长
    - petal width,花瓣宽
    Classes: 3
    - Iris Setosa,牵牛 -> 1
    - Iris Versicolour,云芝 -> 2
    - Iris Virginica -> 3
    """
    iris_data = pd.read_csv(PATH+"iris.csv",header=0)
    iris_data = pd.DataFrame(iris_data)
    iris_data["class"].replace("Iris-setosa", 1, inplace=True)
    iris_data["class"].replace("Iris-versicolor", 2, inplace=True)
    iris_data["class"].replace("Iris-virginica", 3, inplace=True)
    iris_array = np.array(iris_data)
    def __init__(self):
        self.data = self.iris_array[:,:-1]
        self.target = self.iris_array[:,-1]
        pass
    def load_data(self):
        return self.iris_array
    def load_df(self):
        return self.iris_data
    pass
