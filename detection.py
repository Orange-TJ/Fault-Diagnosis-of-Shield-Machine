import itertools
import random

import numpy as np
import pandas as pd
from numpy import empty
from pyod.utils import evaluate_print
from pyod.utils.example import visualize
from scipy.spatial.distance import pdist, cdist
from sklearn import metrics, preprocessing
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ONPE import NPE
from cfsfdp import DensityPeak

class Detection:
    """
    异常检测算法
    """
    def __init__(self):
        '''
        构造器，初始化相关参数
        '''
        self.points_set = []

    def __initDensityPeak(self):
        '''
        初始化
        :return:
        '''
    #X需要是np.array
    def fit(self, X,y):
        #训练模型 计算特征点集合
        #聚类ICFSFDP
        #计算距离矩阵？
        X = np.array(X)  #防止输入的不是np.array
        #distanceMartix = cdist(X,X, 'euclidean')
        self.dp = DensityPeak(X, 0.2, 0.05, "max")
        self.dp.cluster()
        cluster_center = self.dp.clusterCenter_l
        #类簇中心决策权值
        representativeness_l = self.dp.representativeness_l[cluster_center]
        zip_center_represent = zip(cluster_center,representativeness_l)
        #降序
        sorted_zip = sorted(zip_center_represent,key=lambda x:x[1],reverse=True)
        sorted_cluster, sorted_represent = zip(*sorted_zip)
        for index,cluster in enumerate(sorted_cluster):
            #一个类簇的样本点？
            samples = X[self.dp.label_l==self.dp.label_l[cluster]]
            #按照局部密度升序排序
            densities = self.dp.densities_l[self.dp.label_l==self.dp.label_l[cluster]]
            zip_samples_densities = zip(samples, densities)
            # 升序
            zip_sd = sorted(zip_samples_densities, key=lambda x: x[1])
            sorted_samples, sorted_densities = zip(*zip_sd)
            #sorted_samples = list(itertools.chain(*sorted_samples))
            data = np.array([list(item) for item in sorted_samples])
            while data.shape[0] != 0:   #data中还有点
                #第0行——
                d = np.array(data[0,:])
                data = np.delete(data,0,axis=0)
                self.points_set.append(d)
                #从sorted_samples中移除距离d小于dc的元素
                a = d.shape[0]
                d = d.reshape(1,d.shape[0])
                dis = cdist(d, data, 'euclidean').flatten()  #1*p
                #q = np.where(dis <= self.dp.dc_f, dis)
                mask = dis<=self.dp.dc_f
                #距离小于的下标 需要删除
                ind = []
                for j,item in enumerate(mask):
                    if item:
                        ind.append(j)
                if ind:
                    data = np.delete(data, ind, axis=0)
        self.points_set = np.array(self.points_set)
        #return self.points_set

        #data需要是1*n的np.array

    def predict(self,data):
        '''
        :param data:
        :return:
        1表示非异常点
        -1表示异常点
        '''
        if self.points_set is empty:
            Warning("需要先训练模型！")
            return
        #判别因子 默认为1  1.2的时候有一个样本误检 1.25
        factor = 1.25
        dis = cdist(data, self.points_set, 'euclidean') # 1*p
        #小于dc 的为1 大于的0 求和大于0表示至少有一个距离小于dc ——表示非异常点
        total = np.sum(np.where(dis<factor*self.dp.dc_f,1,0),axis=1)
        total = total.reshape(total.shape[0],1)
        ans = np.ones([total.shape[0],total.shape[1]])
        ans[np.where(total<=0)]=-1
        return ans
    def predictL(self,data):
        '''
        为了和论文一致
        1表示异常
        0表示非异常点
        '''
        if self.points_set is empty:
            Warning("需要先训练模型！")
            return
        dis = cdist(data, self.points_set, 'euclidean') # 1*p
        #小于dc 的为1 大于的0 求和大于0表示至少有一个距离小于dc ——表示非异常点
        total = np.sum(np.where(dis<self.dp.dc_f,1,0),axis=1)
        total = total.reshape(total.shape[0],1)
        ans = np.zeros([total.shape[0],total.shape[1]])
        ans[np.where(total<=0)]=1
        return ans

if __name__ == '__main__':
    '''
    df = pd.read_csv("resample.csv")
    data_y = df["label"]
    data_x = df.values[:, 1:-1]  # 去掉开头的id和结尾的label
    
    df = pd.read_csv("data_annotation.csv")
    y = df["label"]
    x = df.values[:, 1:-1]  # 去掉开头的id和结尾的label
    '''
    #shiyu_2
    df = pd.read_csv("shiyu_2.csv")
    y = df["label"]
    x = df.values[1:10000, 1:-1]  # 去掉开头的id和结尾的label
    x = np.array(x)
    x_train = x[0:8000]
    x_test = x[8000:10000]
    y_train = y[0:8000]
    y_test = y[8000:9999]
    scaler1 = StandardScaler()  # 将数据归一到0到1，可以根据数据特点归一到-1到1
    x_train = scaler1.fit_transform(x_train).tolist()  # 归一化
    x_test = scaler1.transform(x_test).tolist()
    '''
    scaler1 = MinMaxScaler(feature_range=(0, 1))  # 将数据归一到0到1，可以根据数据特点归一到-1到1
    x = scaler1.fit_transform(x).tolist()  # 归一化
    x_dec = np.array(x_dec)
    # 仅仅选取25条
    seed = 100
    random.seed(seed)
    random.shuffle(x_dec)
    random.seed(seed)  # 一定得重复在写一遍,和上面的seed要相同,不然y_batch和x_batch打乱顺序会不一样,其实无所谓，因为label都是1
    random.shuffle(y_dec)
    x_dec = x_dec[:50, :]
    y_dec = y_dec[:50]
    x_test = np.vstack((x_test, x_dec))
    y_test = np.hstack((y_test, y_dec))
    '''
    dec = Detection()
    dec.fit(x_train,y_train)
    y_test_pred = dec.predictL(x_test)

    print("acc:%0.4f"%metrics.accuracy_score(y_test, y_test_pred))
    print("rec:%0.4f"%metrics.recall_score(y_test, y_test_pred))
    print("pre:%0.4f"%metrics.precision_score(y_test, y_test_pred))
    print("f1:%0.4f"%metrics.f1_score(y_test, y_test_pred))
    #print("rocauc:%0.3f"%metrics.roc_auc_score(y_test,y_test_scores))
    print(metrics.confusion_matrix(y_test, y_test_pred))


