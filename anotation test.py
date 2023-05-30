import random

import numpy as np
import pandas as pd
from pyod.models.cof import COF
from pyod.models.ecod import ECOD
from pyod.models.gmm import GMM
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
# 划分训练集、测试集
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.utils import evaluate_print
from pyod.utils.example import visualize
from sklearn.model_selection import train_test_split
from pyod.models.knn import KNN
from sklearn import metrics
df = pd.read_csv("data_annotation.csv")
y = df["label"]
x = df.values[:,1:-1]  #去掉开头的id和结尾的label
scaler1 = MinMaxScaler(feature_range=(0, 1))  # 将数据归一到0到1，可以根据数据特点归一到-1到1
x = scaler1.fit_transform(x).tolist()  # 归一化
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20, shuffle=True)
df = pd.read_csv("data_annotation_dec.csv")
y_dec = df["label"]
x_dec = df.values[:,1:-1]  #去掉开头的id和结尾的label
scaler1 = MinMaxScaler(feature_range=(0, 1))  # 将数据归一到0到1，可以根据数据特点归一到-1到1
x_dec = scaler1.fit_transform(x_dec).tolist()  # 归一化
x_dec = np.array(x_dec)
#仅仅选取25条
seed=100
random.seed(seed)
random.shuffle(x_dec)
random.seed(seed)#一定得重复在写一遍,和上面的seed要相同,不然y_batch和x_batch打乱顺序会不一样,其实无所谓，因为label都是1
random.shuffle(y_dec)
x_dec = x_dec[:50,:]
y_dec = y_dec[:50]
x_test = np.vstack((x_test,x_dec))
y_test = np.hstack((y_test,y_dec))

clf = LOF(contamination=0.05,n_neighbors=5)
clf = GMM(contamination=0.05)
clf = KNN(contamination=0.05,radius=0.5,n_neighbors=5)
clf = IForest(contamination=0.05,max_features=1)

clf.fit(x_train)

# 返回训练数据X_train上的异常标签和异常分值
y_train_pred = clf.labels_  # 返回训练数据上的分类标签 (0: 正常值, 1: 异常值)
y_train_scores = clf.decision_scores_  # 返回训练数据上的异常值 (分值越大越异常)
print(y_train_pred)
# 用训练好的clf来预测未知数据中的异常值
y_test_pred = clf.predict(x_test)  # 返回未知数据上的分类标签 (0: 正常值, 1: 异常值)
print(np.sum(y_test_pred==1))#异常值数量

y_test_scores = clf.decision_function(x_test)  #  返回未知数据上的异常值 (分值越大越异常)
print(y_test_pred==y_test)
print("acc:%0.4f"%metrics.accuracy_score(y_test, y_test_pred))
print("rec:%0.4f"%metrics.recall_score(y_test, y_test_pred))
print("pre:%0.4f"%metrics.precision_score(y_test, y_test_pred))
print("f1:%0.4f"%metrics.f1_score(y_test, y_test_pred))
print("rocauc:%0.3f"%metrics.roc_auc_score(y_test,y_test_scores))
print(metrics.confusion_matrix(y_test, y_test_pred))
evaluate_print(clf, y_test, y_test_scores)
tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=40, n_iter=1000)
x_tsne_train = tsne.fit_transform(x_train)
x_tsne_test = tsne.fit_transform(x_test)
visualize(clf, x_tsne_train, y_train, x_tsne_test, y_test, y_train_pred,
    y_test_pred, show_figure=True, save_figure=False)