import joblib
import matplotlib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from detection import Detection
from imblearn.over_sampling import KMeansSMOTE
from sklearn.cluster import MiniBatchKMeans

class CAT:
    def __init__(self):
        #预训练模型
        #最大最小归一化可能后续实时诊断会有问题
        #self.scaler1 = MinMaxScaler(feature_range=(0, 1))  # 将数据归一到0到1，可以根据数据特点归一到-1到1
        self.scaler1 = StandardScaler()  # 将数据归一到0到1，可以根据数据特点归一到-1到1
        self.fitByFile()
        self.model = joblib.load('cat.dat')

    def resample(self,file):
        df = pd.read_csv(file)
        y_all = df.values[:, -1]
        x_all = df.values[:, 1:-1]  # 去掉开头的id和结尾的label
        x_all = self.scaler1.fit_transform(x_all).tolist()  # 归一化
        x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=20, shuffle=True)
        '''
        #过采样 实验时用到，加上之后时间会剧增，暂时先注释掉了
        # 多数类样本数量
        data_train = np.array(x_train)
        marjority_number = len(data_train[y_train == 1])
        sampling_strategy = {}
        for i in np.unique(y_train):
            sampling_strategy.update({i: marjority_number})
        sm = KMeansSMOTE(
            sampling_strategy=sampling_strategy,
            # cluster_balance_threshold=1,
            kmeans_estimator=MiniBatchKMeans(batch_size=512, random_state=0), random_state=42
        )
        x_train, y_train = sm.fit_resample(x_train, y_train)
        '''
        return x_train, y_train, x_test, y_test

    def fitByFile(self,filename='data_test_2.csv'):
        # 读取data.csv仅仅为了之后特征重要性图的列名
        #df = pd.read_csv(filename)
        #过采样数据 输入的数据已经经过归一化降维了
        x_train, y_train, x_test, y_test = self.resample(filename)
        #训练异常检测模型
        self.detection = Detection()
        self.detection.fit(x_train,y_train)
        '''
        model = CatBoostClassifier(iterations=500,
                                   # task_type="GPU",
                                   # custom_loss='F1',
                                   # eval_metric='F1',
                                   depth=6,
                                   l2_leaf_reg=3,
                                   cat_features=categorical_features_indices,
                                   learning_rate=0.04,
                                   loss_function='MultiClass',
                                   logging_level='Verbose')
        model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], plot=True)
        
        #保存训练好的模型
        import joblib
        joblib.dump(model,'cat.dat')
        '''
        #训练CatBoost
        train_dataset = Pool(data=x_train, label=y_train)
        eval_dataset = Pool(data=x_test, label=y_test)
        categorical_features_indices = None
        model = CatBoostClassifier(iterations=100,
                                   task_type="GPU",
                                   # custom_loss='F1',
                                   # eval_metric='F1',
                                   depth=10,
                                   l2_leaf_reg=3,
                                   cat_features=categorical_features_indices,
                                   learning_rate=0.02,
                                   loss_function='MultiClass',
                                   logging_level='Verbose')

        grid = {'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25],
                'depth': [4, 6, 8, 10],
                'l2_leaf_reg': [1, 3, 5, 7, 9]}
        randomized_search_result = model.randomized_search(grid,
                                                           X=x_train,
                                                           y=y_train,
                                                           plot=True)
        model.fit(train_dataset)
        self.model = model
        # 保存训练好的模型
        import joblib
        joblib.dump(self.model, 'cat.dat')

    def fitByData(self,data):
        # 输入需要是一个np.array
        return

    #data为预测样本 array
    #ans -1 表示未知类别样本 否则为样本类别
    def predict(self,data):
        #转为二维矩阵  19为特征维度
        data = data.reshape(1, 19)
        # 导入归一化模型
        data = self.scaler1.transform(data)
        #异常检测算法检测
        flag_dec = self.detection.predict(data)
        #flag_dec == 1
        if  flag_dec == -1:
            print("该样本为未知类别样本，请人工标注")
            ans = -1
        else:
            ans = self.model.predict(data)
        return ans

if __name__ == '__main__':
    '''
    font_path = r"C:\\Users\happy cola\.conda\envs\ONPE\Lib\site-packages\matplotlib\mpl-data\fonts\ttf\tnwsimsun.ttf"
    # 字体加载
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    print(prop.get_name())  # 显示当前使用字体的名称
    # 字体设置
    matplotlib.rcParams['font.sans-serif'] = prop.get_name()  # 根据名称设置字体
    matplotlib.rcParams['font.family'] = 'sans-serif'  # 使用字体中的无衬线体
    matplotlib.rcParams['font.size'] = 12  # 设置字体大小
    matplotlib.rcParams['axes.unicode_minus'] = False  # 使坐标轴刻度标签正常显示正负号
    '''
    config = {
        "font.family": 'serif',
        "font.size": 12,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
    }
    cat = CAT()
    matplotlib.rcParams.update(config)
    '''
    data = np.array(
        [32.7, 13.4, 104.7, 37.9, 34.8, 14.5, 41.5, 47.8, 1.19, 6597, 3.7, 3.1, 39.1, 15.6, 26.9, 49.5, 60.9, 6.2,
          41.5])
    test = np.array([32.7,13.4,104.7,37.9,34.8,14.5,41.5,47.8,1.19,6597,3.7,3.1,39.1,15.6,26.9,49.5,60.9,6.2,41.5])  
    test = test.reshape(1,19)
    #导入归一化模型
    import pickle
    scalerfile = 'test_scaler.sav'
    scaler = pickle.load(open(scalerfile, 'rb'))
    test = scaler.transform(test)
    '''
    #test = np.array([0.6390977443609025,0.9333333333333336,0.3125,0.39755351681957185,0.8260869565217386,0.7987421383647799,0.7499999999999996,0.6845238095238093,0.4500000000000002,0.670917225950783,0.6499999999999999,0.6000000000000001,0.7475247524752475,0.10891089108910879,0.96551724137931,0.1896551724137936,1.0,0.9000000000000004,0.7480314960629921])
    df = pd.read_csv("data_test.csv")
    y_all = df.values[:, -1]
    x_all = df.values[:, 1:-1]  # 去掉开头的id和结尾的label
    label = []
    for i in range(len(x_all)):
        flag = cat.predict(x_all[i,:])
        label.append(int(flag))
    all =np.array([label,list(y_all)])
    #plt.show()