import math
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
# from matplotlib import pyplot as plt, rcParams
from matplotlib import pyplot as plt, font_manager, ticker
from numpy import size
from scipy.spatial.distance import cdist
from sklearn import datasets, preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from ONPE import NPE
import numpy as np
from sklearn import metrics
from te import Pic


class DensityPeak:
    """
    密度峰值聚类算法
    """

    def __init__(self,data, dcRatio=0.2, clusterNumRatio=0.05, dcType="max", kernel="gaussian"):
        '''
        构造器，初始化相关参数
        :param distanceMatrix: 数据集的距离矩阵
        :param dcRatio: 半径比率 通常是0.2
        :param dcType: 半径计算类型 包括‘max’,'ave','min' Hausdorff距离等
        :param kernel: 密度计算时选取的计算函数 包括'cutoff-kernel' 'gaussian-kernel'
        font_path = r"C:\\Users\happy cola\.conda\envs\ONPE\Lib\site-packages\matplotlib\mpl-data\fonts\ttf\tnwsimsun.ttf"
        # 字体加载
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        # print(prop.get_name())  # 显示当前使用字体的名称
        # 字体设置
        matplotlib.rcParams['font.family'] = 'sans-serif'  # 使用字体中的无衬线体
        matplotlib.rcParams['font.sans-serif'] = prop.get_name()  # 根据名称设置字体
        matplotlib.rcParams['font.size'] = 12  # 设置字体大小
        matplotlib.rcParams['axes.unicode_minus'] = False  # 使坐标轴刻度标签正常显示正负号
        '''
        self.data = data
        # 实例间距离矩阵
        self.distance_m = cdist(data, data, 'euclidean')
        # 半径比率  这个不是按照比例确定 dc按照优化确定
        self.dcRatio_f = dcRatio
        # 半径类型
        self.dcType = dcType
        # 密度计算核
        self.kernel = kernel
        # 簇中心数量占比
        self.clusterCenterRatio_f = clusterNumRatio
        # 密度向量，存储密度
        self.densities_l = []
        # 存储master
        self.masters_l = []
        # 存储实例到其master的距离
        self.distanceToMaster_l = []
        # 代表性向量，存储实例的代表性
        self.representativeness_l = []
        # 原始的决策权值 p*s
        self.old_representativeness_l = []
        # 簇中心
        self.clusterCenter_l = []
        # 实例数量
        self.numSample = 0
        # 半径dc
        self.dc_f = 0
        # 数据集最大实例间距离
        self.maxDistance = 0
        # 聚类标签
        self.label_l = []
        # 簇块 一个字典 簇号:[簇块]
        self.clusters_d = {}
        self.__initDensityPeak()

    def __initDensityPeak(self):
        '''
        初始化
        :return:
        '''
        # 实例数量
        self.numSample = len(self.distance_m)
        # 最大实例间距离
        self.maxDistance = self.getMaxDistance()
        # 计算半径dc
        sita = [0.01 * i for i in range(1, 520)]
        sita = np.array(sita)
        self.dc_f = self.getDcNew(sita)
        # 计算密度
        self.densities_l = self.computeDensities()
        # 计算实例到master的距离 相对最小距离
        self.computeDistanceToMaster()
        # 计算实例的代表性
        self.computePriority()

    def getDcNew(self, sita):
        # sita为从0到正无穷的序列
        val = []
        for i in range(len(sita)):
            density_list = self.computeFi(sita[i])
            # 将density归一化
            density_list = density_list / density_list.sum()
            val.append(density_list)
        # val为对应fi序列，每列为一个sita对应的
        # 计算数据熵
        val = np.array(val)
        h = []
        test = size(val, 0)
        for i in range(size(val, 0)):
            t = val[i, :] * np.log(val[i, :])
            h.append(-sum(t))
        ind = h.index(min(h))
        '''
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=None, hspace=None)  #调整子图间隔
        left = 0.125  # the left side of the subplots of the figure
        right = 0.9  # the right side of the subplots of the figure
        bottom = 0.1  # the bottom of the subplots of the figure
        top = 0.9  # the top of the subplots of the figure
        wspace = 0.2  # the amount of width reserved for blank space between subplots,
        # expressed as a fraction of the average axis width
        hspace = 0.2  # the amount of height reserved for white space between subplots,
        # expressed as a fraction of the average axis height
        '''
        #fig = plt.figure(figsize=(3, 3))
        fig = plt.figure(figsize=(3,3))
        plt.plot(sita, h)
        #plt.plot(2*sita, h)
        #plt.title(r"数据熵H曲线图")
        #$\sigma_0$
        plt.xlabel("影响因子")
        #$H$
        plt.ylabel("数据熵")
        #plt.title('a.数据熵曲线图',y=-0.35)
        #plt.ylim([7.45, 7.75])
        #plt.xlim([0,5.2])
        plt.grid(lw=0.15,c='lightgray',linestyle='--')
        #(% f'%sita[ind]+', % f)'%h[ind]
        plt.annotate('$\sigma_0$', xy=(sita[ind], h[ind]), xytext=(sita[ind] + 1, h[ind] - 0.01),
                     fontsize=10, arrowprops=dict(arrowstyle='->'))
        plt.axes().spines['top'].set_color("none")
        plt.axes().spines['right'].set_color("none")
        plt.subplots_adjust(left=0.05, bottom=0.05)

        plt.savefig('数据熵.pdf',dpi = 800, bbox_inches='tight')

        #return np.sqrt(3) * sita[ind]
        print(np.sqrt(3) * sita[ind])
        return np.sqrt(3) * sita[ind]

    def computeFi(self, sita):
        '''
        计算给定sita下的密度，按照高斯核进行计算
        :return:1 /
        '''
        density_list = np.sum(1 / (np.exp(np.power(self.distance_m / sita, 2))), axis=1)
        # 将距离为负数的置为0
        density_list = np.maximum(density_list, 1e-12)
        return density_list

    def getDc(self):
        '''
        计算半径dc
        Hausdorff距离可以理解成一个点集中的点到另一个点集的最短距离的最大值。
        :return:
        '''
        resultDc = 0.0
        if self.dcType == "max":
            '''
            计算最大Hausdorff距离
            '''
            resultDc = self.maxDistance
        elif self.dcType == "ave":
            '''
            平均Hausdorff距离
            '''
            resultDc = np.mean(self.distance_m)
        elif self.dcType == "min":
            '''
            最小Hausdorff距离
            '''
            resultDc = np.min(self.distance_m)
        return resultDc * self.dcRatio_f

    def getMaxDistance(self):
        '''
        计算实例间最大距离
        :return:
        '''
        return np.max(self.distance_m)

    def computeDensities(self):
        '''
        计算密度，按照给定的kernel进行计算
        :return:
        '''
        # 按照高斯核计算
        if self.kernel == 'gaussian':
            # 方法一，使用循环
            # temp_local_density_list = []
            # for i in range(0, self.numSample):
            #     temp_local_density_list.append(self.gaussian_kernel(i))
            # 方法二，使用矩阵运算
            temp_local_density_list = np.sum(1 / (np.exp(np.power(self.distance_m / self.dc_f, 2))), axis=1)
            return temp_local_density_list
        # 按照截断核计算
        elif self.kernel == 'cutoff':
            temp_local_density_list = []
            for i in range(0, self.numSample):
                temp_local_density_list.append(self.cutoff_kernel(i))
            return temp_local_density_list

    def gaussian_kernel(self, i):
        '''
        高斯核计算密度
        :param i: 实例标号
        :return: 密度
        '''
        tempDensity = 0
        for j in range(len(self.distance_m[i])):
            tempDistance = self.distance_m[i][j]
            tempDensity += np.exp(-(tempDistance / self.dc_f) ** 2)
        return tempDensity

    def cutoff_kernel(self, i):
        '''
        截断核计算密度
        :param i: 实例标号
        :return: 密度
        '''
        tempDensity = 0
        for j in range(len(self.distance_m[i])):
            tempDistance = self.distance_m[i][j]
            tempDensity += self.F(tempDistance - self.dc_f)
        return tempDensity

    def F(self, x):
        '''
        截断核计算辅助函数
        :param x: 距离差值
        :return:
        '''
        if x < 0:
            return 1
        else:
            return 0

    def computeDistanceToMaster(self):
        '''
        计算实例到master的距离，同时确定实例的master
        :return:
        '''
        # 将密度降序排序，返回索引
        tempSortDensityIndex = np.argsort(self.densities_l)[::-1]
        # 初始化距离向量
        self.distanceToMaster_l = np.zeros(self.numSample)
        # 密度最高的获得最高优先级#密度最高的应该是距离tempSortDensityIndex[0]最远距离 不是inf float('inf')
        self.distanceToMaster_l[tempSortDensityIndex[0]] = max(self.distance_m[tempSortDensityIndex[0], :])
        # 初始化master向量
        self.masters_l = np.zeros(self.numSample, dtype=int)
        # 密度最高的自己是自己的master
        self.masters_l[tempSortDensityIndex[0]] = 0
        # 计算距离和master
        # 选择密度大于自己且距离最近的作为自己的master
        for i in range(1, self.numSample):
            tempIndex = tempSortDensityIndex[i]
            self.masters_l[tempIndex] = tempSortDensityIndex[
                np.argmin(self.distance_m[tempIndex][tempSortDensityIndex[:i]])]
            self.distanceToMaster_l[tempIndex] = np.min(self.distance_m[tempIndex][tempSortDensityIndex[:i]])
        # print(self.masters_l)

    def computePriority(self):
        '''
        计算代表性（优先级）
        :return:
        '''
        # 将局部密度和相对最小距离归一化到0——10
        scaler = MinMaxScaler(feature_range=(0, 1))  # 将数据归一到0到10，可以根据数据特点归一到0到1
        self.densities_l = self.densities_l[:, np.newaxis]
        self.distanceToMaster_l = self.distanceToMaster_l[:, np.newaxis]
        # t = np.array(self.densities_l).reshape(1,-1)
        # 局部密度 相对最小距离
        self.densities_l = scaler.fit_transform(self.densities_l).flatten()  # 归一化
        self.distanceToMaster_l = scaler.fit_transform(self.distanceToMaster_l).flatten()  # 归一化


        #决策图
        fig = plt.figure(figsize=(3, 3))
        #plt.set_title("决策图")
        #$\\rho$"  ,labelpad=-0.1
        plt.xlabel("局部密度")
        #$\sigma$
        plt.ylabel("相对最小距离")
        #plt.title('b.决策图', y=-0.35)
        plt.scatter(self.densities_l, self.distanceToMaster_l,linewidths=0.1)
        # plt.savefig('决策图.svg', dpi=1000, bbox_inches="tight")
        # plt.savefig('', format='png')
        # r=p*sita
        plt.xlim([0, 1.05])
        plt.ylim([0, 1.05])
        plt.axes().spines['top'].set_color("none")
        plt.axes().spines['right'].set_color("none")
        plt.grid(lw=0.15,c='lightgray',linestyle='--')
        plt.subplots_adjust(left=0.05, bottom=0.05)
        plt.savefig('决策图.pdf', dpi=800, bbox_inches='tight')

        self.old_representativeness_l = np.multiply(self.densities_l, self.distanceToMaster_l)
        # r=exp（p）*sita
        self.representativeness_l = np.multiply(np.exp(self.densities_l), self.distanceToMaster_l)

    def getLabel(self, i):
        '''
        获取实例的标签
        :param i: 实例标号
        :return: 实例聚类标签
        '''
        if self.label_l[i] >= 0:
            return self.label_l[i]
        else:
            # 实例没有标签，则使用其master的标签作为自己的标签 聚类中即为聚类簇号
            return self.getLabel(self.masters_l[i])

    def getClusterCenter(self):
        n = int(self.numSample * self.clusterCenterRatio_f)
        return np.argsort(self.representativeness_l)[-n:][::-1]

    def cluster(self):
        # 按照比例计算聚类簇中心个数 进行聚类
        #:param clusterRatio: 簇中心占比
        #:return:
        # n = int(self.numSample * self.clusterCenterRatio_f)
        n = self.findn2()
        self.clusterN(n=n)

    def findn(self):
        # 优化类簇中心的选择 马春来论文中的方法 不可行
        # 取前30个最大的
        max30 = np.argsort(-self.representativeness_l)[0:30]
        plt.figure(3)
        plt.title("前m个最大决策值点降序图")
        plt.xlabel("序号")
        plt.ylabel("决策权值")
        plt.plot(range(len(max30)), self.representativeness_l[max30], marker='*')
        plt.savefig('决策权值图.svg', dpi=300, bbox_inches="tight")
        # plt.savefig('', format='png')
        # plt.show()
        # 保存偏离度值
        k = []
        te = self.representativeness_l[max30]
        for i in range(1, len(max30) - 3):
            # (r_i+1-r_i)(i-1)/(r_i-r1) 注意python中从0开始 所以是*i
            kii1 = self.representativeness_l[max30[i + 1]] - self.representativeness_l[max30[i]]
            k1i = (self.representativeness_l[max30[i]] - self.representativeness_l[max30[0]]) / i
            # k1i = (self.representativeness_l[max30[i]] - self.representativeness_l[max30[i-1]])
            # beita = (kii1-k1i)/k1i
            beita = (kii1) / k1i
            k.append(beita)
        index = k.index(max(k))
        plt.figure(4)
        plt.title("偏离度曲线图")
        plt.xlabel("点序号")
        plt.ylabel("偏离度")
        plt.plot(range(2, 2 + len(k)), k, marker='*')
        plt.savefig('偏离度折线图.svg', dpi=300, bbox_inches="tight")
        # plt.savefig('', format='png')
        # plt.show()
        n = index + 2
        return n

    def findn2(self):
        # 取前30个最大的
        max30 = np.argsort(-self.representativeness_l)[0:30]
        # plt.savefig('', format='png')
        te = self.representativeness_l[max30]  # 决策权值 前m个降序
        #ai=2ri-ri-1-ri+1
        deta = []
        # deta 0——1
        for i in range(1, len(max30)-1):
            deta.append(np.abs(2*te[i] - te[i - 1] - te[i+1]))
        # 求均值作为阈值
        threshold = np.average(deta)
        seq = range(len(deta), 0, -1)
        index = 2
        for i in range(len(deta) - 1, -1, -1):
            if deta[i] >= threshold:
                index = i
                break
            else:
                continue
        # todo
        # 离群点排除
        plt.figure(figsize=(3, 3))
        #plt.title("c.前m个最大决策值点降序图",y=-0.35)
        plt.xlabel("序号")
        plt.ylabel("决策权值$\gamma$")
        plt.xlim([0,31])
        plt.ylim([0,3])
        plt.plot(range(1, len(max30) + 1), self.representativeness_l[max30],
                 label=r"$\gamma=e^\rho\times\delta$", marker='*')
        max30_2 = np.argsort(-self.old_representativeness_l)[0:30]
        plt.plot(range(1, len(max30_2) + 1), self.old_representativeness_l[max30_2],
                 label=r"$\gamma=\rho\times\delta$", marker='.')
        plt.xticks(np.linspace(0, 30, 7))
        plt.axes().spines['top'].set_color("none")
        plt.axes().spines['right'].set_color("none")
        plt.subplots_adjust(left=0.05, bottom=0.05)
        plt.grid(lw=0.15,c='lightgray',linestyle='--')
        plt.legend(loc='best',frameon=False)
        plt.savefig('决策权值图.pdf', dpi=800, bbox_inches='tight')

        plt.figure(figsize=(3, 3))
        #plt.title("d.变化率曲线图",y=-0.35)
        plt.xlabel("序号")
        plt.ylabel("偏离度")
        plt.xlim([0,31])
        plt.ylim([0,1.2])
        plt.plot(range(1, 1 + len(deta)), deta, marker='*',label="样本点")
        plt.hlines(y=threshold, xmin=0, xmax=31, color='red', linestyle='-.')
        plt.annotate('threshold', xy=(15, threshold), xytext=(20, threshold + 0.15),
                     fontsize=9.5, arrowprops=dict(arrowstyle='->'))
        plt.xticks(np.linspace(0, 30, 7))
        plt.axes().spines['top'].set_color("none")
        plt.axes().spines['right'].set_color("none")
        plt.subplots_adjust(left=0.05, bottom=0.05)
        plt.legend(loc='best',frameon=False)
        plt.grid(b=True, axis='both', lw=0.3,linestyle='dotted', color='b')
        plt.savefig('偏离度折线图.pdf', dpi=800, bbox_inches='tight')

        # plt.savefig('', format='png')
        # 选择从0——index作为聚类中心
        n = index + 1
        return n

    def clusterN(self, n=5):
        '''
        按照给定的簇中心个数进行聚类
        :param n: 簇中心个数
        :return:
        '''
        # 初始化标签向量
        self.label_l = np.zeros(self.numSample, dtype=int)
        self.label_l[self.label_l == 0] = -1
        # 初始化聚类中心
        self.clusterCenter_l = np.argsort(self.representativeness_l)[-n:][::-1]
        #todo 根据距离法则排除聚类中心
        self.clusterCenter_l = np.array(self.clusterCenter_l)
        for i in range(len(self.clusterCenter_l)):
            for j in range(i+1,len(self.clusterCenter_l)):
                dis = self.distance_m[self.clusterCenter_l[i],self.clusterCenter_l[j]]
                print(dis)
                if dis < self.dc_f:
                    self.label_l[self.clusterCenter_l[j]] = i + 1
                    self.clusterCenter_l = np.delete(self.clusterCenter_l,self.clusterCenter_l[j])
        '''
        
        '''
        # 初始化簇号 使用簇号作为聚类标签
        for i in range(n):
            if self.label_l[self.clusterCenter_l[i]] != 0:
                self.label_l[self.clusterCenter_l[i]] = i + 1
                # self.label_l[self.clusterCenter_l[i]] = i + 1
        # 统计聚类标签
        for i in range(self.numSample):
            if self.label_l[i] >= 0:
                continue
            self.label_l[i] = self.getLabel(self.masters_l[i])

        # 初始化聚类簇块
        self.clusters_d = {key: [] for key in self.label_l[self.clusterCenter_l]}

        # 按照聚类结果划分簇块
        for i in self.label_l[self.clusterCenter_l]:
            self.clusters_d[i] += [j for j in range(self.numSample) if self.label_l[j] == i]

    @staticmethod
    def getDistanceByEuclid(instance1, instance2):
        '''
        按照欧氏距离计算实例间距离
        :param instance1: 实例1
        :param instance2: 实例2
        :return: 欧氏距离
        '''
        dist = 0
        for key in range(len(instance1)):
            dist += (float(instance1[key]) - float(instance2[key])) ** 2
        return dist ** 0.5

if __name__ == '__main__':
    font_path = r"C:\\Users\happy cola\.conda\envs\ONPE\Lib\site-packages\matplotlib\mpl-data\fonts\ttf\tnwsimsun.ttf"
    # 字体加载
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    print(prop.get_name())  # 显示当前使用字体的名称
    # 字体设置
    matplotlib.rcParams['font.family'] = 'sans-serif'  # 使用字体中的无衬线体
    matplotlib.rcParams['font.sans-serif'] = prop.get_name()  # 根据名称设置字体
    matplotlib.rcParams['font.size'] = 10.5  # 设置字体大小
    matplotlib.rcParams['axes.unicode_minus'] = False  # 使坐标轴刻度标签正常显示正负号
    #matplotlib.rc('text', usetex=True)
    '''
    
    # 导入数据原始数据集 而非过采样之后的
    df = pd.read_csv("shiyu_2.csv")
    data_y = df["label"][0:2000]
    data_x = df.values[0:2000, 1:-1]  # 去掉开头的id和结尾的label
    
    '''

    df = pd.read_csv("data_test_2.csv")
    data_y = df["label"]
    data_x = df.values[:, 1:-1]  # 去掉开头的id和结尾的label
    #df = pd.read_csv("data_total.csv")
    #去重
    #df = df.drop_duplicates(keep="first")
    #data_x, data_y, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=20, shuffle=True)
    # 归一化数据
    '''
    # use function preprocessing.scale to standardize X
    data = preprocessing.scale(data_x)  # 调用sklearn包的方法
    '''
    scaler = MinMaxScaler(feature_range=(0, 1))  # 将数据归一到0到1，可以根据数据特点归一到-1到1
    data_x = scaler.fit_transform(data_x).tolist()  # 归一化


    # 降维NPE类  对数据进行降维
    #npe = NPE(6)
    npe = NPE(dim=5)
    # 降维后的数据
    data = npe.dimension(data_x, 0)
    #x_train, x_test, y_train, y_test = train_test_split(data, data_y, test_size=0.3, random_state=20, shuffle=True)
    # 画图
    # 保存归一化器
    # scalerfile = 'scaler.sav'
    # pickle.dump(scaler, open(scalerfile, 'wb'))
    # 聚类
    # 进行聚类
    dp = DensityPeak(data, 0.2, 0.05, "max")
    dp.cluster()
    for key, value in dp.clusters_d.items():
        print("簇号=", key, ",cluster= ", value)
        print("簇长度=", len(value))
        #if key == 6:
            #df = df.drop(value)
    #df.to_csv('data_test.csv')
    # fig = plt.figure(5)
    # 4.聚类结果可视化
    # 进行数据降维处理
    # 显示图像 t-sne 降维至二维
    label_pred = dp.label_l  # 获取聚类标签
    print("V-measure: %0.3f" % metrics.v_measure_score(data_y, label_pred))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(data_y, label_pred))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(data_y, label_pred,
                                               average_method='arithmetic'))
    print("FMI: %0.3f" % metrics.fowlkes_mallows_score(data_y, label_pred))
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=20, n_iter=1000)
    x_tsne = tsne.fit_transform(data)
    df1 = pd.DataFrame(x_tsne, columns=['x', 'y'])
    # 转换完后会有nan 很奇怪
    #lab1 = pd.DataFrame(data_y, columns=['label'])
    lab1 = pd.DataFrame(data_y.values, columns=['label'])
    result1 = pd.concat([df1, lab1], axis=1)
    #
    fig = plt.figure(figsize=(3, 3))
    #,
    sns.scatterplot(data=result1, x="x", y="y", palette=sns.color_palette("hls", len(np.unique(data_y))), hue='label',linewidth=0)
    plt.title("a.原始类簇分布图", y=-0.28)
    # plt.xlabel("xxxxxx",labelpad=1)
    # plt.ylabel("xxxxxx", labelpad=1)
    # plt.tick_params(pad=100)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # plt.subplots_adjust( left=0.8,bottom=0.1)
    plt.savefig('原始类簇分布图.pdf', dpi=800, bbox_inches='tight')


    lab2 = pd.DataFrame(label_pred, columns=['label'])
    result2 = pd.concat([df1, lab2], axis=1)
    # , palette = sns.color_palette("hls", 5),style='label'
    fig = plt.figure(figsize=(3, 3))
    sns.scatterplot(data=result2, x="x", y="y", palette=sns.color_palette("hls", len(dp.clusterCenter_l)), hue='label',linewidth=0)
    plt.title("b.ICFSFDP算法聚类结果图", y=-0.28)
    # plt.xlabel("xxxxxx",labelpad=1)
    # plt.ylabel("xxxxxx", labelpad=1)
    # plt.tick_params(pad=100)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # plt.subplots_adjust( left=0.8,bottom=0.1)
    plt.savefig('ICFSFDP算法聚类结果图.pdf', dpi=800, bbox_inches='tight')
    # 绘图
    plt.show()

