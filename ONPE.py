import pandas as pd
class NPE:
    def __init__(self,dim=10,k=3):
        import shogun as sg
        #模型
        self.converter = sg.NeighborhoodPreservingEmbedding()
        # set target dimensionality
        self.converter.set_target_dim(dim)
        # set number of neighbors
        self.converter.set_k(k)
        # set number of threads
        self.converter.parallel.set_num_threads(2)
        # set nullspace shift (optional)
        self.converter.set_nullspace_shift(-1e-6)

    #data为输入数据，格式为n*p  n为样本数量，p为维度数量
    def dimension(self,data,flag=1):
        import shogun as sg
        from sklearn.decomposition import PCA
        #PCA预处理
        if flag == 0:
            ans = data
            return ans
        data = PCA(n_components='mle').fit_transform(data)
        #转置
        feature_matrix = data.T
        features = sg.RealFeatures(feature_matrix)
        embedding = self.converter.embed(features)
        features_dimension = embedding.get_feature_matrix()
        ans = features_dimension.T
        #输出为n*k  n为样本数量 k为降维后的维度
        return ans

if __name__ == '__main__':
    npe = NPE()
    # load data
    df = pd.read_csv("data_total.csv")
    y = df["label"]
    x = df.values[:,1:-1]  #去掉开头的id和结尾的label
    x_dim = npe.dimension(x)
    print(3)