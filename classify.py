import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from imblearn.over_sampling import KMeansSMOTE, SVMSMOTE
from matplotlib import rcParams
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ONPE import NPE
from smote import so

'''
#读取data.csv仅仅为了之后特征重要性图的列名
filename = 'data_test_2.csv'
df=pd.read_csv(filename)
x_train,y_train,x_test,y_test = so(filename)
'''

'''
df = pd.read_csv("train_30l.csv")
y_train = df.values[:,-1]
x_train = df.values[:,0:-1]  #去掉开头的id和结尾的label

df = pd.read_csv("test.csv")
y_test = df.values[:,-1]
x_test = df.values[:,0:-1]  #去掉开头的id和结尾的label
'''


df = pd.read_csv("data_train_final.csv")
y_train = df.values[:,-1]
x_train = df.values[:,1:-1]  #去掉开头的id和结尾的label
df = pd.read_csv("data_test_final.csv")
y_test = df.values[:,-1]
x_test = df.values[:,1:-1]  #去掉开头的id和结尾的label

scaler1 = MinMaxScaler(feature_range=(0, 1))  # 将数据归一到0到1，可以根据数据特点归一到-1到1
x_train = scaler1.fit_transform(x_train).tolist()  # 归一化

# 多数类样本数量
data_train = np.array(x_train)
marjority_number = len(data_train[y_train == 1])
sampling_strategy = {}
for i in np.unique(y_train):
    sampling_strategy.update({i: marjority_number})

'''
import smote_variants as sv
oversampler = sv.MulticlassOversampling(oversampler='DBSMOTE',
                                            oversampler_params={'eps':0.6,'random_state': 6,'min_samples':5})
# X_samp and y_samp contain the oversampled dataset
x_train, y_train = oversampler.sample(np.array(x_train), np.array(y_train))
'''
sm = KMeansSMOTE(
    sampling_strategy=sampling_strategy,
    # cluster_balance_threshold=1,
    kmeans_estimator=MiniBatchKMeans(batch_size=512, random_state=0), random_state=42
)
x_train, y_train = sm.fit_resample(x_train, y_train)
'''
sm = SVMSMOTE(sampling_strategy=sampling_strategy,random_state=42)
x_train, y_train = sm.fit_resample(x_train, y_train)
from smote_variants import Borderline_SMOTE1
from smote_variants import Borderline_SMOTE2
sm = Borderline_SMOTE2(sampling_strategy=sampling_strategy,random_state=42)
x_train, y_train = sm.fit_resample(np.array(x_train), y_train)
'''
scaler2 = MinMaxScaler(feature_range=(0, 1))  # 将数据归一到0到1，可以根据数据特点归一到-1到1
x_test = scaler2.fit_transform(x_test).tolist()  # 归一化


# 划分训练集、测试集
'''
filename = 'data_test_2.csv'
df=pd.read_csv(filename)
y = df.values[:,-1]
x = df.values[:,1:-1]  #去掉开头的id和结尾的label
scaler = MinMaxScaler(feature_range=(0, 1))  # 将数据归一到0到1，可以根据数据特点归一到-1到1
x = scaler.fit_transform(x).tolist()  # 归一化
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20, shuffle=True)

'''

'''
# 归一化数据

scaler2 = MinMaxScaler(feature_range=(0, 1))  # 将数据归一到0到1，可以根据数据特点归一到-1到1
x_test = scaler2.fit_transform(x_test).tolist()  # 归一化
# 降维NPE类
npe1 = NPE()
# 降维后的数据
x_train = npe1.dimension(x_train, 0)
npe2 = NPE()
x_test = npe2.dimension(x_test,0)
'''


train_dataset = Pool(data=x_train,label=y_train)
eval_dataset = Pool(data=x_test,label=y_test)
#np.where(x_train.dtypes != np.float)[0]
categorical_features_indices = None
model = CatBoostClassifier(iterations=200,
                            task_type="GPU",
                            #custom_loss='F1',
                            #eval_metric='F1',
                           depth=10,
                           l2_leaf_reg = 3,
                           cat_features=categorical_features_indices,
                           learning_rate=0.02,
                           loss_function='MultiClass',
                           logging_level='Verbose')

grid = {'learning_rate': [0.05, 0.1,0.15,0.2,0.25],
        'depth': [4,6,8, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9]}
randomized_search_result = model.randomized_search(grid,
                                                   X=x_train,
                                                   y=y_train,
                                                   plot=True)
model.fit(train_dataset)
#保存训练好的模型
import joblib
joblib.dump(model,'cat.dat')
'''
#导入模型
load_model = joblib.load('cat.dat')
load_model.predict(test)
#预测
# Get predicted classes
preds_class = model.predict(eval_dataset)
# Get predicted probabilities for each class
preds_proba = model.predict_proba(eval_dataset)
# Get predicted RawFormulaVal
preds_raw = model.predict(eval_dataset, 
                          prediction_type='RawFormulaVal')

#predict_proba(x_test) 返回所属类别概率
'''
y_pred = model.predict(x_test)

# 模型评价
y_pred_train = model.predict(x_train)
y_pred_valid = model.predict(x_test)
from sklearn import metrics
print("ACC: %0.4f"
      % metrics.accuracy_score(y_test,y_pred_valid))
print("f1_score: %0.4f"
      % metrics.f1_score(y_test,y_pred_valid,average='weighted'))
print("kappa: %0.4f"%metrics.cohen_kappa_score(y_test,y_pred_valid))
print(metrics.confusion_matrix(y_test,y_pred_valid))

y_true = np.array(y_test)
d = np.c_[y_pred, y_true]


config = {
    "font.family": 'serif',
    "font.size": 12,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)
# 特征重要度排序

data = pd.DataFrame({'feature_importance': model.feature_importances_,
              'feature_names': model.feature_names_}).sort_values(by=['feature_importance'],                                                   ascending=True)
fea_ = data["feature_importance"]
fea_name = data["feature_names"]
plt.figure(figsize=(10, 10))
p = fea_name.tolist()
p = list(map(int, p))
names = df.columns.tolist()
names.pop(0)
plt.barh(fea_name, fea_, height=0.5)
for a,b in zip( fea_,fea_name): # 添加数字标签
   print(a,b)
   plt.text(a+0.001, b,'%.3f'%float(a)) # a+0.001代表标签位置在柱形图上方0.001处
plt.yticks(p, names)
plt.xlabel('feature importance') # x 轴
plt.ylabel('features') # y轴
plt.title('Feature Importances') # 标题
#data.plot.barh(x='feature_names',y='feature_importance')

# 混淆矩阵
def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)
    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.0f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()

#混淆矩阵的类别
cata = ['I', 'II', 'III', 'VI','V','Ⅵ','Ⅶ','Ⅷ','Ⅸ','Ⅹ','ⅩI', 'ⅩII', 'ⅩIII', 'ⅩVI','ⅩV','ⅩⅥ','ⅩⅦ','ⅩⅧ','ⅩⅨ','ⅩⅩ']
classes = cata[:len(np.unique(y_train))]

# 获取混淆矩阵
cm = confusion_matrix(y_true, y_pred)
# Normalize by row
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized)

plot_confusion_matrix(cm, 'confusion_matrix.png', title='confusion matrix')


'''
df0 = pd.read_csv("data_total.csv")
x0 = df0.values[:,1:-1]  #去掉开头的id和结尾的label
y0 = df0["label"]
#归一化
# 导入归一化模型
import pickle
scalerfile = 'scaler.sav'
scaler = pickle.load(open(scalerfile, 'rb'))
data_x0 = scaler.transform(x0)

#降维NPE类
npe1 = NPE()
#降维后的数据
x0 = npe1.dimension(data_x0,0)

y0_pre = model.predict(x0)

# 获取混淆矩阵
cm = confusion_matrix(y0, y0_pre)
# Normalize by row
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized)
'''






