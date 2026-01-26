import numpy as np
import pandas as pd
import joblib
from jieba.lac_small.predict import batch_size
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.nn import Sequential
from torch.utils.hipify.hipify_python import preprocessor

from deep_learning.common.functions import sigmoid,softmax

# 读取数据
def get_data():
    # 1.读取数据集
    data = pd.read_csv('../data/train.csv')
    x = data.drop(axis=1,columns=['label'])
    y = data['label']
    x_train,x_test,y_train,y_test = train_test_split(x,y,
                                    test_size=0.2,random_state=42)
    # 3.归一化
    preprocessor = MinMaxScaler()
    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)

    return x_test,y_test

# 加载模型
def init_network():
    network = joblib.load('../models/mnist_classifier.pkl') # 找不到
    return network

# 前向转播
def forward(network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)
    return y

# 主流程
# 1. 读取数据
x, y = get_data()
# 2. 加载模型
network = init_network()
# 3. 分批推理
n = x.shape[0]
batch_size = 100
accurate_cunt = 0
for i in range(0,n, batch_size):
    x_batch = x[i:i+batch_size]
    y_probs = forward(network,x)
    y_pred = np.argmax(y_probs, axis=1)
    # 4. 计算准确率
    accurate_cunt += np.sum(np.equal(y_pred,y[i:i+batch_size]))
print("准确率： ", accurate_cunt / n)
