import numpy as np
# 阶跃函数
# x传入标量
def step_function0(x):
    if x > 0:
        return 1
    else:
        return 0
# x传入向量
def step_function1(x):
    return np.array(x >= 0, dtype=int)

# sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# relu函数
def relu(x):
    return np.maximum(0, x)

# softmax函数
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

# identify 函数
def identity(x):
    return x

# MSE 回归问题常用损失函数
def mean_squared_error(y_true, y_pred):
    return 0.5 * np.sum((y_true - y_pred) ** 2)

# 交叉熵损失函数
def cross_entropy_error(y_true, y_pred):
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, y_true.size)
        y_pred = y_pred.reshape(1, y_pred.size)
    if y_pred.size == y_true.size:
        y_pred = y_pred.argmax(axis=1)
    n = y_true.shape[0]
    return -np.sum(np.log(y_true[np.arange(n),y_pred] + 1e-7)) / n
# 测试
# x = np.array([0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5])
# print(softmax(x))