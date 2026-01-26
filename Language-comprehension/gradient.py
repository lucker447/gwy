import numpy as np
# 数值微分
def numerical_diff0(f, x):
    h = 1e-4
    return (f(x + h) - f(x) )/h
# 中心点数值微分
def numerical_diff1(f, x):
    h = 1e-4
    return (-f(x - h) + f(x + h)) / (2 * h)

# 梯度(多元函数)
def gradient(f, x):
    # 初始化梯度向量，与自变量同形状
    h = 1e-4
    grad = np.zeros_like(x, dtype=np.float64)
    # 遍历每个自变量，计算对应的偏导数
    for i in range(len(x)):
        # 保存原始值，避免修改输入
        x_i = x[i]
        # 计算x[i]+h处的函数值（其他变量不变）
        x[i] = x_i + h
        f_plus = f(x)
        # 计算x[i]-h处的函数值（其他变量不变）
        x[i] = x_i - h
        f_minus = f(x)
        # 中心差分公式计算偏导数：∂f/∂x_i ≈ [f(x_i+h) - f(x_i-h)] / (2h)
        grad[i] = (f_plus - f_minus) / (2 * h)
        # 恢复原始值
        x[i] = x_i
    return grad
#
def numerical_gradient(f, X):
    if X.ndim == 1:
        return gradient(f, X)
    else:
        grad = np.zeros_like(X, dtype=np.float64)
        for i in range(len(X)):
            grad[i] = gradient(f, X[i])
        return grad