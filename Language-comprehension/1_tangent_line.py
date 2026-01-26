from deep_learning.common.gradient import numerical_diff1
import matplotlib.pyplot as plt
import numpy as np

# 目标函数
def f(x):
    return 0.01 * x ** 2 + 0.1 * x

# 切线方程
def tangent_function(f, x):
    a = numerical_diff1(f, x)
    b = f(x) - a * x
    return lambda x: a * x + b

if __name__ == '__main__':
    x = np.arange(0.0, 20.0, 0.1)
    y = f(x)

    tf = tangent_function(f, 10)
    y2 = tf(x)

    plt.plot(x, y, label='f(x)')
    plt.plot(x, y2, label='f2(x)')
    plt.show()