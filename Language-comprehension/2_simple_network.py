import numpy as np
from deep_learning.common.gradient import numerical_gradient
from deep_learning.common.functions import softmax,cross_entropy_error

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)
    # 前向传播
    def forward(self,X):
        a = np.dot(X, self.W)
        z = softmax(a)
        return z

    # 损失函数
    def loss(self, x, t):
        y = self.forward(x)
        return cross_entropy_error(y,t)


if __name__ == '__main__':
    x = np.array([0.6,0.9])
    t = np.array([0,0,1])

    net = SimpleNet()
    loss = lambda W: net.loss(x,t)

    dW = numerical_gradient(loss, net.W)
    print(dW)

