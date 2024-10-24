"""
@FileName：linear_classifier.py
@Description：线性分类器，利用数据来训练(train)出W,b，从而实现预测(predict)
@Author：Hang Yin
@Time：2024/8/14 16:15
"""
import numpy as np
from classifiers.linear_svm import *
from classifiers.linear_softmax import *

"""
创建线性分类器
1.训练参数矩阵W（10×3073），包含权重与偏差
2.进行预测，使用线性计算公式s = Wx得到最终预测得分最大值
"""


class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        """
        编写线性分类的训练器，基本思想就是利用沿梯度下降的方式（随机梯度下降法SGD）来找到使得损失最小的W。存进分类器对象的字段中，
        @param X:用于训练的图像集(N×3073)
        @param y:用于训练的标签集（N,）
        @param learning_rate:学习率，也是步长，用于梯度下降，最优化
        @param reg:正则惩罚系数
        @param num_iters:进行最优化时下降的次数,迭代次数
        @param batch_size:在每一步下降使用训练示例的个数
        @param verbose:如果为True，打印最优化的过程
        @return:一个包含每次训练迭代计算的损失的列表
        """
        # 取得训练集大小与图像维度
        num_train, dim = X.shape
        # 取得训练类别的个数
        num_classes = np.max(y)+1
        # 记录损失每次下降计算的损失
        loss_history = []
        # 如果没有缓存的W，随机初始化W(C×Dim)，我写的不是按照参考写的
        if self.W is None:
            self.W = np.random.randn(num_classes, dim)
        # 开始进行迭代优化找到损失最小W，次数为num_iter
        for it in range(num_iters):
            # 每次迭代选出 batch_size 个训练样本投入到 SVM 中，然后再计算一次 Loss 函数进行梯度下降，避免计算太频繁导致时间消耗过大
            X_batch, y_batch = None, None
            bindex = np.random.choice(num_train, batch_size, replace=False)
            X_batch = X[bindex]
            y_batch = y[bindex]
            # 计算损失和梯度
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss) # 记录损失
            # 使用梯度与学习率更新权重，SGD
            self.W -= learning_rate * grad
            # 打印本次迭代计算的损失
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
        return loss_history

    def predict(self, X):
        """
        通过训练得到最优W后，可以利用其来预测图片，通过score = WX计算分数
        @param X:待预测的图像数据集(N×D)
        @return:y_pred预测得到的标签集
        """
        y_pred = np.zeros(X.shape[0])
        # 计算得分矩阵(N×C)
        score_for_all = X.dot(self.W.T)
        y_pred = np.argmax(score_for_all, axis=1)
        return y_pred



    def loss(self, X_batch, y_batch, reg):
        """
        提供一个loss函数的接口以供子类重写
        @param X_batch:经过挑选后的图像集
        @param y_batch:经过挑选后的标签集
        @param reg:正则惩罚参数
        @return:
        """
        pass


"""
继承LinearClassifier的一个子类，其中SVM使用重写了loss方法，这个类同样拥有其他方法
"""


class LinearSVM(LinearClassifier):
    def loss(self, X_batch, y_batch, reg):
        """

        @param X_batch:
        @param y_batch:
        @param reg:
        @return:
        """
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


"""
继承LinearClassifier的一个子类，其中softmax使用重写了loss方法，这个类同样拥有其他方法
"""


class LinearSoftmax(LinearClassifier):
    def loss(self, X_batch, y_batch, reg):
        """

        @param X_batch:
        @param y_batch:
        @param reg:
        @return:
        """
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
