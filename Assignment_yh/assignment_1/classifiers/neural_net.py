"""
@FileName：neural_net.py
@Description：实现两层神经网络图像分类器
@Author：Hang Yin
@Time：2024/10/18 15:35
"""
from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange


class TwoLayerNet(object):
    """
    一个两层的全连接网络(FC)，网络结构如下：
    1个N维输入层，1个H维隐藏层，最后输出C个类的分数。
    训练设计：
    softmax损失函数、L2正则表达式、ReLU激活函数
    简单可视化：
    input - fully connected layer - ReLU - fully connected layer - softmax
    第二个全连接层的输出是每个类的分数
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        初始化模型。连接线上的权重初始化为小随机值，偏差初始化为0，采用字典与矩阵存储
        @param input_size:输入向量的维度D，输入的应该是展平的向量
        @param hidden_size:隐藏层的维度H
        @param output_size:输出层的维度C，也是类的个数
        @param std:标准差，用于权重的初始化设置
        """
        """
        需要设置的参数有W1,b1,W2,b2,他们的形状：
        W1:(D,H)
        b1:(H,)
        W2:(H,C)
        b2:(C,)
        """
        self.params = dict()
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        计算两层神经网络的损失和梯度
        @param X:输入的图像数据集(N,D),N个图片，每个D维
        @param y:对应的标签数据集(N,),N个图片对应N个标签，有监督学习
        @param reg:正则常量
        @return:y为None返回分数矩阵(N,C),y不为None返回损失矩阵、梯度矩阵
        """
        # 拆包参数
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N,D = X.shape[0], X.shape[1]

        # 计算前向传播forward
        scores = None
        h_out1 = np.maximum(0,np.dot(X,W1) + b1) # (N, H)
        scores = np.dot(h_out1,W2)+b2 # (N, C)
        if y is None:
            return scores

        # 计算损失, 总损失 = 数据损失(softmax) + 正则损失(L2:1/2*reg*W*W)
        loss = None
        # 数据损失,只需要用到最后的分数scores
        # 减去最大值，防止指数化越界
        shift_scores = scores - np.max(scores, axis=1).reshape(-1, 1)
        softmax_output = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis=1).reshape(-1, 1)
        loss = -1 * np.sum(np.log(softmax_output[range(N), list(y)]))
        loss /= N
        #正则损失
        loss += 0.5 * reg * (np.sum(W1*W1) + np.sum(W2*W2))

        # 计算梯度,反向传播:本地梯度×前向梯度，链式法则，并且用字典存储
        grads = dict()
        # 需要计算的是损失之于所有参数的梯度，W1,b1，W2，b2
        # 先计算softmax中dLoss/dscore (N, C)
        dscore = softmax_output.copy()
        dscore[range(N), list(y)] -= 1
        dscore /= N
        grads['W2'] = np.dot(h_out1.T, dscore) + reg * W2
        grads['b2'] = np.sum(dscore, axis = 0)
        dh = np.dot(dscore, W2.T)
        dh_Relu = (h_out1 > 0) * dh
        grads['W1'] = np.dot(X.T, dh_Relu) + reg*W1
        grads['b1'] = np.sum(dh_Relu, axis = 0)
        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        进行训练，SGD进行梯度下降，进行最优化
        @param X:训练图像集(N,D)
        @param y:训练标签集，有监督学习
        @param X_val:验证图像集
        @param y_val:验证标签集
        @param learning_rate:学习率，步长
        @param learning_rate_decay:随周期改变学习的变化值
        @param reg:正则系数
        @param num_iters:最优化时，每轮下降梯度的次数
        @param batch_size:批次，每个step训练使用的数据个数
        @param verbose:选择打印最优化的过程
        @return:记录历史损失、历史训练集精度、历史验证集精度的字典
        """
        # 得到训练数据个数
        num_train = X.shape[0]
        # 计算训练轮数epoch
        iterations_per_epoch = max(num_train / batch_size, 1)
        # 使用SGD进行最优化我们的模型
        loss_history = list()
        train_acc_history = list()
        val_acc_history = list()
        for it in range(num_iters):
            X_batch, y_batch = None, None
            # 创建一个随机小批次
            idx = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[idx]
            y_batch = y[idx]
            # 计算损失和梯度
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)
            # 使用梯度进行梯度下降更新参数
            self.params['W2'] += - learning_rate * grads['W2']
            self.params['b2'] += - learning_rate * grads['b2']
            self.params['W1'] += - learning_rate * grads['W1']
            self.params['b1'] += - learning_rate * grads['b1']
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        前向传播，进行预测，选择最大分数
        @param X:将要预测的图像数据集(N,D)
        @return:返回预测的类
        """
        y_pred = None
        h = np.maximum(0, np.dot(X, self.params['W1'])+self.params['b1'])
        scores = np.dot(h,self.params['W2'])+self.params['b2']
        y_pred = np.argmax(scores, axis=1)
        return y_pred