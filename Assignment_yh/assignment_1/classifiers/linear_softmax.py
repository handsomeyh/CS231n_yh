"""
@FileName：linear_softmax.py
@Description：softmax损失函数
@Author：Hang Yin
@Time：2024/8/14 16:15
"""

import numpy as np
from random import shuffle


def L_i(W, x, y):
    """
    计算单一图片的softmax损失的非向量实现
    @param W:权重矩阵其中拟合了偏差，形状为(10×3073)
    @param x:一个图像的数据，一个列向量(3073×1)
    @param y:图像对应的正确类的标签
    @return: 返回该图片对所有类最后得到的softmax损失值
    """
    # 计算得分（10×1）
    scores = W.dot(x)
    num_class = W.shape[0]
    # 进行归一化
    scores -= np.max(scores, axis=1)
    p = np.exp(scores) / np.sum(np.exp(scores))
    loss_i = -1 * np.log(p)
    return loss_i


def softmax_loss_naive(W, X, y, reg):
    """
    计算图片集的softmax损失的非向量实现，
    @param W:包含权重与偏差的权重矩阵形状为（10×3073）
    @param X:测试图像集（N×3073）
    @param y:图像集对应的标签集（N,）其中的元素是标签，对应cifar10就是0-9的整数
    @param reg:正则化惩罚的参数
    @return: 1.返回损失（） 2. 相对与权重的梯度（应该是与优化有关）
    """
    # 计算数据损失
    dW = np.zeros_like(W)
    loss = 0.0
    num_train = X.shape[0]
    num_classes = W.shape[0]
    # 对于所有训练图片的分数（N×10）
    scores_for_all = X.dot(W.T)
    for i in range(num_train):
        scores_i = scores_for_all[i]
        scores_i -= np.max(scores_i)
        exp_scores = np.exp(scores_i)
        # 求概率
        p_yi = exp_scores[y[i]] / np.sum(exp_scores)
        loss += -1 * np.log(p_yi)
        dW[y[i]] -= X[i]
        for j in range(num_classes):
            p_j = exp_scores[j] / np.sum(exp_scores)
            dW[j] +=  X[i] *  p_j

    loss /= num_train
    # 计算正则化损失
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += 2 * reg * W


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    计算图片集softmax损失的半向量实现
    @param W:包含权重与偏差的权重矩阵形状为（10×3073）
    @param X:测试图像集（N×3073）
    @param y:图像集对应的标签集（N,）其中的元素是标签，对应cifar10就是0-9的整数
    @param reg:正则化惩罚的参数
    @return: 1.返回损失（） 2. 相对与权重的梯度（应该是与优化有关）
    """
    # 计算数据损失
    loss = 0.0
    num_train = X.shape[0]
    # 对于所有训练图片的分数（N×10）
    scores_for_all = X.dot(W.T)
    # 避免指数化后越界
    f_for_all = scores_for_all - np.max(scores_for_all, axis=1).reshape((-1, 1))
    # 对于所有图片的概率(N×10)
    p_s = np.exp(f_for_all) / np.sum(np.exp(f_for_all), axis=1).reshape((-1, 1))
    # 运用了整数数组索引，两个数组长度相同
    loss_for_all = -1 * np.log(p_s[range(num_train), y])
    loss += np.sum(loss_for_all) / num_train

    # 计算正则化损失
    loss += reg * np.sum(W * W)

    # 求梯度
    dS = np.zeros_like(p_s)
    dS[range(num_train), y] = -1
    dS += p_s
    dW = dS.T.dot(X)
    dW /= num_train
    dW = dW + 2 * reg * W

    return loss, dW
