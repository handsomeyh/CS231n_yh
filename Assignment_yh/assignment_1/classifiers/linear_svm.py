"""
@FileName：linear_svm.py
@Description：svm损失函数，分为非向量、半向量、全向量
@Author：Hang Yin
@Time：2024/8/14 16:13
"""

import numpy as np
from random import shuffle


def L_i(W, x, y):
    """
    计算单一图片SVM损失的非向量实现
    @param W:权重矩阵其中拟合了偏差，形状为(10×3073)
    @param x:一个图像的数据，一个列向量(3073×1)
    @param y:图像对应的正确类的标签
    @return: 返回该图片对于所有类最后得到的损失值
    """
    # SVM偏差值
    delta = 1
    # 计算分数
    scores = W.dot(x)
    # 计算损失
    loss_i = 0
    correct_score = scores[y]
    num_class = W.shape[0]
    for j in range(num_class):
        if j == y:
            # 如果遍历到正确的类，跳过
            continue
        margin = scores[j] - correct_score + delta
        if margin > 0:
            loss_i += margin
    return loss_i


def L_i_vectorized(W, x, y):
    """
    计算单一图片SVM损失的向量实现
    @param W:权重矩阵其中拟合了偏差，形状为(10×3073)
    @param x: 一个图像的数据，一个列向量(3073×1)
    @param y:图像对应的正确类的标签
    @return: 返回该图片对于所有类最后得到的损失值
    """
    # SVM偏差值
    delta = 1
    # 计算分数，scores是[10×1]
    scores = W.dot(x)
    # 计算损失,利用向量广播,结果的形状与scores相同
    margins = np.maximum(0, scores - scores[y] + delta)
    margins[y] = 0
    loss_i = margins.sum()
    return loss_i


def svm_loss_naive(W, X, y, reg):
    """
    计算图片集的SVM损失的非向量实现，
    @param W:包含权重与偏差的权重矩阵形状为（10×3073）
    @param X:测试图像集（N×3073）
    @param y:图像集对应的标签集（N,）其中的元素是标签，对应cifar10就是0-9的整数
    @param reg:正则化惩罚的参数
    @return: 1.返回最终损失（） 2.返回梯度矩阵（形状与W相同10×3073）
    """
    # 先计算数据损失loss, 梯度学完优化再进行计算梯度
    dW = np.zeros_like(W)
    num_class = W.shape[0]
    num_train = X.shape[0]
    loss = 0.0
    delta = 1
    for i in range(num_train):
        x_i = X[i].reshape((-1, 1))
        scores = W.dot(x_i)
        correct_class_score = scores[y[i]]
        for j in range(num_class):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + delta
            if margin > 0:
                # 进行损失值的加和
                loss += margin
                # 更新梯度矩阵,由于折页函数要在满足margin>0
                dW[j] += X[i]
                dW[y[i]] -= X[i]
    loss /= num_train
    dW /= num_train

    # 然后是正则化损失，使用*就逐元素乘法
    loss += reg*np.sum(W*W)
    dW += 2 * reg * W

    # 计算梯度,遍历每个类
    # 梯度的计算先分析微分对wyi进行微分得到公式然后实现

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    计算图片集SVM损失的半向量实现
    @param W:包含权重与偏差的权重矩阵形状为（10×3073）
    @param X:测试图像集（N×3073）
    @param y:图像集对应的标签集（N,）其中的元素是标签，对应cifar10就是0-9的整数
    @param reg:正则化惩罚的参数
    @return: 1.返回损失（） 2. 相对与权重的梯度（应该是与优化有关）
    """
    loss = 0.0
    delta = 1
    num_train = X.shape[0]
    # 一次性得到所有图像的分数，形状是N×10
    scores_for_all = X.dot(W.T)
    # 得到所有的图片的数据损失，其中使用了整数数组索引、广播等机制,也可以使用布尔矩阵,这里需要将正确分裂列向量转置为行向量
    margin_for_all = np.maximum(0, scores_for_all - scores_for_all[range(num_train), [y]].T + delta)
    margin_for_all[range(num_train), y] = 0
    loss_for_all = np.sum(margin_for_all, axis=1)
    loss = loss_for_all.sum()/num_train
    # 然后是正则化损失，使用*就逐元素乘法
    loss += reg * np.sum(W * W)

    # 计算梯度矩阵
    dW = np.zeros_like(W)
    dS = np.zeros_like(margin_for_all)
    # counts二维的布尔矩阵,将系数矩阵独(N,10)，符合函数dSdot(X)
    dS = (margin_for_all > 0).astype(int)
    dS[range(num_train), y] = -np.sum(counts, axis=1)
    dW = dS.T.dot(X)/num_train + 2 * reg * W

    return loss, dW