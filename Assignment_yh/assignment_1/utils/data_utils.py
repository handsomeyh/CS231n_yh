"""
@FileName：data_utils.py
@Description：用于读取数据的工具
@Author：Hang Yin
@Time：2024/8/9 22:20
"""
import os
import pickle as pl
import numpy as np


def unpickle(file):
    """
    将cifar10文件开箱
    @param file:cifar10文件路径名
    @return:数据集的字典类型，字典有两个字段
    """
    with open(file, 'rb') as fo:
        dict = pl.load(fo, encoding='bytes')
    return dict


def load_CIFAR10_batch(file_path):
    """
    用来加载单一的数据集，将这一部分功能模块化
    @param file_path:经过拼接形成的文件路径
    @return:返回该数据集的四维形状，刚从字典中读取出来的是二维的（10000，3072），要正常切分样本数，高，宽，通道数（10000，32，32，3）
    """
    data_dict = unpickle(file_path)
    # X
    X = np.array(data_dict[b'data']).astype(np.float32)
    # 本身是一维的
    Y = np.array(data_dict[b'labels'])
    # 图像按行优先顺序存储，因此数组的前 32 个条目是图像第一行的红色通道值。以此类推，一定要先分三份再高维装置
    X = X.reshape(X.shape[0], 3, 32, 32).astype(np.float32)
    X = X.transpose(0,2,3,1)
    return X, Y


def load_CIFAR10(ROOT):
    """
    加载CIFAR10数据集中的数据，训练集有5个，测试集有1个，需要读取五个数据集
    @param file_path: 文件路径
    @return:训练集图像矩阵、训练集标签、测试集图像矩阵、测试集标签，并且这些数据被拉成一维数组
    """
    # 1.获取训练数据样本
    # 用于及
    xs = []
    ys = []
    for i in range(1, 6):
        fname = os.path.join(ROOT, 'data_batch_%d' % i)
        X, Y = load_CIFAR10_batch(fname)
        xs.append(X)
        ys.append(Y)
    # xs中5个二维数组，对其进行纵向拼接，axix=0为纵向拼接
    Xtr = np.concatenate(xs, axis=0)
    Ytr = np.concatenate(ys, axis=0)

    # 2.获取测试数据样本，测试集只有一个
    testpath = os.path.join(ROOT, 'test_batch')
    Xts, Yts = load_CIFAR10_batch(testpath)
    return Xtr, Ytr, Xts, Yts



