"""
@FileName：gradient_check.py
@Description：检查分析梯度的正确性（梯度校验）
@Author：Hang Yin
@Time：2024/8/21 17:17
"""
import numpy as np
from random import randrange


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    函数f在x处的数值差值梯度的简单计算，先不用管svm函数与softmax函数,弄懂原理,我们使用中心差值公式
    @param f:待求数值梯度的函数,它是只有一个参数的函数,仅仅返回一个数字和函数
    @param x:待求数值梯度的点位置
    @param verbose:是否进行对应维度以及梯度输出
    @param h:差值定义，点与点之间的距离
    @return:返回数值梯度值
    """
    # 计算最初点的函数值
    fx = f(x)
    grad = np.zeros_like(x)
    # 取得x的迭代器,需要多维坐标（x1,x2,x3,...）
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    # 开始矩阵各个位置的梯度计算
    while not it.finished:
        # 取得位置
        ix = it.multi_index
        val_origin = x[ix]
        # 计算f(x+h)的键值坐标
        x[ix] = val_origin + h
        fx_1 = f(x)
        # 进行f(x-h)的键值计算
        x[ix] = val_origin - h
        fx_2 = f(x)
        # 还原维度位置变量
        x[ix] = val_origin
        # 计算该位置的导数
        grad[ix] = (fx_1 - fx_2) / (2*h)
        if verbose:
            print(ix, grad[ix])
        # 进行下一个位置的迭代
        it.iternext()
    return grad



def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    为一个输入输出都是数组的函数计算数值差值梯度
    @param f:接受一个数组返回一个数组
    @param x:点位置
    @param df:函数f对应的数值梯度
    @param h:差值
    @return:梯度
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        val_origin = x[ix]
        x[ix] = val_origin + h
        pos = f(x).copy() # 输出结构不是一个值而是数组，因此需要拷贝
        x[ix] = val_origin - h
        neg = f(x).copy() # 输出结构不是一个值而是数组
        x[ix] = val_origin
        grad[ix] = np.sum((pos - neg)*df) / (2*h) # 本质是符合函数求导
        it.iternext()
    return grad


def eval_numerical_gradient_blobs(f, inputs, output, h=1e-5):
    """

    @param f:
    @param inputs:
    @param output:
    @param h:
    @return:
    """


def eval_numerical_gradient_net(net, inputs, output, h=1e-5):
    """

    @param net:
    @param inputs:
    @param output:
    @param h:
    @return:
    """


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    """
    进行梯度检测：检查分析梯度的正确性，返回相对误差
    @param f:检测函数
    @param x:检测点
    @param analytic_grad:微分分析梯度
    @param num_checks:检查次数,默认10次
    @param h:差值，默认0.000001=1e-5
    @return:相对误差
    """
    for i in range(num_checks):
        # 随机选择一个位置
        ix = tuple([randrange(m) for m in x.shape])
        val_origin = x[ix]
        x[ix] = val_origin + h
        fx_1 = f(x)
        x[ix] = val_origin - h
        fx_2 = f(x)
        x[ix] = val_origin
        grad_numerical = (fx_1 - fx_2) / (2 * h)
        grad_analytic = analytic_grad[ix]
        # 计算相对误差
        rel_error = relative_error(grad_numerical, grad_analytic)
        print('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))


def relative_error(x, y):
    """
    返回两sh相对误差
    @param x:
    @param y:
    @return:相对误差
    """
    return (abs(x-y))/(abs(x)+abs(y))