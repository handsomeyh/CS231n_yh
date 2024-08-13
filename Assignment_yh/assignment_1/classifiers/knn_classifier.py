"""
@FileName：knn_classifier.py
@Description：K最邻近分类器算法
@Author：Hang Yin
@Time：2024/8/8 11:22
"""
import numpy as np

"""
创建K最邻近分类器
基本思想：不仅只找到一个最邻近而是找到k个最邻近，由k个进行标签投票，票数最多获胜
"""
# 继承object
class K_nearest_neighbor_classifier(object):
    """
    knn类，其中选择使用L1与L2距离
    """
    def __init__(self):
        pass

    def train(self, X, Y):
        """
        进行分类器训练，对于数据驱动来说就是将数据存在类的字段中
        @param X: 图像数据集
        @param Y: 标签数据集
        @return: null
        """
        self.train_X = X
        self.train_Y = Y

    def predict(self, X, k=1, loops=0):
        """
        进行标签预测，对输入测试图像集预测其标签集
        @param X:输入图像集
        @param k:为预测标签投票的最邻近邻居个数
        @param num_loops:选择计算距离度量的函数的循环个数，可以采用0，1，2三种循环方式实现
        @return:返回预测标签集
        """
        distances = self.select_computer_distance(X, loops)
        return self.predict_labels(distances, k)

    def select_computer_distance(self, X, loops=0):
        """
        将选择循环次数模块化
        @param X:输入图像集
        @param loops:选择计算距离度量的函数的循环个数，可以采用0，1，2三种循环方式实现
        @return:返回距离集
        """
        if loops == 0:
            return self.computer_distance_by_no_loops(X)
        elif loops == 1:
            return self.computer_distance_by_one_loops(X)
        elif loops == 2:
            return self.computer_distance_by_two_loops(X)
        else:
            print("循环跳数非法输入！！！")


    def computer_distance_by_two_loops(self, X):
        """
        使用了两次嵌套循环，进行距离度量的计算
        @param X:待预测测试集
        @return:返回预测测试图像集与训练图像集之前的距离集，二维形状，i测试坐标，j训练坐标
        """
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        distances = np.zeros((num_test, num_train))
        # 进行嵌套循环，先测试，后训练
        for i in range(num_test):
            for j in range(num_train):
                # L1:曼哈顿距离
                # distances[i, j] = np.sum(np.abs(X[i, :] - self.train_X[j]), axis=1)
                # L2：欧几里得距离
                distances[i, j] = np.sqrt(np.sum((self.train_X[j] - X[i])**2))
        return distances


    def computer_distance_by_one_loops(self, X):
        """
        与上面相同，之所以能够使用1次循环是利用了numpy数组的广播特性以及逐元素计算的特性
        @param X: 待预测测试集
        @return:返回预测测试图像集与训练图像集，同样是二维的
        """
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        distances = np.zeros((num_test, num_train))
        for i in range(num_test):
            # L1:曼哈顿距离
            # distances[i] = np.sum(np.abs(self.train_X - X[i]), axis=1)
            # L2：欧几里得距离
            distances[i] = np.sqrt(np.sum((self.train_X - X[i])**2, axis=1))
        return distances

    def computer_distance_by_no_loops(self, X):
        """
        与上面相同，之所以能够使用0次循环是利用了numpy数组的广播特性以及逐元素计算的特性
        @param X: 待预测测试集
        @return:返回预测测试图像集与训练图像集，同样是二维的
        """
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        distances = np.zeros((num_test, num_train))
        # L1：曼哈顿距离，无法不循环，绝对值不好讨论
        # L2：欧几里得距离
        # 将公式的平方拆分开来
        distances = np.sqrt(-2*np.dot(X, self.train_X.T) + np.sum(np.square(self.train_X), axis=1) + np.transpose([np.sum(np.square(X), axis=1)]))
        return distances

    def predict_labels(self, distances, k=1):
        """
        使用测试点与训练点之间的距离集，来预测一个标签为每个测试点，预测方式K最邻近投票
        @param distances:距离集
        @param k:超参数K值
        @return:返回测试集对应的标签集
        """
        num_test = distances.shape[0]
        Y_pred = np.zeros(num_test)
        # i代表第i个训练图像
        for i in range(num_test):
            # 初始化一个最邻近标签集，为每次选择缓存最邻近标签
            closest_y = []
            # 对第i个图像的距离进行排序，切片得到前面k个
            closest_y = self.train_Y[np.argsort(distances[i])[0:k]]
            # 技术投票寻出最大，作为第i张图片的标签，记住标签是0-9的数字表示的，不是字符串
            # 使用bincount得到长度为10的数组，索引是标签，值是标签出现次数
            Y_pred[i] = np.bincount(closest_y).argmax()
        return Y_pred



