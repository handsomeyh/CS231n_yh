"""
@FileName：knn_classifier.py
@Description：最邻近分类器算法
@Author：Hang Yin
@Time：2024/8/8 11:22
"""
import numpy as np

"""
构建最邻近分类器类
其中类中有两个主要的分类APi：train()与predict()
我们需要将三维图片摊开成为1维数组，这样方便度量距离的计算
"""
# python3中无需手动继承object
class nearest_neighobr_classifier:
    def __init__(self):
        pass

    def trans(self, X, Y):
        """
        进行训练，对于数据驱动来说就是将所有图像（使用二维数组表示）与标签存储在类中
        @param X:所有训练图像
        @param Y:所有图像的标签
        @return:不返回，相当于做初始化
        """
        self.Xtr = X
        self.Ytr = Y

    def predict1(self, X):
        """
        进行预测，p1中使用曼哈顿距离作为度量距离
        @param X:输入测试图像集（N*3072），每一行是一个图片
        @return:输出预测标签集
        """
        tset_num = X.shape[0]
        Y_pred = np.zeros(tset_num, dtype=self.Ytr.dtype)

        # 遍历测试集中的所有图像
        for i in range(tset_num):
            # 先读取将要预测的图像
            # 曼哈顿距离集，其中利用了广播的思想，提高效率,self.Xtr-X[i]
            # axis=1进行横向相加
            distances = np.sum(np.abs(self.Xtr - X[i]), axis=1)
            # 找到曼哈顿距离最小
            min_index = np.argmin(distances)
            Y_pred[i] = self.Ytr[min_index]
        return Y_pred


    def predict2(self, X):
        """
        进行预测，p2中使用欧几里得作为度量距离
        @param X:输入测试图像集
        @return:输出预测标签集
        """
        tset_num = X.shape[0]
        Y_pred = np.zeros(tset_num, dtype=self.Ytr.dtype)

        # 遍历测试集中的所有图像
        for i in range(tset_num):
            # 先读取将要预测的图像
            # 曼哈顿距离集，其中利用了广播的思想，提高效率,self.Ytr - img_i
            # axis=1进行横向相加
            distances = np.sqrt(np.sum((self.Xtr-X[i])**2, axis=1))
            # 找到曼哈顿距离最小
            min_index = np.argmin(distances)
            Y_pred[i] = self.Ytr[min_index]
        return Y_pred



