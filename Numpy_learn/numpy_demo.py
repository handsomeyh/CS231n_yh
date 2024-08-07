# 首先导入需要的numpy包
import numpy as np

# 初试化一个一维数组，秩为1
a = np.array([1, 2, 3, 4])
print(a)
# shape属性获得数组的形状，先行后列，(4,)表示一维，4个元素
print(type(a), a.shape, a[0], a[1], a[2])
a[0] = 9
print(a)

print("=================1=================")
# 初始化二维数组,可以看作
"""
b = [[1, 2, 9], 
     [3, 4, 8]]
"""
b = np.array([[1, 2, 9], [3, 4, 8]])
print(b.shape)
print(b)
print(b[0, 0], b[1, 0])

print("=================2=================")
# 创建数组的方式
# 1.zeros()用来创建一个全0.的数组，可以通过参数确定数组形状与数据类型,数据类型默认为float，形状由第一个参数(3,4)决定
zeros = np.zeros((3, 4), int)
print(zeros)
print(zeros.shape)
# 2.ones()与zeros()的作用相同，唯一不同在于数据是1
ones = np.ones((3, 4), int)
print(ones)
# 3.full()则是创建一个全是指定数字的数组，其他效果相同
eights = np.full((3, 4), 8, int)
print(eights)
# 4.eye()则是返回一个二维数组，对角线上为1，其他位置为零。,
c = np.eye(2)
print(c)
d = np.eye(3, k=1, dtype=int)  # 还可以进行偏移
print(d)
# random()可以创建随机填充的数组,random的范围再(0,1)之间
e = np.random.random((3, 3))
print(e)
# arange()可以在范围内创建数组
f = np.arange(1, 20, 2, int)
print(f)

print("================3==================")
"""
数组切片索引
记住索引的规则[start: stop], 左闭右开的区间
重要注意，切片获得的矩阵仅仅是一个视图，用的数据的地址依旧是在原矩阵上面
"""
# 首先，创建一个秩为2，形状为（3，4）的数组
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
# 使用切片其中一个子矩阵
arr1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=int)
print(arr1)
# [[1 2]
#  [5 6]]
arr_son1 = arr1[:2, :2]
print(arr_son1)
# [[ 2  3  4]
#  [ 6  7  8]
#  [10 11 12]]
arr_son2 = arr1[:, 1:]
print(arr_son2)
# 我们对索引切片的子矩阵进行数据改变, 原矩阵也同样发生了变化
arr_son2[0, 0] = -2
print(arr_son2)
print(arr1)

print("================4==================")
# 我们能够混合使用整数索引与切片索引，这样可以获得降维的矩阵向量
# 重新定义矩阵
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
arr2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=int)
print(arr2, arr2.shape)
# 我们来获得第一行行向量
arr2_vector1 = arr2[0, :]
print(arr2_vector1, arr2_vector1.shape)
# 将矩阵进行分块，得到第一行形成的分块矩阵
arr2_son1 = arr2[0:1, :]
print(arr2_son1, arr2_son1.shape)
# 以相同的方式可以获取列向量
arr2_vector2 = arr2[:, 1]
arr2_son2 = arr2[:, 1:2]
print(arr2_vector2, arr2_vector2.shape)
print(arr2_son2, arr2_son2.shape)

print("================5==================")
"""
整数数组索引，
与切片索引不同，他可以使用另一数组的内容来构造任意数组，这种方式不是创建子数组视图，
而是根据索引数组来选择原始数组的元素，创建一个全新的数组。
"""
arr3 = np.array([[1, 2], [3, 4], [5, 6]], dtype=int)
print(arr3)
# 使用整数数组索引，第一个数组表示行坐标，第二个是列坐标,
print(arr3[[0, 1, 2], [0, 1, 0]])  # [1 4 5]
# 创建一个新的数组使用整数数组索引
arr3_new = np.array([arr3[0, 0], arr3[1, 1], arr3[2, 0]])
print(arr3_new)
# 在整数数组索引，一个有用的技巧是从矩阵的每一行中选择或者改变一个元素
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]]
# 形状为（4，3）
arr4 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=int)
print(arr4)
# 创建一个索引数组
indices = np.array([0, 2, 0, 1])
# 对于第一给行索引数组的技巧就是用范围创建数组函数arange(), 列索引数组就自定义
arr4_new1 = np.array(arr4[np.arange(4), indices])
print(arr4_new1)
# 指定位置进行批量数据操作
arr4[np.arange(4), indices] += 10
print(arr4)

print("================6==================")
"""
布尔数组索引
"""
arr5 = np.array([[1, 2], [3, 4], [5, 6]], dtype=int)
print(arr5)
# 布尔数组
bool_idx = (arr5 > 2)
# [[False False]
#  [ True  True]
#  [ True  True]]
print(bool_idx)
print(arr5[bool_idx])
print(arr5[(arr5 > 2) & (arr5 < 6)])

print("================7==================")
"""
基本的数学运算可以在数组上进行逐元素的操作，
这些数学函数可以作为运算重载符出现也可以作为模块函数出现
矩阵x：
[[1, 2], 
 [3, 4]]
矩阵y：
[[5, 6], 
 [7, 8]]
"""
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)
# 加法：+ 或者用numpy的函数add()
print("加：")
print(x + y)
print(np.add(x, y))
# 减
print("减：")
print(x - y)
print(np.subtract(x, y))
# 乘，点乘要么向量要么同型
print("乘：")
print(x * y)
print(np.multiply(x, y))
# 除
print("除：")
print(x / y)
print(np.divide(x, y))
# 开方
print("开放：")
print(np.sqrt(x))
# 以上五种都是逐元素计算
# 向量求内积（数量积），本质上就是矩阵乘法，不使用*，而是使用numpy的dot()或者@
# 定义向量
print("矩阵叉乘：")
vector1 = np.array([9, 10])
vector2 = np.array([11, 12])
# 进行向量求内积, 矩阵乘法（矩阵外积）是向量的内积
print(vector1.dot(vector2))  # produce 219
print(np.dot(vector1, vector2))
print(vector1 @ vector2)
# 向量乘矩阵,
print(x.dot(vector1))
print(np.dot(x, vector1))
print(vector1.dot(x))
print(np.dot(vector1, x))
# 矩阵叉乘
print(x.dot(y))
print(np.dot(x, y))
print(x@y)
# sum()函数用来求和矩阵所有元素，每行之和，每列之和,，axis参数默认空，0求列和，1求行和
# [[1, 2],
#  [3, 4]]
print(np.sum(x))
print(x.sum())
print(np.sum(x, axis=0))
print(x.sum(axis=0))
print(np.sum(x, axis=1))
print(x.sum(axis=1))
# 矩阵转置，使用数组对象的T属性即可,但是向量不会变化，因为反应不出横竖
print(x.T)
print(vector1.T)
print("================7==================")
"""
广播机制：他允许Numpy在执行算术运算时处理不同形状的数组。
"""
# 示例1：添加一个常熟向量到矩阵的每一行上
x_b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v_b = np.array([1, 0, 1])
# 创建一个形似x_b的空矩阵
y_b1 = np.empty_like(x_b)
# 使用循环的方式进行加和
print("使用循环方式：")
for i in range(4):
    y_b1[i, :] = x_b[i, :] + v_b
print(y_b1)

# 循环方式在处理大数据集的方式时效率不高，
# 将向量 v 加到矩阵 x 的每一行上，等同于创建一个新的矩阵 vv，
# 这个矩阵是通过将向量 v 垂直堆叠多次形成的。
# tile()
# v_b本身是一维，（4，1）先第一维复制一次，第二维复制4次
print("使用tile():")
vv_b = np.tile(v_b, (4, 1))
print(vv_b)
y_b2 = x_b + vv_b
print(y_b2)

# 使用广播的方式,自动扩展形状较小的数组来适配较大的数组
print("使用广播的方式：")
y_b3 = x_b + v_b
print(y_b3)

print("================7==================")
"""
广播的一些用法
"""
# 用法1：计算两个向量的外积，也就是叉乘
vector3 = np.array([1, 2, 3])
vector4 = np.array([4, 5])
# 先将vector3转换成列向量（3，1），通过广播机制求得，向量外积是列向量乘行向量
outer = np.reshape(vector3, (3, 1)) * vector4
print(outer)
# print(vector3 * vector4)
# print(vector3 @ vector4)
# print(vector3 @ vector4)
# print(np.reshape(vector3, (3, 1)) @ np.reshape(vector4, (1, 3)))
# 用法2：广播机制为每一行增加一个向量，上面已经用到过
m1 = np.array([[1, 2, 3], [4, 5, 6]])
print(m1 + vector3)  # 广播机制扩展了vector3的形状
# 用法3：为每一列增加一个向量,利用好转置
print((m1.T + vector4).T)
# 或者可以重塑vector4的形状，利用广播机制进行加
print(m1 + np.reshape(vector4, (2, 1)))
# 用法3：矩阵乘数字
print(m1 * 2)

