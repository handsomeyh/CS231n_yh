"""
@FileName：numpy_demo2.py
@Description：在做实验时对一些numpy函数进行测试
@Author：Hang Yin
@Time：2024/8/8 18:13
"""
import numpy as np

Ypred = np.zeros(10000, dtype=np.float32)

a1 = np.arange(1, 10)
a2 = np.arange(1, 10)
a3 = (a1 == a2)
print(a3)

print([1]*10)
print([[1]]+[[2]])
a = [[1]]
b = [[2]]
print(np.vstack((a,b)))


