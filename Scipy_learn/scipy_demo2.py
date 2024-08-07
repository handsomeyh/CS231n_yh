import numpy as np
from scipy.spatial.distance import pdist, squareform

# 每行创建一个点
x = np.array([[0, 1], [1, 0], [2, 0]])
print(x)

# 计算所有行的欧几里得距离,
dist = squareform(pdist(x, 'euclidean'))
print(dist)
# 结果为：
# [[0.         1.41421356 2.23606798]
#  [1.41421356 0.         1.        ]
#  [2.23606798 1.         0.        ]]
# 意思每个点到其他点的距离，

