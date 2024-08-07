import matplotlib
import matplotlib.pyplot as plt
import numpy as np
"""
matplotlib是python用来绘图的库
"""
# 绘制正选函数，
x = np.arange(0, 5*np.pi, 0.000001)
# 表明x, y之间的函数关系
y = np.sin(x)
# 图形绘制，显示
plt.plot(x, y)
plt.show()

