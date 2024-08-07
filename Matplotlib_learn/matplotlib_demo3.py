import matplotlib.pyplot as plt
import numpy as np

# linspace()在指定时间间隔内返回均匀分布的样本
a = np.linspace(0, 2*np.pi, 1024)
plt.axes(polar=True)
plt.plot(a, 5*(1-np.sin(a)), color='red')
plt.show()


