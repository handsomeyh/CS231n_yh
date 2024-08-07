# 进行更加精细的绘图
import matplotlib.pyplot as plt
import numpy as np

# 同时在一张图上面绘制正弦与余弦
x = np.arange(0, 5*np.pi, 0.01)
y_sin = np.sin(x)
y_cos = np.cos(x)


plt.plot(x, y_sin, label="sin")
plt.plot(x, y_cos, label="cos")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.legend(['sin', 'cos'])
plt.title('sin and cos')
plt.show()

