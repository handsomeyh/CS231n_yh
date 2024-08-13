import numpy as np
from imageio import imread
import matplotlib.pyplot as plt

img = imread('./data/img1.jpg')
img_tinted = img * [1, 0.95, 0.9]

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(img)

"""
关于子图subplot():
subplot() 方法在绘图时需要指定位置,subplots() 方法可以一次生成多个，在调用时只需要调用生成对象的 ax 即可
以上函数将整个绘图区域分成 nrows 行和 ncols 列，然后从左到右，从上到下的顺序对每个子区域进行编号 1...N ，（从1开始，不是0）
左上的子区域的编号为 1、右下的区域编号为 N，编号可以通过参数 index 来设置
比如并列放两张图：第一章subplot(1,2,1) 第二章subplot(1,2,2) 
"""
# Show the tinted image
plt.subplot(1, 2, 2)

# A slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(img_tinted))
plt.show()
