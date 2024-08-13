# 导入scipy相关包：图像处理、函数等等
# 完蛋完蛋misc被弃用了，from scipy.misc import imread, imsave, imresize
from imageio import imread, imwrite
from skimage.transform import resize

# 利用misc API进行图像处理，毕竟是计算机视觉
# 首先是读取图片文件，图片用矩阵表示分辨率为1206*1206
# 图像是一个三维图像高1206，宽1206，颜色通道数3（红绿蓝）
img1 = imread("./data/img1.jpg")
print(img1)
print(img1.dtype, img1.shape)

# 对颜色进行调节,通过标量来缩放颜色通道
img1_tinted = img1 * [1.75, 1.75, 1.75]
# 进行大小的缩放，imresize()
img1_tinted = resize(img1_tinted, (800, 800))

# 保存修改后的图片到文件夹
imwrite("./data/img1_tinted3.jpg", img1_tinted, format='jpg')
