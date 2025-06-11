import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.style as mplstyle

# 使用 fast 样式以提高性能
mplstyle.use('fast')

# 定义 Sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 定义神经网络类比的曲面函数
def neural_network_surface(x, y):
    return sigmoid(x) * sigmoid(y)

# 创建数据网格
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = neural_network_surface(X, Y)

# 创建一个图形并设置无边框
fig = plt.figure(frameon=False)
ax = fig.add_subplot(111, projection='3d')

# 绘制表面，调整 rstride 和 cstride 以提高分辨率
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', antialiased=True, rstride=5, cstride=5)

# 移除轴
ax.axis('off')

# 保存图像为 PDF
plt.savefig('neural_network_surface.pdf', bbox_inches='tight', pad_inches=0)

# 显示图像（可选）
plt.show()
