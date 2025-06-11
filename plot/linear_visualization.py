import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_3d_activation(func_name='relu', save_path='figures', filename=None):
    # 创建网格数据
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)

    # 根据函数名称计算 Z 值
    if func_name == 'relu':
        Z = np.maximum(X, Y)
    elif func_name == 'sigmoid':
        Z = 1 / (1 + np.exp(-(X**2 + Y**2)))
    elif func_name == 'tanh':
        Z = np.tanh(np.sqrt(X**2 + Y**2))
    elif func_name == 'leaky_relu':
        alpha = 0.1
        Z = np.where(X >= 0, X, alpha * X) + np.where(Y >= 0, Y, alpha * Y)
    elif func_name == 'swish':
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))
        Z = (X + Y) * sigmoid(X + Y)
    elif func_name == 'linear':
        Z = X + Y
    else:
        raise ValueError("Unsupported activation function")

    # 创建 3D 绘图
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制曲面
    ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0, antialiased=True, alpha=0.9)

    # 隐藏坐标轴和边框
    ax.set_axis_off()

    # 设置背景透明
    fig.patch.set_alpha(0.0)
    ax.patch.set_facecolor('none')

    # 调整视角角度（可自定义）
    ax.view_init(elev=30, azim=-60)

    plt.tight_layout()

    # 保存为透明背景 PNG
    if not filename:
        filename = f"activation_3d_{func_name}"

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        png_path = os.path.join(save_path, f"{filename}.png")
        fig.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
        print(f"透明背景的 3D 激活函数图已保存至: {png_path}")

    plt.close(fig)


if __name__ == "__main__":
    # 支持的类型：'relu', 'sigmoid', 'tanh', 'leaky_relu', 'swish', 'linear'
    plot_3d_activation(func_name='tanh', save_path='figures', filename='activation_icon_3d')
