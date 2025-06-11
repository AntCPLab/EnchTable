import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_single_wave_3d(save_path='figures', filename='wave_3d_icon'):
    # 创建网格数据（x: 波动方向，y: 横向扩展）
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.linspace(-1, 1, 60)
    X, Y = np.meshgrid(x, y)

    # 构造波浪函数（Z 值）—— 正弦波沿 x 方向传播，并降低振幅
    Z = 0.5 * np.sin(X) * np.exp(-0.2 * Y**2)  # 振幅减半，波形更平缓

    # 创建 3D 绘图
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制曲面
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0, antialiased=True, alpha=0.95)

    # 隐藏坐标轴和边框
    ax.set_axis_off()

    # 设置背景透明
    fig.patch.set_alpha(0.0)
    ax.patch.set_facecolor('none')

    # 设置视角角度（比之前向上转动 20 度）
    ax.view_init(elev=55, azim=-110)  # elev 提高到 55°，从上往下看

    plt.tight_layout()

    # 保存为透明背景 PNG
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        png_path = os.path.join(save_path, f"{filename}.png")
        fig.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
        print(f"三维波浪图已保存至: {png_path}")

    plt.close(fig)


if __name__ == "__main__":
    plot_single_wave_3d(save_path='figures', filename='wave_3d_icon')
