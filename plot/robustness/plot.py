import matplotlib.pyplot as plt
import matplotlib

print(matplotlib.matplotlib_fname())
# import seaborn as sns
import numpy as np
fontsize=12
lw = 4
matplotlib.rcParams.update({'font.size': fontsize, 'font.family': 'Times New Roman', 'font.weight': 'normal'})

# 设置风格
# sns.set(style="whitegrid")

# 模拟数据：每个子图有9个柱子（3组，每组3个）
np.random.seed(42)
attacks = ['Role Play', 'Slang', 'Miss Spelling', 'Logic Apeal', 'Authority Endorsement', 'Misrepresentation', 'Evidence-based Persuasion', 'Expert Endorsement', 'ICA', 'DRA']

data = [
    [0.3636, 0.1341, 0.8886, 0.0409, 0.4932, 0.0227, 0.1614],
    [0.4182, 0.125, 0.8045, 0.1136, 0.4182, 0.0523, 0.1818],
    [0.4614, 0.0705, 0.7750, 0.1, 0.4727, 0.0477, 0.2568],
    [0.6795, 0.1977, 0.7523, 0.3705, 0.8000, 0.1602, 0.4182],
    [0.6136, 0.2227, 0.8227, 0.3182, 0.8500, 0.0864, 0.4114],
    [0.6, 0.2795, 0.8295, 0.475, 0.8205, 0.1591, 0.3318],
    [0.5705, 0.1114, 0.6318, 0.2841, 0.7091, 0.0568, 0.2841],
    [0.6114, 0.1659, 0.7818, 0.3614, 0.8227, 0.1068, 0.3364],
    # [0.05, 0.025, 0.675, 0, 0.565, 0.165, 0.195],
    # [0.13, 0.025, 0.9100, 0.13, 0.8100, 0.355, 0.29]
    [0.02, 0.005, 0.585, 0, 0.525, 0.09, 0.19],
    [0.805, 0.035, 0.91, 0, 0.915, 0, 0.92]
]
data = {
    f"Subplot {i+1}": {
        'Instruct': [data[i][0], data[i][0], data[i][0]],   # 3个值
        'SFT': [data[i][2], data[i][4], data[i][6]],        # 3个值
        'Ours': [data[i][1], data[i][3], data[i][5]]       # 3个值
    }
    for i in range(10)
}


# 创建2x5的子图布局
fig, axes = plt.subplots(2, 5, figsize=(14, 5), sharey=True)
# fig, axes = plt.subplots(2, 5, figsize=(20, 8))

# 方法名和子项数量
methods = ['Instruct', 'SFT', 'Ours']
sub_items = ['Code', 'Math', 'Medical']

# 颜色设置
# colors = sns.color_palette("Set2", 3)
# colors = ['#F0E68C', '#FFA500', '#00008B', '#ADD8E6', '#90EE90']
colors = ['#4A90E2', '#BDC3C7', '#FFA726']

# x轴位置
bar_width = 0.25
index = np.arange(len(sub_items))

# 遍历所有子图画图
for idx, (title, values) in enumerate(data.items()):
    ax = axes[idx // 5, idx % 5]

    # 绘制每一组柱状图
    for i, method in enumerate(methods):
        ax.bar(index + i * bar_width, values[method], width=bar_width,
               label=method, color=colors[i], zorder=100, edgecolor='black')

    # 子图标题与样式
    if idx % 5 == 0:
        ax.set_ylabel('Unsafe Rate')
    ax.set_title(attacks[idx])
    ax.set_xticks(index + bar_width)
    ax.tick_params(axis='x', which='both', length=0)
    ax.set_xticklabels([f"{m}" for m in sub_items])
    ax.set_ylim(0, 1.0)
    # ax.legend(prop={'size': 8})
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

# 添加图例
fig.legend(handles=[plt.Rectangle((0,0),1,1, color=colors[i], edgecolor='black') for i in range(len(methods))],
           labels=methods,
           loc='upper center',
           bbox_to_anchor=(0.5, 1.0),
           ncol=len(methods),
           prop={'size': 12})

# 自动调整子图间距
plt.tight_layout()
fig.subplots_adjust(top=0.85)
# ax.tick_params(axis='x', which='both', length=0)
fig.savefig('robustness.png')
fig.savefig('robustness.pdf')
