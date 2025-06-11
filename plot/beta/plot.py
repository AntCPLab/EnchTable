import matplotlib.pyplot as plt
import matplotlib

fontsize=16
matplotlib.rcParams.update({'font.size': fontsize, 'font.family': 'Times New Roman', 'font.weight': 'normal'})
# 数据定义
x = [0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
he = [0.671, 0.640, 0.665, 0.659, 0.640, 0.555, 0.494, 0.317, 0.012]
he_plus = [0.622, 0.579, 0.610, 0.604, 0.561, 0.494, 0.445, 0.293, 0.006]
unsafety = [0.33277777, 0.166666666, 0.085, 0.0516666,
            0.02388888, 0.03, 0.1161111, 0.305, 0.3283333]
colors = ['#4A90E2', '#BDC3C7', '#FFA726', 'gray']
# 创建两个子图，共享 x 轴
fig, ax = plt.subplots(1, 2, figsize=(9, 4))

# 左边：Utility 折线图
ax[0].plot(x, he, marker='o', color=colors[0], linestyle='-', linewidth=2, label='HE', zorder=100)
ax[0].plot(x, he_plus, marker='^', color=colors[1], linestyle='--', linewidth=2, label='HEP', zorder=100)
# ax1.set_title('Utility (HE vs HE Plus) and Unsafe Rate vs X')

# 右边：Unsafe Rate 折线图
ax[0].plot(x, unsafety, marker='s', color=colors[2], linestyle='-.', linewidth=2, label='Unsafe Rate', zorder=100)
ax[0].set_xlabel(r'$\beta$')
ax[0].set_ylabel('Performance', fontsize=20)
ax[0].grid(linestyle='--', alpha=0.7, zorder=0)
# ax.axhline(y=0.07, color='black', linestyle='--', linewidth=1.5, label='Unsafe Bound', alpha=0.9)
ax[0].plot([x[0], x[-1]], [0.07, 0.07],
         color='gray', linestyle='dashdot', linewidth=2, label='Unsafe Bound', zorder=0)
ax[0].set_ylim([0., 0.8])
ax[0].set_xticks([0.04, 0.1, 0.16], ['0.04', '0.10', '0.16'])
# ax[0].legend(loc='upper right', prop={'size': 12})


# 左边：Utility 折线图
x = [125, 156, 188, 218, 250, 281, 312, 375, 406]

he = [0.549, 0.628, 0.622, 0.646, 0.659, 0.646, 0.646, 0.646, 0.659]
he_plus = [0.488, 0.555, 0.543, 0.598, 0.604, 0.598, 0.591, 0.591, 0.598]
unsafety = [0.320085, 0.021287, 0.012616, 0.018390, 0.004607, 0.040546, 0.068099, 0.100985, 0.129198]
ax[1].plot(x, he, marker='o', color=colors[0], linestyle='-', linewidth=2, label='HE', zorder=100)
ax[1].plot(x, he_plus, marker='^', color=colors[1], linestyle='--', linewidth=2, label='HEP', zorder=100)
# ax1.set_title('Utility (HE vs HE Plus) and Unsafe Rate vs X')

# 右边：Unsafe Rate 折线图
ax[1].plot(x, unsafety, marker='s', color=colors[2], linestyle='-.', linewidth=2, label='Unsafe Rate', zorder=100)
ax[1].set_xlabel(r'$T$')
# ax[1].set_ylabel('Performance', fontsize=20)
ax[1].grid(linestyle='--', alpha=0.7, zorder=0)
# ax.axhline(y=0.07, color='black', linestyle='--', linewidth=1.5, label='Unsafe Bound', alpha=0.9)
ax[1].plot([x[0], x[-1]], [0.07, 0.07],
         color='gray', linestyle='dashdot', linewidth=2, label='Unsafe Bound', zorder=0)
ax[1].set_ylim([0., 0.8])
ax[1].set_xticks([125, 225, 325, 425], ['125', '225', '325', '425'])
# ax[1].legend(loc='upper right', prop={'size': 12})

methods = ['HE', 'HEP', 'Unsafe Rate', 'Unsafe Bound']
handles1, labels1 = ax[0].get_legend_handles_labels()
fig.legend(handles=handles1,
           labels=methods,
           loc='upper center',
           bbox_to_anchor=(0.5, 1.07),
           ncol=len(methods),
           prop={'size': 12})

# 自动调整布局
plt.tight_layout()
fig.savefig('beta.png', bbox_inches='tight')
fig.savefig('beta.pdf', bbox_inches='tight')

