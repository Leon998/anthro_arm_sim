import matplotlib.pyplot as plt
import numpy as np

# 数据准备
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 创建图形
fig, ax = plt.subplots()

# 绘制曲线
ax.plot(x, y1, label='Sine Wave', color='blue')
ax.plot(x, y2, label='Cosine Wave', color='red')

# 添加图例
# 将图例放在顶部，设置 bbox_to_anchor 和 loc
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()
