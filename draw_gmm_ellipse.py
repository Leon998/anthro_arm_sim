# 文件路径: gmm_visualization_adjusted.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

# 定义函数绘制高斯分布的椭圆
def plot_gmm(gmm, X, ax, scale_factor=3.0):
    """绘制GMM聚类结果及其高斯分布的椭圆"""
    colors = ['orange', 'orange', 'orange']
    for i, (mean, covar, color) in enumerate(zip(gmm.means_, gmm.covariances_, colors)):
        if covar.ndim == 1:
            covar = np.diag(covar)
        # 计算椭圆的轴长度和旋转角度
        v, w = np.linalg.eigh(covar)
        v = scale_factor * np.sqrt(2.0) * np.sqrt(v)  # 增大椭圆的轴长度
        u = w[0] / np.linalg.norm(w[0])  # 旋转方向
        angle = np.arctan2(u[1], u[0]) * 180.0 / np.pi
        # 添加半透明椭圆
        ellipse = Ellipse(mean, v[0], v[1], 180.0 + angle, color=color, alpha=0.3)
        ax.add_patch(ellipse)
    # 绘制数据点
    ax.scatter(X[:, 0], X[:, 1], s=10, color='black')

# 生成模拟数据
np.random.seed(1)

# 调整样本数量
n_samples1 = 50  # 红色分布
n_samples2 = 50  # 绿色分布
n_samples3 = 50  # 蓝色分布

# 红色分布（左上）
mean1 = [-1, 2]
cov1 = [[1.5, 0.8], [0.8, 1.0]]

# 绿色分布（左下）
mean2 = [-3, -1]
cov2 = [[0.5, 0.2], [0.2, 0.7]]

# 蓝色分布（右侧）
mean3 = [2, 1]
cov3 = [[0.6, -0.4], [-0.4, 0.7]]

# 生成数据点
X1 = np.random.multivariate_normal(mean1, cov1, n_samples1)
X2 = np.random.multivariate_normal(mean2, cov2, n_samples2)
X3 = np.random.multivariate_normal(mean3, cov3, n_samples3)

X = np.vstack((X1, X2, X3))

# GMM 模型拟合
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)

# 可视化结果
fig, ax = plt.subplots(figsize=(6, 4))
plot_gmm(gmm, X, ax, scale_factor=3.0)  # 调整椭圆尺度因子
# 隐藏坐标轴刻度
ax.set_xticks([])  # 隐藏x轴刻度
ax.set_yticks([])  # 隐藏y轴刻度
plt.show()
