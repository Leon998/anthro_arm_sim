import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# 设置随机种子
np.random.seed(0)

# 生成数据，假设有三个高斯分布中心，彼此较近
n_samples = 50
mean1 = [11, 8]
cov1 = [[1, 0.5], [0.5, 1]]

mean2 = [13, 12]
cov2 = [[1, -0.1], [-0.1, 1]]

mean3 = [16, 10]
cov3 = [[1, -0.5], [-0.5, 1]]

# 从三个不同的高斯分布中生成样本
data1 = np.random.multivariate_normal(mean1, cov1, n_samples)
data2 = np.random.multivariate_normal(mean2, cov2, n_samples)
data3 = np.random.multivariate_normal(mean3, cov3, n_samples)
data = np.vstack([data1, data2, data3])

# 拟合高斯混合模型
gmm = GaussianMixture(n_components=3, covariance_type='full')
gmm.fit(data)
labels = gmm.predict(data)

# 绘制数据点和拟合的高斯混合模型
plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], c='black', s=10, label='Data Points')  # 所有数据点都为黑色
plt.title("2D Gaussian Mixture Model with Three Adjacent Distributions")
plt.xlabel("Target position")
plt.ylabel("Feature")

# 绘制GMM的椭圆轮廓
def plot_gmm_ellipse(gmm, ax, colors=['red', 'green', 'blue']):
    for i, (pos, covar) in enumerate(zip(gmm.means_, gmm.covariances_)):
        eigenvalues, eigenvectors = np.linalg.eigh(covar)
        angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])
        angle = np.degrees(angle)
        width, height = 2 * np.sqrt(eigenvalues)  # 2 std devs
        ellip = plt.matplotlib.patches.Ellipse(
            pos, 2*width, 2*height, angle=angle, color=colors[i], alpha=0.3
        )
        ax.add_patch(ellip)

plot_gmm_ellipse(gmm, plt.gca())

# 隐藏x和y轴的坐标值
plt.gca().set_xticks([])
plt.gca().set_yticks([])

# plt.legend()
plt.show()
