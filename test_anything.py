import numpy as np
from sklearn.decomposition import PCA

# # 原始数据矩阵 X，形状为 (n_samples, n_features)
# X = np.array([[2.5, 0.5], [0.5, 1.5], [2.2, 1.9], [1.9, 0.8]])
# print(X.shape)
# # 执行 PCA，降维到 1 维
# pca = PCA(n_components=1)
# pca.fit(X)

# # 获取主成分矩阵 W
# W = pca.components_
# print(W.shape)
# # 手动中心化原始数据
# X_centered = X - np.mean(X, axis=0)

# # 手动进行数据投影 (等价于 pca.transform(X))
# Z_manual = np.dot(X_centered, W.T)

# # 使用 scikit-learn 的 transform 方法
# Z_sklearn = pca.transform(X)

# # 输出手动计算结果与 sklearn 计算结果
# print("手动计算的 Z:")
# print(Z_manual)

# print("scikit-learn 计算的 Z:")
# print(Z_sklearn)

from math import pi

bounds = ([-pi, pi/2], [0, pi], [-pi/2, pi/2], [0, pi], [-pi/2, pi/2], [-pi/4, pi/4], [-pi/2, pi/2])
with open('bounds.txt', 'w') as f:
    for bound in bounds:
        f.write(f"{bound[0]},{bound[1]}\n")  # 将每个边界对写入文件

# 从 txt 文件中读取 bounds
bounds_from_file = []

with open('bounds.txt', 'r') as f:
    for line in f:
        lower, upper = line.strip().split(',')  # 按逗号分隔每一行
        # 处理 None 值
        lower = float(lower) if lower != 'None' else None
        upper = float(upper) if upper != 'None' else None
        bounds_from_file.append((lower, upper))


# 打印从文件中读取的 bounds
print(bounds_from_file)