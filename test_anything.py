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


import itertools
import time

# # 创建一个循环模式，每10次切换一次
# pattern = itertools.cycle([1] * 10 + [2] * 10)

# for i in itertools.count():  # 无限次迭代
#     current_pattern = next(pattern)  # 每次获取当前模式值
#     time.sleep(0.1)
#     if current_pattern == 1:
#         # 每10次迭代执行的语句
#         if i % 10 == 0:
#             print(f"第 {i} 次迭代：执行语句 A")
#     else:
#         # 下一组10次迭代执行的语句
#         if i % 10 == 0:
#             print(f"第 {i} 次迭代：执行语句 B")


for i in range(100):  # 假设我们有100次迭代
    print((i // 10) % 7)
    # if (i // 10) % 2 == 0:
    #     output = 1
    # else:
    #     output = 2
    # print(f"第 {i} 次迭代输出：{output}")
