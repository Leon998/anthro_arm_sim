import numpy as np

def euclidean_distance(point1, point2):
    """计算两个点之间的欧氏距离"""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def frechet_distance(P, Q):
    """计算两条路径 P 和 Q 之间的 Fréchet 距离"""
    n, m = len(P), len(Q)
    # 初始化缓存矩阵，存储递归结果以避免重复计算
    dp = np.full((n, m), -1.0)
    
    def recursive_calculate(i, j):
        # 如果结果已计算过，则直接返回
        if dp[i, j] > -1:
            return dp[i, j]
        
        # 计算起点距离
        if i == 0 and j == 0:
            dp[i, j] = euclidean_distance(P[0], Q[0])
        elif i == 0:
            dp[i, j] = max(recursive_calculate(0, j-1), euclidean_distance(P[0], Q[j]))
        elif j == 0:
            dp[i, j] = max(recursive_calculate(i-1, 0), euclidean_distance(P[i], Q[0]))
        else:
            # 取三个方向的最小值，并取当前点的最大距离
            dp[i, j] = max(
                min(recursive_calculate(i-1, j), recursive_calculate(i, j-1), recursive_calculate(i-1, j-1)),
                euclidean_distance(P[i], Q[j])
            )
        return dp[i, j]
    
    return recursive_calculate(n-1, m-1)

# 示例路径
P = [(0, 0), (1, 1), (2, 2)]
Q = [(0, 1), (1, 2), (2, 3)]

# 计算Fréchet距离
distance = frechet_distance(P, Q)
print(f"Fréchet Distance: {distance}")
