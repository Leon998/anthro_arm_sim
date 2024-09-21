import numpy as np
from scipy.spatial.transform import Rotation as R

# 输入四元数 (代表末端朝向) 和点的位置
q = [0.707, 0.0, 0.707, 0.0]  # 示例四元数
p_end = np.array([1.0, 1.0, 1.0])  # 末端点位置
p_joint = np.array([0.0, 0.0, 0.0])  # 最后一个关节位置

# 1. 计算旋转轴向量
v = p_end - p_joint

# 2. 计算旋转角度 (向量长度)
theta = np.linalg.norm(v)

# 3. 归一化旋转轴
if theta != 0:
    v_hat = v / theta  # 单位向量
else:
    v_hat = np.zeros_like(v)

# 4. 将四元数转换为旋转矩阵
rotation_matrix = R.from_quat(q).as_matrix()

# 5. 将旋转矩阵应用到归一化后的旋转轴
rotated_v_hat = rotation_matrix @ v_hat

# 6. 计算最终的旋转向量
rotation_vector = rotated_v_hat * theta

print("旋转向量: ", rotation_vector)
