import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_to_angular_velocity(quaternions, time_interval):
    """
    计算每个时间点的角速度。
    
    参数：
    - quaternions: 形状为 (N, 4) 的四元数序列，N 是时间点数量。
    - time_interval: 固定的时间间隔。
    
    返回：
    - angular_velocities: 形状为 (N, 3) 的角速度序列，每个时刻对应的角速度向量。
    """
    angular_velocities = []
    
    # 遍历每一对相邻的四元数
    for i in range(1, len(quaternions)):
        # 当前四元数和前一个四元数
        q1 = R.from_quat(quaternions[i-1])
        q2 = R.from_quat(quaternions[i])
        
        # 计算相对旋转: q_rel = q1.inverse() * q2
        q_rel = q1.inv() * q2
        
        # 提取旋转向量
        rotvec = q_rel.as_rotvec()  # 旋转向量，单位是弧度
        
        # 角速度: ω = Δθ / Δt
        angular_velocity = rotvec / time_interval
        
        angular_velocities.append(angular_velocity)
    angular_velocities.append([0, 0, 0])
    return np.array(angular_velocities)

# 示例四元数序列 (N, 4)
quaternions = [
    [1, 0, 0, 0],  # 四元数 q1
    [0.707, 0.707, 0, 0],  # 四元数 q2
    [0, 1, 0, 0]  # 四元数 q3
]
time_interval = 1.0  # 固定的时间间隔

# 计算角速度
angular_velocities = quaternion_to_angular_velocity(quaternions, time_interval)
print(angular_velocities)
