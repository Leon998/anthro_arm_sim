import pandas as pd
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import os


def get_transformed_trajectory(file_name, base_position, cut_data=False, orientation=False, tg_based = False):
    """
    Transform keypoints' trajectory into base coordinate

    tg_based: 转换到target坐标系下
    """
    base_cols, eb_cols, wr_cols, ee_cols, target_cols = get_col_index(file_name)
    T_w2base = read_data(file_name, base_cols, cut_data)
    T_w2eb = read_data(file_name, eb_cols, cut_data)
    T_w2wr = read_data(file_name, wr_cols, cut_data)
    T_w2ee = read_data(file_name, ee_cols, cut_data)
    T_w2tg = read_data(file_name, target_cols, cut_data)
    if not tg_based:
        qs_base2eb, ts_base2eb = keypoint_transform(T_w2base, T_w2eb, base_position)
        qs_base2wr, ts_base2wr = keypoint_transform(T_w2base, T_w2wr, base_position)
        qs_base2ee, ts_base2ee = keypoint_transform(T_w2base, T_w2ee, base_position)
        if orientation:
            return qs_base2eb, ts_base2eb, qs_base2wr, ts_base2wr, qs_base2ee, ts_base2ee
        else:
            return ts_base2eb, ts_base2wr, ts_base2ee
    elif tg_based:
        qs_tg2eb, ts_tg2eb = keypoint_transform(T_w2tg, T_w2eb, 0)
        qs_tg2wr, ts_tg2wr = keypoint_transform(T_w2tg, T_w2wr, 0)
        qs_tg2ee, ts_tg2ee = keypoint_transform(T_w2tg, T_w2ee, 0)
        if orientation:
            return qs_tg2eb, ts_tg2eb, qs_tg2wr, ts_tg2wr, qs_tg2ee, ts_tg2ee
        else:
            return ts_tg2eb, ts_tg2wr, ts_tg2ee

def get_ee_rotation(ts_base2wr, qs_base2ee, ts_base2ee):
    
    pass


def get_col_index(file_name):
    data_cols = [i for i in range(2, 37)]
    col_name_list = np.array(pd.read_csv(file_name, usecols=data_cols, skiprows=2, nrows=1)).reshape(-1)
    # 找出每个字符串首次出现的索引
    unique_elements, indices = np.unique(col_name_list, return_index=True)
    # 按索引排序，以保持原始数组的顺序
    sorted_indices = np.sort(indices)
    # 结果
    col_name_dict = {col_name_list[i]: i+2 for i in sorted_indices}
    target_cols = [i for i in range(col_name_dict['target'], col_name_dict['target']+7)]
    base_cols = [i for i in range(col_name_dict['base'], col_name_dict['base']+7)]
    eb_cols = [i for i in range(col_name_dict['eb'], col_name_dict['eb']+7)]
    wr_cols = [i for i in range(col_name_dict['wr'], col_name_dict['wr']+7)]
    for key in col_name_dict.keys():
        if key not in ['target', 'base', 'eb', 'wr']:
            ee_key = key
            break
    ee_cols = [i for i in range(col_name_dict[ee_key], col_name_dict[ee_key]+7)]
    return base_cols, eb_cols, wr_cols, ee_cols, target_cols

def read_data(file_name, data_cols, cut_data=False):
    df_raw = pd.read_csv(file_name, usecols=data_cols, skiprows=6)
    if cut_data:
        T_w2data = np.array(df_raw)[cut_data[0]:cut_data[1]]
    else:
        T_w2data = np.array(df_raw)
    return T_w2data
    
def keypoint_transform(T_w2base, T_w2k, base_position):
    """
    Transform Tw2k into Tbase2k (in all time)
    """
    qs_w2base = T_w2base[:, :4]
    ts_w2base = T_w2base[:, 4:7]
    qs_w2k = T_w2k[:, :4]
    ts_w2k = T_w2k[:, 4:7]
    num_frame = T_w2base.shape[0]
    qs_base2k = np.zeros((1, 4))
    ts_base2k = np.zeros((1, 3))
    for i in range(num_frame):
        # base
        q_w2base = qs_w2base[i, :]
        t_w2base = ts_w2base[i, :]
        # keypoint
        q_w2k = qs_w2k[i, :]
        t_w2k = ts_w2k[i, :]
        # Transformation
        q_base2k, t_base2k, _ = coordinate_transform(q_w2k, t_w2k, q_w2base, t_w2base)
        # Concatenateis=0)
        qs_base2k = np.concatenate((qs_base2k, q_base2k.reshape(1, 4)), axis=0)
        ts_base2k = np.concatenate((ts_base2k, t_base2k.reshape(1, 3)), axis=0)
    qs_base2k = qs_base2k[1:, :]
    ts_base2k = ts_base2k[1:, :] + base_position
    return qs_base2k, ts_base2k

def coordinate_transform(q_w2k, t_w2k, q_w2base, t_w2base):
    """
    Transform Tw2k into Tbase2k (in unit time)
    """
    r_w2base = R.from_quat(q_w2base).as_matrix()  # Get object rotation matrix from quaternion
    r_w2k = R.from_quat(q_w2k).as_matrix()  # Get hand rotation matrix from quaternion
    # Transform
    r_base2k = (np.linalg.inv(r_w2base)).dot(r_w2k)
    q_base2k = R.from_matrix(r_base2k).as_quat()
    t_base2k = np.linalg.inv(r_w2base).dot(t_w2k + (-t_w2base))
    tf_base2k = np.concatenate((q_base2k, t_base2k), axis=0)
    return q_base2k, t_base2k, tf_base2k

def down_sample(data, interval=1):
    return data[::interval, :]

def calculate_speed_3d(displacement_matrix, time_interval=0.01):
    displacement_diff = np.diff(displacement_matrix, axis=0)
    speed_matrix = displacement_diff / time_interval
    speed_matrix = np.vstack((speed_matrix, [0, 0, 0]))
    return speed_matrix

def calculate_angular_speed(quaternions, time_interval=0.01):
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
    
    # 末尾时刻速度为零    
    angular_velocities.append([0, 0, 0])
    return np.array(angular_velocities)

def compute_point_dist(A, B):
    for (a, b) in zip(A, B):
        d = math.sqrt(math.pow(a[0]-b[0],2)
                      +math.pow(a[1]-b[1],2)
                      +math.pow(a[2]-b[2],2))
        print(d)

def hand_init_bias(y, bias):
    for i in range(len(y.T)):
        y[2,i] = y[2,i] + bias  # bias on z axis
    return y


# if __name__ == "__main__":
