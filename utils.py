import pandas as pd
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import os


base_cols = [16, 17, 18, 19, 20, 21, 22]
eb_cols= [23, 24, 25, 26, 27, 28, 29]
wr_cols= [30, 31, 32, 33, 34, 35, 36]
ee_cols=[2, 3, 4, 5, 6, 7, 8]
target_cols = [9, 10, 11, 12, 13, 14, 15]

def read_data(file_name, data_cols, cut_data=False):
    df_raw = pd.read_csv(file_name, usecols=data_cols, skiprows=6)
    if cut_data:
        T_w2data = np.array(df_raw)[cut_data[0]:cut_data[1]]
    else:
        T_w2data = np.array(df_raw)
    return T_w2data

def get_transformed_trajectory(file_name, base_position, down_sample=1, cut_data=False, orientation=False):
    """
    Transform keypoints' trajectory into base coordinate
    """
    T_w2base = read_data(file_name, base_cols, cut_data)
    T_w2eb = read_data(file_name, eb_cols, cut_data)
    T_w2wr = read_data(file_name, wr_cols, cut_data)
    T_w2ee = read_data(file_name, ee_cols, cut_data)
    qs_base2eb, ts_base2eb = keypoint_transform(T_w2base, T_w2eb, base_position, down_sample)
    qs_base2wr, ts_base2wr = keypoint_transform(T_w2base, T_w2wr, base_position, down_sample)
    qs_base2ee, ts_base2ee = keypoint_transform(T_w2base, T_w2ee, base_position, down_sample)
    if orientation:
        return qs_base2eb, ts_base2eb, qs_base2wr, ts_base2wr, qs_base2ee, ts_base2ee
    else:
        return ts_base2eb, ts_base2wr, ts_base2ee
    
def keypoint_transform(T_w2base, T_w2k, base_position, down_sample=1):
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
    qs_base2k = qs_base2k[1::down_sample, :]
    ts_base2k = ts_base2k[1::down_sample, :] + base_position
    return qs_base2k, ts_base2k

def coordinate_transform(q_w2k, t_w2k, q_w2base, t_w2base):
    r_w2base = R.from_quat(q_w2base).as_matrix()  # Get object rotation matrix from quaternion
    r_w2k = R.from_quat(q_w2k).as_matrix()  # Get hand rotation matrix from quaternion
    # Transform
    r_base2k = (np.linalg.inv(r_w2base)).dot(r_w2k)
    q_base2k = R.from_matrix(r_base2k).as_quat()
    t_base2k = np.linalg.inv(r_w2base).dot(t_w2k + (-t_w2base))
    tf_base2k = np.concatenate((q_base2k, t_base2k), axis=0)
    return q_base2k, t_base2k, tf_base2k

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
#     file_name = 'trajectories/mocap_csv/622/622_pour_000.csv'
#     T_w2base = np.array(pd.read_csv(file_name, usecols=[2, 3, 4, 5, 6, 7, 8], skiprows=6))
#     T_w2eb = np.array(pd.read_csv(file_name, usecols=[9, 10, 11, 12, 13, 14, 15], skiprows=6))
#     ts_w2base = T_w2base[::2, 4:7]
#     ts_w2eb = T_w2eb[::2, 4:7]
#     compute_point_dist(ts_w2base, ts_w2eb)
