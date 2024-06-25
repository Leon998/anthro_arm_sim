import pandas as pd
import numpy as np
import math
from scipy.spatial.transform import Rotation as R


def read_data(file_name):
    df_raw = pd.read_csv(file_name, usecols=[2, 3, 4, 5, 6, 7, 8], skiprows=6)
    data_raw = np.array(df_raw)
    num_frame = data_raw.shape[0]
    Q_wbase = data_raw[:, :4]  # X,Y,Z,W
    T_wbase = data_raw[:, 4:7]  # X,Y,Z
    return Q_wbase, T_wbase

def get_transformed_position(file_name, base_position, down_sample=2):
    """
    Transform keypoints' position into base coordinate
    """
    T_w2base = np.array(pd.read_csv(file_name, usecols=[2, 3, 4, 5, 6, 7, 8], skiprows=6))
    T_w2eb = np.array(pd.read_csv(file_name, usecols=[9, 10, 11, 12, 13, 14, 15], skiprows=6))
    T_w2wr = np.array(pd.read_csv(file_name, usecols=[16, 17, 18, 19, 20, 21, 22], skiprows=6))
    T_w2ee = np.array(pd.read_csv(file_name, usecols=[23, 24, 25, 26, 27, 28, 29], skiprows=6))
    ts_base2eb = keypoint_transform(T_w2base, T_w2eb)[::down_sample, :] + base_position
    ts_base2wr = keypoint_transform(T_w2base, T_w2wr)[::down_sample, :] + base_position
    ts_base2ee = keypoint_transform(T_w2base, T_w2ee)[::down_sample, :] + base_position
    return ts_base2eb, ts_base2wr, ts_base2ee
    
def keypoint_transform(T_w2base, T_w2k):
    qs_w2base = T_w2base[:, :4]
    ts_w2base = T_w2base[:, 4:7]
    qs_w2k = T_w2k[:, :4]
    ts_w2k = T_w2k[:, 4:7]
    num_frame = T_w2base.shape[0]
    ts_base2k = np.zeros((1, 3))
    for i in range(num_frame):
        # base
        q_w2base = qs_w2base[i, :]
        t_w2base = ts_w2base[i, :]
        # keypoint
        q_w2k = qs_w2k[i, :]
        t_w2k = ts_w2k[i, :]
        # Transformation
        _, t_base2k, _ = coordinate_transform(q_w2k, t_w2k, q_w2base, t_w2base)
        # Concatenateis=0)
        ts_base2k = np.concatenate((ts_base2k, t_base2k.reshape(1, 3)), axis=0)
    ts_base2k = ts_base2k[1:, :]
    return ts_base2k

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


if __name__ == "__main__":
    file_name = 'trajectories/mocap_csv/622/622_pour_000.csv'
    T_w2base = np.array(pd.read_csv(file_name, usecols=[2, 3, 4, 5, 6, 7, 8], skiprows=6))
    T_w2eb = np.array(pd.read_csv(file_name, usecols=[9, 10, 11, 12, 13, 14, 15], skiprows=6))
    ts_w2base = T_w2base[::2, 4:7]
    ts_w2eb = T_w2eb[::2, 4:7]
    compute_point_dist(ts_w2base, ts_w2eb)
