import pandas as pd
import numpy as np
import math


def read_data(file_name):
    """
    Extract rotation (quaternion type) and translation, and transform into numpy array
    Parameters
    ----------
    file_name : name of csv file
    Returns
    ----------
    Q_wh, T_wh, Q_wo, T_wo, num_frame : quaternion and translation, stack vertically by time
        In shape of (num_frame, 4), (num_frame, 3), (num_frame, 4), (num_frame, 3)
    """
    df_raw = pd.read_csv(file_name, usecols=[2, 3, 4, 5, 6, 7, 8], skiprows=6)
    data_raw = np.array(df_raw)
    num_frame = data_raw.shape[0]
    Q_wh = data_raw[:, :4]  # X,Y,Z,W
    T_wh = data_raw[:, 4:7]  # X,Y,Z
    return Q_wh, T_wh

def get_point_dist(y):
    for i in range(len(y.T)):
        if i > 0:
            d = math.sqrt(math.pow((y[0,i]-y[0,i-1]),2)
                          +math.pow((y[1,i]-y[1,i-1]),2)
                          +math.pow((y[2,i]-y[2,i-1]),2))
            print(d)

def hand_init_bias(y, bias):
    for i in range(len(y.T)):
        y[2,i] = y[2,i] + bias  # bias on z axis
    return y