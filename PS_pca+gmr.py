import pybullet as p
import time
import pybullet_data
import math
import os, sys
sys.path.append(os.getcwd())
from utils import *
from Robot_arm import ROBOT
from sklearn import decomposition
from sklearn.mixture import BayesianGaussianMixture
from gmr import GMM, kmeansplusplus_initialization, covariance_initialization
import matplotlib.pyplot as plt


def GMR_sample(X_train, target_position):
    """
    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features + 3) (3 for target_positions)
    target_position : conditioned target position, shape (1, 3)

    Returns
    ----------
    sampled_position : mean value of GMM under the conditioned target_position, shape (1, n_features)
    """
    # GMR
    random_state = np.random.RandomState(0)
    n_components = 4
    initial_means = kmeansplusplus_initialization(X_train, n_components, random_state)
    initial_covs = covariance_initialization(X_train, n_components)
    bgmm = BayesianGaussianMixture(
        n_components=n_components, max_iter=500,
        random_state=random_state).fit(X_train)
    gmm = GMM(n_components=n_components, priors=bgmm.weights_, means=bgmm.means_,
              covariances=bgmm.covariances_, random_state=random_state)
    cgmm = gmm.condition([-3, -2, -1], target_position)
    sampled_position = cgmm.to_mvn().mean
    return sampled_position


arm = "arm_sx"  # 用哪个arm
tool = "pry2"  # 用哪个工具
train_subject = 'sx'  # 用哪些示教数据

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # 先不渲染
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setGravity(0,0,0)
planeId = p.loadURDF("plane.urdf")
robot = ROBOT(arm, tool)
kpt_ee = ROBOT.keypoint(robot, robot.ee_index)

# Rendering
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=-135,
                                 cameraPitch=-36, cameraTargetPosition=[0.2,0,0.5])

#################################### Loading lfd data ########################################
tool_class = tool[:-1]
data_path = 'trajectories/mocap_csv/lfd/'+ tool_class +'/'
base_bias = robot.base_bias  # 肩宽、肩厚、肩高补偿
if train_subject == 'all':
    files = get_all_file_paths(data_path)
else:
    files = get_all_file_paths(data_path + train_subject + '/')
frames = [0, 1]

# ========================================================================================== #
######################################## Training ############################################
# ========================================================================================== #

# ================= Config ========================== #
train_list = [i for i in range(0, len(files))]
# train_list = [4, 5, 7, 8, 13, 14, 16, 17]
print(train_list)
test_index = 15  # 测试文件索引
PCA = False
# =================================================== #

# point docker
ts_tg2eb = np.empty((0, 3))
ts_tg2wr = np.empty((0, 3))
ts_tg2ee = np.empty((0, 3))
logqs_tg2ee = np.empty((0, 3))
ts_base2tg = np.empty((0, 3))
qs_base2tg = np.empty((0, 4))

# 提取所有示教数据在一时刻的关键点位置
for file_index in train_list:
    file_name = files[file_index]
    _, t_tg2eb, _, t_tg2wr, q_tg2ee, t_tg2ee, _, _ = get_transformed_trajectory(file_name, 
                                                                                base_bias,
                                                                                cut_data=frames,
                                                                                orientation=True,
                                                                                tg_based=True)  # target坐标系下的所有点坐标
    _, t_base2eb, _, t_base2wr, q_base2ee, t_base2ee, q_base2tg, t_base2tg = get_transformed_trajectory(file_name, 
                                                                                                        base_bias,
                                                                                                        cut_data=frames,
                                                                                                        orientation=True)  # 机器人坐标系下的所有点坐标
    p.addUserDebugPoints(t_tg2ee, [[1, 0, 0]], 5)
    p.addUserDebugPoints(t_tg2wr, [[0, 1, 0]], 5)
    p.addUserDebugPoints(t_tg2eb, [[0, 0, 1]], 5)
    # 先实现四元数到欧氏空间转换
    log_q = quaternion2euler(q_tg2ee.reshape(-1))
    ts_tg2eb = np.vstack((ts_tg2eb, t_tg2eb))
    ts_tg2wr = np.vstack((ts_tg2wr, t_tg2wr))
    ts_tg2ee = np.vstack((ts_tg2ee, t_tg2ee))
    logqs_tg2ee = np.vstack((logqs_tg2ee, log_q))
    ts_base2tg = np.vstack((ts_base2tg, t_base2tg))
    qs_base2tg = np.vstack((qs_base2tg, q_base2tg))

############### PCA ###################
# 需要先对所有关键点做PCA，方差小于某个阈值，那么说明是强约束，否则拿去做GMM
pca_eb = decomposition.PCA(n_components=3)
pca_wr = decomposition.PCA(n_components=3)
pca_ee = decomposition.PCA(n_components=3)
pca_logq_ee = decomposition.PCA(n_components=3)

pca_eb.fit(ts_tg2eb)
print("Elbow's explained variance: ", pca_eb.explained_variance_)
pca_wr.fit(ts_tg2wr)
print("Wrist's explained variance: ", pca_wr.explained_variance_)
pca_ee.fit(ts_tg2ee)
print("EE's explained variance: ", pca_ee.explained_variance_)
pca_logq_ee.fit(logqs_tg2ee)
print("EE orientation's explained variance: ", pca_logq_ee.explained_variance_)

########## Constrain Learning ###########
# 经PCA后发现，EE为p2p约束，EE orientation在转换到欧氏空间后为p2l约束，因此可以直接采样出约束点
# 对于笛卡尔约束，直接GMR
ts_tg2C = np.hstack((ts_tg2ee, logqs_tg2ee))
C_train = np.hstack((ts_tg2C, ts_base2tg))

# 然后对隐式约束进行联合PCA：PCA on eb and wr，定义为IC（implicit constrain，隐式约束）
ts_tg2IC = np.hstack((ts_tg2eb, ts_tg2wr))
if PCA:
    print("Original goal shape: ", ts_tg2IC.shape)
    pca = decomposition.PCA(n_components=3)
    pca.fit(ts_tg2IC)
    X = pca.transform(ts_tg2IC)
    print("Transformed component shape: ", X.shape)
    print("Explained variance: ", pca.explained_variance_)
    print("P:", pca.components_, pca.components_.shape)
    # print("Explained variance ratio: ", pca.explained_variance_ratio_)

    # # visualizing manifold
    # fig = plt.figure(1, figsize=(4, 3))
    # ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
    # X = ts_tg2IC
    # ax.scatter(X[:27, 0], X[:27, 1], X[:27, 2], c='g')
    # ax.scatter(X[27:, 0], X[27:, 1], X[27:, 2], c='b')
    # plt.show()

    # 对降维后的子空间做GMM+GMR
    X_train = np.hstack((X, ts_base2tg))
else:
    X_train = np.hstack((ts_tg2IC, ts_base2tg))

print("GMR train set shape: ", X_train.shape)


# ============================================================================================= #
########################################## Testing ##############################################
# ============================================================================================= #
# 读取测试数据
test_file = files[test_index]
print(test_file)
_, t_tg2eb_test, _, t_tg2wr_test, q_tg2ee_test, t_tg2ee_test, q_tg2tg_test, t_tg2tg_test = get_transformed_trajectory(test_file, 
                                                                                                    base_bias,
                                                                                                    cut_data=frames,
                                                                                                    orientation=True,
                                                                                                    tg_based=True)  # target坐标系下的所有点坐标
_, t_base2eb_test, _, t_base2wr_test, q_base2ee_test, t_base2ee_test, q_base2tg_test, t_base2tg_test = get_transformed_trajectory(test_file, 
                                                                                                    base_bias,
                                                                                                    cut_data=frames,
                                                                                                    orientation=True)  # base坐标系下的所有点坐标

# 根据测试时的目标位置，从GMR中采样出的均值
# Constrain
C_mu = GMR_sample(C_train, t_base2tg_test)
print("constrain mean:", C_mu)
# cons_t_tg2ee = C_mu[0:3]
# cons_q_tg2ee = euler2quaternion(C_mu[3:])
cons_t_tg2ee = pca_ee.mean_
cons_q_tg2ee = euler2quaternion(pca_logq_ee.mean_)
print("Constrains in tg: ", cons_q_tg2ee, cons_t_tg2ee)
# Feature space
mu = GMR_sample(X_train, t_base2tg_test)
print("Feature space mean: ", mu)
if not PCA:
    # GMR得到的腕、肘关节位置
    p.addUserDebugPoints([mu[3:]], [[0, 1, 0]], 10)  # tg坐标系下
    p.addUserDebugPoints([mu[:3]], [[0, 0, 1]], 10)
    _, t_base2wr = rgbody_transform(q_base2tg_test, t_base2tg_test, unit_quaternion, mu[3:])
    _, t_base2eb = rgbody_transform(q_base2tg_test, t_base2tg_test, unit_quaternion, mu[:3])
    p.addUserDebugPoints(t_base2wr, [[0, 1, 0]], 10)  # base坐标系下
    p.addUserDebugPoints(t_base2eb, [[0, 0, 1]], 10)
    # 真实的腕、肘关节位置
    p.addUserDebugPoints(t_tg2wr_test, [[0.5, 1, 0.5]], 10)  # tg坐标系下
    p.addUserDebugPoints(t_tg2eb_test, [[0.5, 0.5, 1]], 10)
    p.addUserDebugPoints(t_base2wr_test, [[0.5, 1, 0.5]], 10)  # base坐标系下
    p.addUserDebugPoints(t_base2eb_test, [[0.5, 0.5, 1]], 10)
# 测试数据的目标位置，先转换成0行的数组
q_base2tg_test = q_base2tg_test.reshape(-1)
t_base2tg_test = t_base2tg_test.reshape(-1)
# 测试数据的真实关键点位置，先转换成0行的数组
q_base2ee_test = q_base2ee_test.reshape(-1)
t_base2ee_test = t_base2ee_test.reshape(-1)

######## Optimization in pca space #######
# Constrain
cons_q_base2ee, cons_t_base2ee = rgbody_transform(q_base2tg_test, t_base2tg_test, cons_q_tg2ee, cons_t_tg2ee)
print("target position in base:", t_base2tg_test)
print("Real ee constrains in base: ", q_base2ee_test, t_base2ee_test)
print("Learned ee constrains in base: ", cons_q_base2ee, cons_t_base2ee)
p.addUserDebugPoints([t_base2tg_test], [[0, 0, 0]], 10)  # 目标位置（黑）
p.addUserDebugPoints([t_base2ee_test], [[0, 1, 0]], 10)  # 真实末端位置（绿）
p.addUserDebugPoints([cons_t_base2ee], [[1, 1, 1]], 10)  # 学到的末端约束位置（白）
print("Target position: ", t_base2tg_test)


kpt_list = [robot.elbow_index, robot.wrist_index]
cons_dict = {robot.ee_index:cons_t_base2ee}
# ee_ori = q_base2ee_test
ee_ori = cons_q_base2ee
q_init = [0. for i in range(robot.dof)]
if PCA:
    q_star = robot.feature_space_opt_position(pca, kpt_list, cons_dict, ee_ori, q_init, q_base2tg_test, t_base2tg_test, mu)
else:
    q_star = robot.cartesian_space_opt_position(kpt_list, cons_dict, ee_ori, q_init, q_base2tg_test, t_base2tg_test, mu)
robot.FK(q_star)
print(q_star)


while True:
    p.stepSimulation()
    time.sleep(1./240.)