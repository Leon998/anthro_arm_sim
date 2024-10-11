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
    X_train : array-like of shape (n_samples, 6) (6 = 3 for pca dim + 3 for target_positions)
    target_position : conditioned target position, shape (1, 3)

    Returns
    ----------
    sampled_position : mean value of GMM under the conditioned target_position, shape (1, 3)
    """
    # GMR
    random_state = np.random.RandomState(0)
    n_components = 5
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


physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # 先不渲染
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setGravity(0,0,0)
planeId = p.loadURDF("plane.urdf")
robot = ROBOT("arm_bottle1_demo")
kpt_ee = ROBOT.keypoint(robot, robot.ee_index)

# Rendering
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=-135,
                                 cameraPitch=-36, cameraTargetPosition=[0.2,0,0.5])

base_position = np.array(robot.startPos) + np.array([-0.05, 0.1, -0.15])  # 肩宽、肩厚、肩高补偿
main_path = 'trajectories/mocap_csv/710/bottle/'
file_path = main_path + "source/"
files = os.listdir(file_path)
file_list = [i for i in range(0,54)]
segment_file = np.loadtxt(main_path + "segment.txt")
# point docker
ts_tg2eb = np.empty((0, 3))
ts_tg2wr = np.empty((0, 3))
ts_tg2ee = np.empty((0, 3))
logqs_tg2ee = np.empty((0, 3))
ts_base2tg = np.empty((0, 3))
qs_base2tg = np.empty((0, 4))

test_index = 5  # 测试数据索引
# 提取所有示教数据在一时刻的关键点位置
for file_index in file_list:
    file_name = file_path + files[file_index]
    segment_index = int(segment_file[file_index])
    frames = [-2, -1]
    _, t_tg2eb, _, t_tg2wr, q_tg2ee, t_tg2ee, _, _ = get_transformed_trajectory(file_name, 
                                                                                base_position,
                                                                                cut_data=frames,
                                                                                orientation=True,
                                                                                tg_based=True)  # 机器人坐标系下的所有点坐标
    _, t_base2eb, _, t_base2wr, q_base2ee, t_base2ee, q_base2tg, t_base2tg = get_transformed_trajectory(file_name, 
                                                                                                        base_position,
                                                                                                        cut_data=frames,
                                                                                                        orientation=True)  # target坐标系下的所有点坐标
    if file_index < 27:
        p.addUserDebugPoints(t_tg2ee, [[1, 0, 0]], 5)
        p.addUserDebugPoints(t_tg2wr, [[0, 1, 0]], 5)
        p.addUserDebugPoints(t_tg2eb, [[0, 0, 1]], 5)
    else:
        p.addUserDebugPoints(t_tg2ee, [[1, 0.5, 0.5]], 5)
        p.addUserDebugPoints(t_tg2wr, [[0.5, 1, 0.5]], 5)
        p.addUserDebugPoints(t_tg2eb, [[0.5, 0.5, 1]], 5)
    # 先实现四元数到欧氏空间转换
    log_q = quaternion2euler(q_tg2ee.reshape(-1))
    ###############################
    ts_tg2eb = np.vstack((ts_tg2eb, t_tg2eb))
    ts_tg2wr = np.vstack((ts_tg2wr, t_tg2wr))
    ts_tg2ee = np.vstack((ts_tg2ee, t_tg2ee))
    logqs_tg2ee = np.vstack((logqs_tg2ee, log_q))
    ts_base2tg = np.vstack((ts_base2tg, t_base2tg))
    qs_base2tg = np.vstack((qs_base2tg, q_base2tg))
    if file_index == test_index:
        q_base2ee_test = q_base2ee.reshape(-1)
        t_base2ee_test = t_base2ee.reshape(-1)

####################################### PCA ######################################################
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

####################### Constrain Learning #########################
# 经PCA后发现，EE为p2p约束，EE orientation在转换到欧氏空间后为p2l约束，因此可以直接采样出约束点
# 现直接定义约束
cons_t_tg2ee = pca_ee.mean_
cons_q_tg2ee = euler2quaternion(pca_logq_ee.mean_)
print("Constrains in tg: ", cons_q_tg2ee, cons_t_tg2ee)

# 然后对隐式约束进行联合PCA：PCA on eb and wr，定义为IC（implicit constrain，隐式约束）
ts_tg2IC = np.hstack((ts_tg2eb, ts_tg2wr))
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

################################################### GMM + Opt ###########################################
# 对降维后的子空间做GMM+GMR
X_train = np.hstack((X, ts_base2tg))
print("GMR train set shape: ", X_train.shape)
# 根据测试时的目标位置，从GMR中采样出子空间的均值
t_base2tg = ts_base2tg[test_index]
q_base2tg = qs_base2tg[test_index]
mu = GMR_sample(X_train, t_base2tg)
print("Sampled subspace_mean: ", mu)


# ### TODO Optimization in pca space ###
# Constrain
cons_q_base2ee, cons_t_base2ee = rgbody_transform(q_base2tg, t_base2tg, cons_q_tg2ee, cons_t_tg2ee)
print("target position in base:", t_base2tg)
print("Real ee constrains in base: ", q_base2ee_test, t_base2ee_test)
print("Learned ee constrains in base: ", cons_q_base2ee, cons_t_base2ee)
p.addUserDebugPoints([t_base2tg], [[0, 0, 0]], 5)
p.addUserDebugPoints([t_base2ee_test], [[0, 1, 0]], 5)  # 真实末端位置
p.addUserDebugPoints([cons_t_base2ee], [[1, 1, 1]], 5)  # 学到的末端约束位置


kpt_list = [robot.elbow_index, robot.wrist_index]
cons_dict = {robot.ee_index:t_base2ee_test}
ee_ori = q_base2ee_test
q_init = [0. for i in range(robot.dof)]
q_star = robot.subspace_opt_position(pca, kpt_list, cons_dict, ee_ori, q_init, q_base2tg, t_base2tg, mu)
robot.FK(q_star)


# # 直接逆变换到笛卡尔空间得到关键点位置，老方法。
# sampled_goal = pca.inverse_transform(sampled_position.reshape(1, -1)).reshape(-1)
# goal_tg2eb, goal_tg2wr = sampled_goal[:3], sampled_goal[3:]
# print(goal_tg2eb, goal_tg2wr)




while True:
    p.stepSimulation()
    time.sleep(1./240.)