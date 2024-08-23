import pybullet as p
import numpy as np
from scipy.optimize import minimize


class ROBOT:
    def __init__(self, name, dof=7):
        self.startPos = [0, 0, 1]
        self.startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.dof = dof
        self.init_joint_angles = [0. for i in range(self.dof)]
        self.robot_id = p.loadURDF("models/"+name+"/urdf/"+name+".urdf", 
                      self.startPos, self.startOrientation, useFixedBase=1)
        self.joints_indexes = [i for i in range(p.getNumJoints(self.robot_id)) if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED]
        self.shoulder_index, self.elbow_index, self.wrist_index, self.ee_index = self.joints_indexes[0], self.joints_indexes[2], self.joints_indexes[5], self.joints_indexes[6]
        for i in range(len(self.joints_indexes)):
            p.resetJointState(bodyUniqueId=self.robot_id,
                              jointIndex=i,
                              targetValue=self.init_joint_angles[i],
                              targetVelocity=0)
        self.q_init, self.dq_init, self.ddq_init = self.get_joints_states()

    def get_joints_states(self):
        joint_states = p.getJointStates(self.robot_id, range(p.getNumJoints(self.robot_id)))
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques
    
    def get_error(self, goal_pos, index, ee=True):
        current_pos = p.getLinkState(self.robot_id, index)[0]
        error = goal_pos - current_pos
        return error
    
    def FK(self, q):
        for i in range(len(self.joints_indexes)):
            p.resetJointState(bodyUniqueId=self.robot_id,
                              jointIndex=i,
                              targetValue=q[i],
                              targetVelocity=0)
        self.q, self.dq, self.ddq = self.get_joints_states()

    def get_jacobian(self):
        self.q, self.dq, self.ddq = self.get_joints_states()
        zero_vec = [0.0] * p.getNumJoints(self.robot_id)
        jac_t, jac_r = p.calculateJacobian(self.robot_id, self.ee_index, p.getLinkState(self.robot_id, self.ee_index)[2], self.q, self.dq, zero_vec)
        self.J = np.array(jac_t)

    def opt_kpt(self, sample_len, ee_traj, wrist_traj, elbow_traj):
        """
        考虑全局最优
        """
        q_init = [0. for i in range(sample_len*self.dof)]  # 长度为n×m（目标数×关节自由度数）
        def eqn(q):
            Error = []
            i, j, k= 0, 0, 0
            for g in ee_traj:
                self.FK(q[i:i+self.dof])
                Error.append(np.linalg.norm(self.get_error(g, self.ee_index)))
                i += self.dof
            for g in wrist_traj:
                self.FK(q[j:j+self.dof])
                Error.append(np.linalg.norm(self.get_error(g, self.wrist_index)))
                j += self.dof
            for g in elbow_traj:
                self.FK(q[k:k+self.dof])
                Error.append(np.linalg.norm(self.get_error(g, self.elbow_index)))
                k += self.dof
            # 以下几种误差的求法都可以，QP最好
            # Error = np.linalg.norm(np.array(Error))
            # Error = np.sum(np.array(Error))
            Error = np.dot(np.array(Error).T, np.array(Error)) + 0.001 *np.dot(q.T, q)  # QP问题
            return Error
        Q_star = minimize(eqn, q_init, method='BFGS')
        Error = Q_star.fun
        Q_star = np.array(Q_star.x).reshape([sample_len,self.dof])
        return Q_star, Error

    class keypoint:
        def __init__(self, robot, index, hasPrevPose=0, prevPose=0):
            self.robot = robot
            self.robot_id = robot.robot_id
            self.index = index
            self.hasPrevPose = hasPrevPose
            self.prevPose = prevPose
            self.traj = []
            if self.index == self.robot.ee_index:
                self.color = [1, 0, 0]
            elif self.index == self.robot.wrist_index:
                self.color = [0, 1, 0]
            else:
                self.color = [0, 0, 1]

        def save_traj(self):
            ls = p.getLinkState(self.robot_id, self.index)
            if self.index == self.robot.ee_index:
                self.traj.append(np.hstack((ls[0], ls[1])))
            else:
                self.traj.append(ls[0])
            
        def draw_traj(self):
            ls = p.getLinkState(self.robot_id, self.index)
            if (self.hasPrevPose):
                p.addUserDebugLine(self.prevPose, ls[0], self.color, 3, 15)
            self.hasPrevPose = 1
            self.prevPose = ls[0]

        def reset_pose(self):
            self.hasPrevPose = 0
            self.prevPose = 0