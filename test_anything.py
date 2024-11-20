import numpy as np

def interpolate_trajectories(traj1, traj2, target_length=None):
    """
    Interpolates two trajectories along the first dimension to make their lengths equal.
    
    Parameters:
        traj1 (np.ndarray): First trajectory of shape (N1, 3).
        traj2 (np.ndarray): Second trajectory of shape (N2, 3).
        target_length (int, optional): The target length for interpolation. If None, 
                                       it uses the length of the longer trajectory.
    
    Returns:
        np.ndarray, np.ndarray: Two interpolated trajectories of equal length.
    """
    # Get the lengths of the two trajectories
    len1, len2 = traj1.shape[0], traj2.shape[0]
    
    # Determine the target length
    if target_length is None:
        target_length = max(len1, len2)
    
    # Create the interpolation indices
    target_indices = np.linspace(0, 1, target_length)
    indices1 = np.linspace(0, 1, len1)
    indices2 = np.linspace(0, 1, len2)
    
    # Interpolate each trajectory
    traj1_interp = np.array([np.interp(target_indices, indices1, traj1[:, i]) for i in range(3)]).T
    traj2_interp = np.array([np.interp(target_indices, indices2, traj2[:, i]) for i in range(3)]).T
    
    return traj1_interp, traj2_interp

# Example trajectories
traj1 = np.random.rand(156, 3)  # Trajectory 1 of shape (156, 3)
traj2 = np.random.rand(196, 3)  # Trajectory 2 of shape (196, 3)

# Interpolate to make them of equal length
traj1_interp, traj2_interp = interpolate_trajectories(traj1, traj2)

print("Interpolated Trajectory 1 shape:", traj1_interp.shape)
print("Interpolated Trajectory 2 shape:", traj2_interp.shape)
