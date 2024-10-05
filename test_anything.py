import numpy as np

kpt_positions = []
current_pos = np.array([1, 2, 3])
kpt_positions = np.hstack((kpt_positions, current_pos)).reshape(-1)
current_pos = np.array([4, 5, 6])
kpt_positions = np.hstack((kpt_positions, current_pos)).reshape(-1)
print(kpt_positions)