# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:47:50 2024

@author: MOSA
"""

import numpy as np
import time

import matplotlib.pyplot as plt

# Initialize the wheel center position
wheel_center_position = np.array([0.447245, 0.658389, -0.206452, 1])
handmarker_position = np.array([0.453226, 0.800979, -0.082482, 1])
# Initialize the wheel radius
distances_marker = [
    np.linalg.norm(handmarker_position - wheel_center_position)
]
distances_marker = np.array(distances_marker)
user_arm_extended_distance = np.sqrt((distances_marker**2) - (0.055**2))
print(user_arm_extended_distance)
