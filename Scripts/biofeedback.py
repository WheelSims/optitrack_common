# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:43:24 2024

@author: User
"""

import optitrack as ot
import numpy as np
import kineticstoolkit.lab as ktk
import time


# Initialize the wheel center position
wheel_center_position = np.array([-0.281997, -0.379727, -0.599712, 1])
# Initialize the wheel radius
wheel_radius = 0.2585


def detect_events(ts: ktk.TimeSeries) -> ktk.TimeSeries:
    """
    This function detects pushes and recoveries in the time series data based on hand position relative to the wheel center and velocity.

    Parameters
    ----------
    ts: ktk.TimeSeries
        Time series data representing the motion of the hand.

    Returns
    -------
    ktk.TimeSeries
        Time series data with detected push and recovery events.
    
    """
    ts = ts.resample(60.0)
    ts.data["Velocity"] = ktk.filters.savgol(ts, window_length=11, poly_order=2, deriv=1).data["Position"]

    pushes_indices = []  # List to store indices of detected pushes
    recoveries_indices = [] # List to store indices of detected recoveries

    for i in range(1, len(ts.data["Position"])):
        distance = np.linalg.norm(ts.data['Position'][i][:3] - wheel_center_position[:3]) 
        velocity_x = ts.data["Velocity"][i][0]  # Velocity in the x direction
        prev_velocity_x = ts.data["Velocity"][i - 1][0]  # Previous velocity in the x direction

        # Check if the distance is within an interval of +- 28% of the radius
        if (wheel_radius - (wheel_radius * 0.28)) <= distance <= (wheel_radius + (wheel_radius * 0.28)):
            # Check if velocity crosses zero in x direction
            if velocity_x * prev_velocity_x <= 0:  # Sign change indicates a zero crossing
                # Check if velocity is decreasing (push) or increasing (recovery)
                if velocity_x < 0:  # Velocity is decreasing
                    pushes_indices.append(i)
                else:  # Velocity is increasing
                    recoveries_indices.append(i)

    # Add events for detected pushes
    for idx in pushes_indices:
        ts = ts.add_event(ts.time[idx], "push")

    # Add events for detected recoveries
    for idx in recoveries_indices:
        ts = ts.add_event(ts.time[idx], "recovery")

    # Plot the time series with push events
    ts.plot()

    return ts
 

# Function to execute the measurement sequence
def run_measurement() -> None:
    """
    Executes the measurement sequence by starting and stopping the OptiTrack connection and detecting events.

    Returns
    -------
    None

    """
    ot.start()
    # Wait a few seconds for data (e.g. 8 seconds)
    time.sleep(8)
    ot.stop()
    # Retrieve position data in TimeSeries format
    ts = ot.fetch()
    detect_events(ts)