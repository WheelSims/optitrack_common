# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:00:05 2024

@author: User
"""

import optitrack as ot
import numpy as np
import kineticstoolkit.lab as ktk
import matplotlib.pyplot as plt
import time

# Initialize the wheel center position
wheel_center_position = np.array([-0.183948, -0.370139, -0.618462, 1])
# Initialize the wheel radius
wheel_radius = 0.28

def detect_events(ts):
    
    # Calculate velocity and acceleration from positions using a Savitzky-Golay filter
    ts.data["Velocity"] = ktk.filters.savgol(ts, window_length=11, poly_order=2, deriv=1).data["Position"]
    ts.data["Acceleration"] = ktk.filters.savgol(ts, window_length=15, poly_order=2, deriv=2).data["Position"]
    
    pushes_indices = []  # List to store indices of detected pushes
    recoveries_indices = []  # List to store indices of detected recoveries
    thrusts_indices = []  # List to store indices of detected thrusts

    last_event = None  # Track the last detected event

    for i in range(1, len(ts.data["Position"])):
        distance = np.linalg.norm(ts.data['Position'][i][:3] - wheel_center_position[:3])
        velocity_x = ts.data["Velocity"][i][0]  # Velocity in the x direction
        acceleration_x = ts.data["Acceleration"][i][0]  # Acceleration in the x direction
        
    
    # Check if the distance is within an interval of ±35% of the radius
        if (wheel_radius - (wheel_radius * 0.40)) <= distance <= (wheel_radius + (wheel_radius * 0.40)):
            if velocity_x > 0:  
                # Check for a negative peak in acceleration
                if acceleration_x < -2:  # Ensure it exceeds the threshold
                    if ts.data["Acceleration"][i-1][0] > acceleration_x < ts.data["Acceleration"][i+1][0]:
                        if last_event != "push":
                            pushes_indices.append(i)
                            last_event = "push"
            elif velocity_x < 0:  
                # Check for a positive peak in acceleration
                if acceleration_x > 2:  # Ensure it exceeds the threshold
                    if ts.data["Acceleration"][i-1][0] < acceleration_x > ts.data["Acceleration"][i+1][0]:
                        if last_event != "recovery":
                            recoveries_indices.append(i)
                            last_event = "recovery"
    
   
    # Add events for detected pushes
    for idx in pushes_indices:
        ts = ts.add_event(ts.time[idx], "push")
    
    # Add events for detected recoveries
    for idx in recoveries_indices:
        ts = ts.add_event(ts.time[idx], "recovery")
    
    print(pushes_indices)
    print(recoveries_indices)
    ts.plot(['Position'])
    return ts

def calculate_push_angle(position1, position2):
    # Extract the direction vectors from the positions
    direction1 = position1[:3] / np.linalg.norm(position1[:3])
    direction2 = position2[:3] / np.linalg.norm(position2[:3])

    # Calculate the dot product between the direction vectors
    dot_product = np.dot(direction1, direction2)

    # Ensure dot product is within range [-1, 1] to avoid arccosine domain errors
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate the push angle in radians
    push_angle_radians = np.arccos(dot_product)

    # Convert the angle to degrees
    push_angle_degrees = np.degrees(push_angle_radians)

    return push_angle_degrees



def main():
    # Start the OptiTrack connection
    ot.start()
    # Create an empty TimeSeries
    ts = ktk.TimeSeries()
    time.sleep(8)
    
    try:
        while True:
            # Fetch the position data from OptiTrack
            new_ts = ot.fetch()
            
            if "Position" in new_ts.data:
                # Append the new data to the existing TimeSeries
                if "Position" not in ts.data:
                    ts.data["Position"] = new_ts.data["Position"]
                    ts.time = new_ts.time
                else:
                    ts.data["Position"] = np.vstack((ts.data["Position"], new_ts.data["Position"]))
                    ts.time = np.concatenate((ts.time, new_ts.time))

                # Apply resampling to the accumulated TimeSeries
                ts_resampled = ts.resample(60.0)

                # Call the detect_events function with the updated TimeSeries
                ts_resampled = detect_events(ts_resampled)

                # Ask the user if they want to stop the measurement
                user_input = input("Press 'q' to stop the measurement: ")
                if user_input.lower() == "q":
                    break

    except KeyboardInterrupt:
        pass

    finally:
        # Stop the OptiTrack connection
        ot.stop()

    # Plot the movement between the first and second push events
    #plot_pushes(ts, 'push', 'push', occurrence1=0, occurrence2=1, inclusive=False)

if __name__ == "__main__":
    main()

