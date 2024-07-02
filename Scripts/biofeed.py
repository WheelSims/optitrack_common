# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:54:58 2024

@author: User
"""

import numpy as np
import kineticstoolkit.lab as ktk
import time
import pandas as pd
import matplotlib.pyplot as plt

# Initialize the wheel center position
wheel_center_position = np.array([-0.234501, -0.359477, -0.613612, 1])
# Initialize the wheel radius
wheel_radius = 0.2585

def detect_events(ts, added_events):
    # Calculate velocity and acceleration from positions using a Savitzky-Golay filter
    ts.data["Velocity"] = ktk.filters.savgol(ts, window_length=11, poly_order=2, deriv=1).data["Position"]
    ts.data["Acceleration"] = ktk.filters.savgol(ts, window_length=15, poly_order=2, deriv=2).data["Position"]

    pushes_indices = []  # List to store indices of detected pushes
    reco_indices = []    # List to store indices of detected recoveries
    thrusts_indices = []  # List to store indices of detected thrusts
    recoveries_indices = []  # List to store indices of detected recoveries

    last_event = None  # Track the last detected event

    for i in range(1, len(ts.data["Position"])):
        distance = np.linalg.norm(ts.data['Position'][i][:3] - wheel_center_position[:3])
        velocity_x = ts.data["Velocity"][i][0]  # Velocity in the x direction
        acceleration_x = ts.data["Acceleration"][i][0]  # Acceleration in the x direction

        # Check if the distance is within an interval of ±40% of the radius
        if (wheel_radius - (wheel_radius * 0.40)) <= distance <= (wheel_radius + (wheel_radius * 0.40)):
            if velocity_x > 0:
                # Check for a negative peak in acceleration
                if acceleration_x < -2:  # Ensure it exceeds the threshold
                    if ts.data["Acceleration"][i - 1][0] > acceleration_x < ts.data["Acceleration"][i + 1][0]:
                        if last_event != "push":
                            pushes_indices.append(i)
                            last_event = "push"
            elif velocity_x < 0:
                # Check for a positive peak in acceleration
                if acceleration_x > 2:  # Ensure it exceeds the threshold
                    if ts.data["Acceleration"][i - 1][0] < acceleration_x > ts.data["Acceleration"][i + 1][0]:
                        if last_event != "reco":
                            reco_indices.append(i)
                            last_event = "reco"

    # Add events for detected pushes
    for idx in pushes_indices:
        event_time = ts.time[idx]
        if (event_time, "push") not in added_events:
            ts = ts.add_event(event_time, "push")
            added_events.add((event_time, "push"))

    # Add events for detected recoveries
    for idx in reco_indices:
        event_time = ts.time[idx]
        if (event_time, "reco") not in added_events:
            ts = ts.add_event(event_time, "reco")
            added_events.add((event_time, "reco"))

    # Combine the detection of thrusts and recoveries into one loop
    push_angles = []
    recovery_distances = []

    for push_index in pushes_indices:
        nearest_recovery_index = next((i for i in reco_indices if i > push_index), None)
        if nearest_recovery_index is not None:
            thrusts_indices.append((push_index, nearest_recovery_index))
            push_position = ts.data["Position"][push_index]
            recovery_position = ts.data["Position"][nearest_recovery_index]
            push_angle = calculate_push_angle(push_position, recovery_position)
            push_angles.append(push_angle)

    for recovery_index in reco_indices:
        nearest_push_index = next((i for i in pushes_indices if i > recovery_index), None)
        if nearest_push_index is not None:
            recoveries_indices.append((recovery_index, nearest_push_index))
            # Calculate the distances between this recovery and the next push
            distances = [np.linalg.norm(ts.data['Position'][j][:3] - wheel_center_position[:3]) for j in range(recovery_index, nearest_push_index)]
            recovery_distances.append(distances)

    # Calculate the minimum distance for each recovery
    min_recovery_distances = [np.min(distances) for distances in recovery_distances]

    # Calculate and return the average angle every three pushes and minimum distance every three recoveries
    average_push_angles = []
    minimum_recovery_distances = []
    
    for i in range(2, len(push_angles), 3):
        average_angle = np.mean(push_angles[i - 2:i + 1])
        average_push_angles.append(average_angle)
        three_push_time = ts.time[thrusts_indices[i][0]]
        if (three_push_time, "three_pushes") not in added_events:
            print(f"Average push angle for 3 pushes: {average_angle:.2f} degrees")
            ts = ts.add_event(three_push_time, "three_pushes")
            added_events.add((three_push_time, "three_pushes"))

    for i in range(2, len(min_recovery_distances), 3):
        distances_for_three_recoveries = min_recovery_distances[i - 2:i + 1]
        avg_min_distance = np.mean(distances_for_three_recoveries)
        minimum_recovery_distances.append(avg_min_distance)
        three_recovery_time = ts.time[recoveries_indices[i][0]]
        if (three_recovery_time, "three_recoveries") not in added_events:
            print(f"Average minimum distance for 3 recoveries: {avg_min_distance:.2f} m")
            ts = ts.add_event(three_recovery_time, "three_recoveries")
            added_events.add((three_recovery_time, "three_recoveries"))

    return ts, added_events, pushes_indices, reco_indices, average_push_angles, minimum_recovery_distances


def calculate_push_angle(position1, position2):
    
    # Extract the direction vectors from the positions
    direction1 = position1[:2] / np.linalg.norm(position1[:2])
    direction2 = position2[:2] / np.linalg.norm(position2[:2])

    # Calculate the dot product between the direction vectors
    dot_product = np.dot(direction1, direction2)

    # Ensure dot product is within range [-1, 1] to avoid arccosine domain errors
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate the push angle in radians
    push_angle_radians = np.arccos(dot_product)

    # Convert the angle to degrees
    push_angle_degrees = np.degrees(push_angle_radians)

    return push_angle_degrees

def calculate_push_rate(ts, pushes_indices):
    """Calculate the push rate in pushes per second."""
    if ts.time.size > 0 and pushes_indices:
        total_time = ts.time[-1] - ts.time[0]
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Number of pushes: {len(pushes_indices)}")
        push_rate = len(pushes_indices) / total_time  # pushes per second
        return push_rate
    else:
        print("No valid time data or pushes indices.")
        return 0



def main():
    import optitrack as ot
    
    ot.start()
    ts = ktk.TimeSeries()
    added_events = set()
    # average_push_angles_all = []
    # minimum_recovery_distances_all = []
    pushes_indices = []  # Initialize pushes_indices at the start of main()

    try:
        while True:
            new_ts = ot.fetch()
            
            if "Position" in new_ts.data:
                if "Position" in ts.data:
                    ts.data["Position"] = np.vstack((ts.data["Position"], new_ts.data["Position"]))
                    ts.time = np.concatenate((ts.time, new_ts.time))
                else:
                    ts.data["Position"] = new_ts.data["Position"]
                    ts.time = new_ts.time

                unique_times, unique_indices = np.unique(ts.time, return_index=True)
                ts.data["Position"] = ts.data["Position"][unique_indices]
                ts.time = unique_times

                ts_resampled = ts.resample(60.0)

                # Handle non-constant sample rate exception
                if not ts_resampled:
                    continue

                ts_resampled, added_events, pushes_indices, reco_indices, average_push_angles, minimum_recovery_distances = detect_events(ts_resampled, added_events)


                # Afficher les valeurs moyennes de push angle à des fins de débogage
                print(f"Current average push angles: {average_push_angles}")
                #print(f"Average push angles: {average_push_angles_all}")

            #time.sleep(0.01)

    except KeyboardInterrupt:
        pass

    finally:
        # Calculate the push rate at the end of recording
        push_rate = calculate_push_rate(ts, pushes_indices)
        print(f"Push rate: {push_rate:.2f} pushes/s")
        if ts.time.size > 0:
            ts = ts.add_event(ts.time[-1], f"push_rate: {push_rate:.2f} pushes/s")

        if len(average_push_angles) > 0 or len(minimum_recovery_distances) > 0:
            # Ensure both lists have the same length by filling the shorter one with NaNs
            min_length = min(len(average_push_angles), len(minimum_recovery_distances))
            average_push_angles = average_push_angles[:min_length]
            minimum_recovery_distances = minimum_recovery_distances[:min_length]

            data = {
                "Average Push Angle (degrees)": average_push_angles,
                "Minimum Recovery Distance (m)": minimum_recovery_distances,
                "Push rate (pushes/s)": push_rate
            }
            df = pd.DataFrame(data)
            df = df.round(2)  # Round all values in the DataFrame to two decimal places
            print(df)
            
            # Save DataFrame to CSV
            df.to_csv('output_data.csv', index=False)

        ot.stop()

if __name__ == "__main__":
    main()
