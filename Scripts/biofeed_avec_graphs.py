# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:09:05 2024

@author: User
"""

import optitrack as ot
import numpy as np
import kineticstoolkit.lab as ktk
import time
import pandas as pd
import matplotlib.pyplot as plt

# Initialize the wheel center position
wheel_center_position = np.array([-0.234501, -0.359477, -0.613612, 1])
# Initialize the wheel radius
wheel_radius = 0.28 #0.2585
# Initialize the user arm extended distance between his hand and the wheel center
user_arm_extended_distance = 0.11

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



# def plot_initial_setup(ax):
#     # Draw the wheel
#     wheel_circle = plt.Circle((0, 0), wheel_radius, color='b', fill=False, linestyle='-', linewidth=2)
#     ax.add_patch(wheel_circle)

#     # Define the reference angle and valid angle zone
#     reference_angle = 5*np.pi/6
#     valid_angle_start_deg = 85
#     valid_angle_end_deg = 100

#     # Convert angles to radians (in the counterclockwise direction)
#     valid_angle_start_rad = reference_angle - np.deg2rad(valid_angle_start_deg)
#     valid_angle_end_rad = reference_angle - np.deg2rad(valid_angle_end_deg)

#     # Generate arc points for the valid angle zone
#     arc1 = np.linspace(valid_angle_start_rad, valid_angle_end_rad, 100)
#     x1 = np.concatenate([[0], wheel_radius * np.cos(arc1), [0]])
#     y1 = np.concatenate([[0], wheel_radius * np.sin(arc1), [0]])

#     # Plot the valid angle zone
#     ax.fill(x1, y1, color='green', alpha=0.3, label=f'Valid zone')

#     # Define the angles for the second valid angle zone (symmetric)
#     valid_angle_start_rad_sym = np.pi - valid_angle_start_rad
#     valid_angle_end_rad_sym = np.pi - valid_angle_end_rad

#     # Generate arc points for the symmetric valid angle zone
#     arc2 = np.linspace(valid_angle_start_rad_sym, valid_angle_end_rad_sym, 100)
#     x2 = np.concatenate([[0], wheel_radius * np.cos(arc2), [0]])
#     y2 = np.concatenate([[0], wheel_radius * np.sin(arc2), [0]])

#     # Plot the symmetric valid angle zone
#     ax.fill(x2, y2, color='green', alpha=0.3)

#     ax.set_xlim(-wheel_radius - 0.1, wheel_radius + 0.1)
#     ax.set_ylim(-wheel_radius - 0.1, wheel_radius + 0.1)
#     ax.set_aspect('equal', 'box')
#     ax.set_title('Wheelchair Wheel with Push Angles')
#     ax.set_xlabel('X Position (m)')
#     ax.set_ylabel('Y Position (m)')
#     ax.legend()
    
#     # Dessiner le rectangle vertical représentant le rayon de la roue
#     wheel_rect = plt.Rectangle((-0.01, -wheel_radius), 0.02, wheel_radius * 2, color='blue', alpha=0.1)
#     ax.add_patch(wheel_rect)

#     # Dessiner la zone idéale de position de la main
#     ideal_zone_bottom = user_arm_extended_distance
#     ideal_zone_top = user_arm_extended_distance + 0.1  # 10 cm au-dessus de l'extension maximale
#     ideal_zone_rect = plt.Rectangle((-0.01, ideal_zone_bottom), 0.02, ideal_zone_top - ideal_zone_bottom, color='red', alpha=0.3)
#     ax.add_patch(ideal_zone_rect)

#     # Enable grid lines
#     ax.grid(True, which='both')  # Major and minor grid lines
#     ax.minorticks_on()           # Enable minor ticks

def plot_wheel_setup(ax):
    # Draw the wheel
    wheel_circle = plt.Circle((0, 0), wheel_radius, color='b', fill=False, linestyle='-', linewidth=2)
    ax.add_patch(wheel_circle)

    # Define the reference angle and valid angle zone
    reference_angle = 5*np.pi/6
    valid_angle_start_deg = 85
    valid_angle_end_deg = 100

    # Convert angles to radians (in the counterclockwise direction)
    valid_angle_start_rad = reference_angle - np.deg2rad(valid_angle_start_deg)
    valid_angle_end_rad = reference_angle - np.deg2rad(valid_angle_end_deg)

    # Generate arc points for the valid angle zone
    arc1 = np.linspace(valid_angle_start_rad, valid_angle_end_rad, 100)
    x1 = np.concatenate([[0], wheel_radius * np.cos(arc1), [0]])
    y1 = np.concatenate([[0], wheel_radius * np.sin(arc1), [0]])

    # Plot the valid angle zone
    ax.fill(x1, y1, color='green', alpha=0.3, label=f'Valid zone')

    # Define the angles for the second valid angle zone (symmetric)
    valid_angle_start_rad_sym = np.pi - valid_angle_start_rad
    valid_angle_end_rad_sym = np.pi - valid_angle_end_rad

    # Generate arc points for the symmetric valid angle zone
    arc2 = np.linspace(valid_angle_start_rad_sym, valid_angle_end_rad_sym, 100)
    x2 = np.concatenate([[0], wheel_radius * np.cos(arc2), [0]])
    y2 = np.concatenate([[0], wheel_radius * np.sin(arc2), [0]])

    # Plot the symmetric valid angle zone
    ax.fill(x2, y2, color='green', alpha=0.3)

    ax.set_xlim(-wheel_radius - 0.1, wheel_radius + 0.1)
    ax.set_ylim(-wheel_radius - 0.1, wheel_radius + 0.1)
    ax.set_aspect('equal', 'box')
    ax.set_title('Wheelchair Wheel with Push Angles')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.legend()

    # Enable grid lines
    ax.grid(True, which='both')  # Major and minor grid lines
    ax.minorticks_on()           # Enable minor ticks

def plot_min_distance_setup(ax):
    # Dessiner le rectangle vertical représentant le rayon de la roue
    wheel_rect = plt.Rectangle((-0.05, 0), 0.1, wheel_radius, color='blue', alpha=0.1)  # Élargi à 0.1 de largeur
    ax.add_patch(wheel_rect)

    # Dessiner la zone idéale de position de la main
    ideal_zone_bottom = user_arm_extended_distance
    ideal_zone_top = user_arm_extended_distance + 0.15  # 15 cm au-dessus de l'extension maximale
    ideal_zone_rect = plt.Rectangle((-0.05, ideal_zone_bottom), 0.1, ideal_zone_top - ideal_zone_bottom, color='green', alpha=0.3)  # Couleur verte
    ax.add_patch(ideal_zone_rect)

    ax.set_title('Ideal Hand Position Zone')
    ax.set_xlim(-0.1, 0.1)  # Adjusted to better fit the rectangle width
    ax.set_ylim(0, wheel_radius)  # Bas de l'axe y à 0
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.grid(True, which='both')  # Major and minor grid lines
    ax.minorticks_on()  # Enable minor ticks

def update_plot(ax1, ax2, push_positions, recovery_positions, minimum_recovery_distances):
    # Mise à jour du graphique de la roue
    ax1.clear()
    plot_wheel_setup(ax1)
    ax1.scatter(push_positions[:, 0], push_positions[:, 1], color='blue', label='Poussées', zorder=5)
    ax1.scatter(recovery_positions[:, 0], recovery_positions[:, 1], color='red', label='Récupérations', zorder=5)
    ax1.legend()

    # Mise à jour du graphique des distances minimales
    ax2.clear()
    plot_min_distance_setup(ax2)

    if len(minimum_recovery_distances) >= 3:
        for i in range(2, len(minimum_recovery_distances), 3):
            distances_for_three_recoveries = minimum_recovery_distances[i - 2:i + 1]
            avg_min_distance = np.mean(distances_for_three_recoveries)
            minimum_recovery_distances.append(avg_min_distance)
            # three_recovery_time = ts.time[recoveries_indices[i][0]]
            # if (three_recovery_time, "three_recoveries") not in added_events:
            #     print(f"Average minimum distance for 3 recoveries: {avg_min_distance:.2f} m")
            #     ts = ts.add_event(three_recovery_time, "three_recoveries")
            #     added_events.add((three_recovery_time, "three_recoveries"))
            ax2.axhline(y=avg_min_distance, color='r', linestyle='-', linewidth=2, label=f'Moyenne en temps réel: {avg_min_distance:.2f}')
            ax2.legend()

    plt.draw()
    plt.pause(0.01)



def main():
    ot.start()
    ts = ktk.TimeSeries()
    added_events = set()
    minimum_recovery_distances = []
    average_push_positions = []
    average_recovery_positions = []
    pushes_indices = []  # Initialize pushes_indices at the start of main()
    avg_push_positions = []  # Initialize average push positions list
    avg_recovery_positions = []  # Initialize average recovery positions list

    # # Initial plot setup
    # plt.ion()  # Enable interactive mode
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
    # plot_initial_setup(ax1)
    
    # Initial plot setup
    plt.ion()  # Enable interactive mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
    plot_wheel_setup(ax1)
    plot_min_distance_setup(ax2)

    push_scatter = ax1.scatter([], [], color='blue', label='Pushes', zorder=5)
    recovery_scatter = ax1.scatter([], [], color='red', label='Recoveries', zorder=5)
    avg_push_scatter = ax1.scatter([], [], color='green', label='Average Pushes', zorder=10)
    avg_recovery_scatter = ax1.scatter([], [], color='orange', label='Average Recoveries', zorder=10)

    try:
        while True:
            new_ts = ot.fetch()
            if new_ts is None:
                continue

            if new_ts.time.size > 0:
                if ts.time.size > 0:
                    ts.data["Position"] = np.vstack((ts.data["Position"], new_ts.data["Position"]))
                    ts.time = np.concatenate((ts.time, new_ts.time))
                else:
                    ts.data["Position"] = new_ts.data["Position"]
                    ts.time = new_ts.time

                # Ensure unique time entries
                unique_times, unique_indices = np.unique(ts.time, return_index=True)
                ts.data["Position"] = ts.data["Position"][unique_indices]
                ts.time = unique_times

                ts_resampled = ts.resample(60.0)

                # Handle non-constant sample rate exception
                if not ts_resampled:
                    continue

                #ts_resampled, added_events, pushes_indices, reco_indices, _, _ = detect_events(ts_resampled, added_events)
                ts_resampled, added_events, pushes_indices, reco_indices, average_push_angles, minimum_recovery_distances = detect_events(ts_resampled, added_events)

                # Update push and recovery positions on the plot
                push_positions = ts_resampled.data["Position"][pushes_indices][:, :2]
                recovery_positions = ts_resampled.data["Position"][reco_indices][:, :2]

                # Adjust the positions so that they fit into the plot with the center at (0,0)
                push_positions_adjusted = push_positions - wheel_center_position[:2]
                recovery_positions_adjusted = recovery_positions - wheel_center_position[:2]
                
                # Calculer la distance minimale entre la main et le centre de la roue pour chaque récupération
                if len(recovery_positions_adjusted) >= 1:
                    distances = np.linalg.norm(recovery_positions_adjusted[-1])
                    min_distance = np.min(distances)
                    minimum_recovery_distances.append(min_distance)

                # Mettre à jour les graphiques
                update_plot(ax1, ax2, push_positions_adjusted, recovery_positions_adjusted, minimum_recovery_distances)

                # push_scatter.set_offsets(push_positions_adjusted)
                # recovery_scatter.set_offsets(recovery_positions_adjusted)

                # # Calculate average positions every 3 pushes and recoveries
                # if len(pushes_indices) >= 3:
                #     avg_push_x = np.mean(push_positions[-3:, 0])
                #     avg_push_y = -np.mean(push_positions[-3:, 1])  # Invert y-coordinate for upper part
                #     avg_push_positions.append((avg_push_x, avg_push_y))

                #     avg_recovery_x = np.mean(recovery_positions[-3:, 0])
                #     avg_recovery_y = -np.mean(recovery_positions[-3:, 1])  # Invert y-coordinate for upper part
                #     avg_recovery_positions.append((avg_recovery_x, avg_recovery_y))

                #     avg_push_scatter.set_offsets(avg_push_positions[-1:])
                #     avg_recovery_scatter.set_offsets(avg_recovery_positions[-1:])

                #     avg_push_positions_adjusted = avg_push_positions - wheel_center_position[:2]
                #     avg_recovery_positions_adjusted = avg_recovery_positions - wheel_center_position[:2]
                #     # Print average positions for debugging purposes
                #     print(f"Average Push Position: {avg_push_positions_adjusted[-1]}")
                #     print(f"Average Recovery Position: {avg_recovery_positions_adjusted[-1]}")

                # # Redraw the plot
                # plt.draw()
                # plt.pause(0.01)  # Pause to update the plot
                
                # # Plot real-time data including the new biofeedback mechanism
                # plot_real_time_data(ax, push_positions_adjusted, recovery_positions_adjusted, average_push_angles, minimum_recovery_distances)

            # Sleep for a short duration to avoid overloading the loop
            time.sleep(0.01)

    except KeyboardInterrupt:
        pass

    finally:
        ot.stop()

if __name__ == "__main__":
    main()