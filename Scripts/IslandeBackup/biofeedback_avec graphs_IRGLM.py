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
wheel_center_position = np.array([0.447245, 0.658389, -0.206452, 1])
# Initialize the wheel radius
wheel_radius = 0.255
# Initialize the user arm extended distance between his hand and the wheel center
user_arm_extended_distance = 0.18086249
threshold_min = -2
threshold_max = 5
average_push_position = [0.63019762, 0.84031197]
average_recovery_position = [0.33883759, 0.87112703]
average_min_distance = 0.23
frequency_average = 3


def detect_events(ts, added_events):
    # Calculate velocity and acceleration from positions using a Savitzky-Golay filter
    ts.data["Velocity"] = ktk.filters.savgol(
        ts, window_length=11, poly_order=2, deriv=1
    ).data["Position"]
    ts.data["Acceleration"] = ktk.filters.savgol(
        ts, window_length=15, poly_order=2, deriv=2
    ).data["Position"]

    pushes_indices = []  # List to store indices of detected pushes
    reco_indices = []  # List to store indices of detected recoveries
    thrusts_indices = []  # List to store indices of detected thrusts
    recoveries_indices = []  # List to store indices of detected recoveries

    last_event = None  # Track the last detected event

    for i in range(1, len(ts.data["Position"])):
        distance = np.linalg.norm(
            ts.data["Position"][i][:3] - wheel_center_position[:3]
        )
        velocity_x = ts.data["Velocity"][i][0]  # Velocity in the x direction
        acceleration_x = ts.data["Acceleration"][i][
            0
        ]  # Acceleration in the x direction

        # Check if the distance is within an interval of ±40% of the radius
        if (
            (wheel_radius - (wheel_radius * 0.40))
            <= distance
            <= (wheel_radius + (wheel_radius * 0.40))
        ):
            if velocity_x > 0:
                # Check for a negative peak in acceleration
                if (
                    acceleration_x < threshold_min
                ):  # Ensure it exceeds the threshold
                    if (
                        ts.data["Acceleration"][i - 1][0]
                        > acceleration_x
                        < ts.data["Acceleration"][i + 1][0]
                    ):
                        if last_event != "push":
                            pushes_indices.append(i)
                            last_event = "push"
            elif velocity_x < 0:
                # Check for a positive peak in acceleration
                if (
                    acceleration_x > threshold_max
                ):  # Ensure it exceeds the threshold
                    if (
                        ts.data["Acceleration"][i - 1][0]
                        < acceleration_x
                        > ts.data["Acceleration"][i + 1][0]
                    ):
                        if last_event != "reco":
                            reco_indices.append(i)
                            last_event = "reco"

    # Add events for detected pushes
    for idx in pushes_indices:
        event_time = ts.time[idx]
        if (event_time, "push") not in added_events:
            ts = ts.add_event(event_time, "push")
            added_events.add(
                (event_time, "push")
            )  # FC Workaround duplicate events

    # Add events for detected recoveries
    for idx in reco_indices:
        event_time = ts.time[idx]
        if (event_time, "reco") not in added_events:
            ts = ts.add_event(event_time, "reco")
            added_events.add((event_time, "reco"))

    # Combine the detection of thrusts and recoveries into one loop
    push_angles = []
    recovery_distances = []
    push_positions = []
    recovery_positions = []

    for push_index in pushes_indices:
        nearest_recovery_index = next(
            (i for i in reco_indices if i > push_index), None
        )
        if nearest_recovery_index is not None:
            thrusts_indices.append((push_index, nearest_recovery_index))
            push_position = ts.data["Position"][push_index]
            recovery_position = ts.data["Position"][nearest_recovery_index]
            push_positions.append(push_position)
            recovery_positions.append(recovery_position)
            push_angle = calculate_push_angle(push_position, recovery_position)
            push_angles.append(push_angle)

    for recovery_index in reco_indices:
        nearest_push_index = next(
            (i for i in pushes_indices if i > recovery_index), None
        )
        if nearest_push_index is not None:
            recoveries_indices.append((recovery_index, nearest_push_index))
            # Calculate the distances between this recovery and the next push
            distances_marker = [
                np.linalg.norm(
                    ts.data["Position"][j][:3] - wheel_center_position[:3]
                )
                for j in range(recovery_index, nearest_push_index)
            ]
            distances_marker = np.array(distances_marker)
            distances = np.sqrt((distances_marker**2) - (0.055**2))
            recovery_distances.append(distances)

    # Calculate the minimum distance for each recovery
    min_recovery_distances = [
        np.min(distances) for distances in recovery_distances
    ]

    # Calculate and return the average angle every three pushes and minimum distance every three recoveries
    average_push_angles = []
    minimum_recovery_distances = []

    for i in range(2, len(push_angles), 3):
        if len(push_angles[i - frequency_average - 1 : i + 1]) > 0:
            average_angle = np.mean(
                push_angles[i - frequency_average - 1 : i + 1]
            )
            average_push_angles.append(average_angle)
            three_push_time = ts.time[thrusts_indices[i][0]]
            if (three_push_time, "three_pushes") not in added_events:
                print(
                    f"Average push angle for {frequency_average} pushes: {average_angle:.2f} degrees"
                )
                ts = ts.add_event(three_push_time, "three_pushes")
                added_events.add((three_push_time, "three_pushes"))

    for i in range(2, len(min_recovery_distances), 3):
        if len(push_angles[i - frequency_average - 1 : i + 1]) > 0:
            distances_for_three_recoveries = min_recovery_distances[
                i - frequency_average - 1 : i + 1
            ]
            avg_min_distance = np.mean(distances_for_three_recoveries)
            minimum_recovery_distances.append(avg_min_distance)
            three_recovery_time = ts.time[recoveries_indices[i][0]]
            if (three_recovery_time, "three_recoveries") not in added_events:
                print(
                    f"Average minimum distance for {frequency_average} recoveries: {avg_min_distance:.2f} m"
                )
                ts = ts.add_event(three_recovery_time, "three_recoveries")
                added_events.add((three_recovery_time, "three_recoveries"))

    return (
        ts,  # TimeSeries d'entrée avec les events ajoutés
        added_events,  # Liste d'events (workaround duplicata d'events dans ts)
        pushes_indices,  # Liste des indices des poussées
        reco_indices,  # Liste des indices des recouvrements
        push_positions,  # Liste de positions (x, y, z, 1)
        recovery_positions,  # Liste de positions (x, y, z, 1)
        average_push_angles,  # Liste des push angles
        minimum_recovery_distances,  # Liste de distances minimal centre de la roue-main
    )


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


def plot_wheel_setup(ax, average_push_position, average_recovery_position):
    # Draw the wheel
    wheel_circle = plt.Circle(
        (0, 0), wheel_radius, color="b", fill=False, linestyle="-", linewidth=2
    )
    ax.add_patch(wheel_circle)

    # Define the ideal zone extent (10%)
    push_zone_extent = 0.1 * wheel_radius
    recovery_zone_extent = 0.1 * wheel_radius

    # Calculate the angular extent for the ideal zones
    push_avg_angle = np.arctan2(
        average_push_position[1], average_push_position[0]
    )
    recovery_avg_angle = np.arctan2(
        average_recovery_position[1], average_recovery_position[0]
    )

    # Calculate the ideal zone angles
    push_zone_start_rad = push_avg_angle - push_zone_extent / wheel_radius
    push_zone_end_rad = push_avg_angle + push_zone_extent / wheel_radius
    recovery_zone_start_rad = (
        recovery_avg_angle - recovery_zone_extent / wheel_radius
    )
    recovery_zone_end_rad = (
        recovery_avg_angle + recovery_zone_extent / wheel_radius
    )

    # Generate arc points for the push ideal zone
    arc1 = np.linspace(push_zone_start_rad, push_zone_end_rad, 100)
    x1 = np.concatenate([[0], wheel_radius * np.cos(arc1), [0]])
    y1 = np.concatenate([[0], wheel_radius * np.sin(arc1), [0]])

    # Plot the push ideal zone
    ax.fill(x1, y1, color="green", alpha=0.3, label="Ideal Zone")

    # Generate arc points for the recovery ideal zone
    arc2 = np.linspace(recovery_zone_start_rad, recovery_zone_end_rad, 100)
    x2 = np.concatenate([[0], wheel_radius * np.cos(arc2), [0]])
    y2 = np.concatenate([[0], wheel_radius * np.sin(arc2), [0]])

    # Plot the recovery ideal zone
    ax.fill(x2, y2, color="green", alpha=0.3)

    ax.set_xlim(-wheel_radius - 0.1, wheel_radius + 0.1)
    ax.set_ylim(-wheel_radius - 0.1, wheel_radius + 0.1)
    ax.set_aspect("equal", "box")
    ax.set_title("Wheelchair wheel with push angles")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.legend(loc="lower left")


def plot_min_distance_setup(ax):
    rect_width = 0.1

    rectangle_bottom = user_arm_extended_distance
    rectangle_top = user_arm_extended_distance + wheel_radius + 0.10
    rectangle_height = rectangle_top - rectangle_bottom

    wheel_rect = plt.Rectangle(
        (-rect_width / 2, rectangle_bottom),
        rect_width,
        rectangle_height,
        color="blue",
        alpha=0.1,
    )
    ax.add_patch(wheel_rect)

    ideal_zone_bottom = user_arm_extended_distance
    ideal_zone_top = average_min_distance * 0.90
    ideal_zone_height = ideal_zone_top - ideal_zone_bottom

    ideal_zone_rect = plt.Rectangle(
        (-rect_width / 2, ideal_zone_bottom),
        rect_width,
        ideal_zone_height,
        color="green",
        alpha=0.3,
    )
    ax.add_patch(ideal_zone_rect)

    ax.set_xlim(-rect_width / 2, rect_width / 2)
    ax.set_ylim(rectangle_bottom - 0.1, rectangle_top)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")

    ax.set_xticks([])
    ax.set_xticklabels([])


def update_push_angle_plot(
    ax1, push_positions, recovery_positions, wheel_radius
):
    ax1.clear()
    plot_wheel_setup(
        ax1, average_push_position, average_recovery_position
    )  # Correct call with a single argument

    # Calculate average positions of pushes every 3 pushes
    average_push_positions = []
    average_recovery_positions = []
    for i in range(2, len(push_positions), 3):
        avg_push_position = np.mean(
            push_positions[i - frequency_average - 1 : i + 1], axis=0
        )
        avg_recovery_position = np.mean(
            recovery_positions[i - frequency_average - 1 : i + 1], axis=0
        )
        average_push_positions.append(avg_push_position)
        average_recovery_positions.append(avg_recovery_position)

    # Check if the lists are not empty
    if average_push_positions and average_recovery_positions:
        # Select only the last section to display
        avg_push_pos = average_push_positions[-1]
        avg_recovery_pos = average_recovery_positions[-1]

        # Calculate direction vectors and lengths to the edge of the circle
        norm_push = np.linalg.norm(avg_push_pos)
        norm_recovery = np.linalg.norm(avg_recovery_pos)

        push_ratio = wheel_radius / norm_push
        recovery_ratio = wheel_radius / norm_recovery

        # Calculate the angles of the lines
        angle_push = np.arctan2(avg_push_pos[1], avg_push_pos[0])
        angle_recovery = np.arctan2(avg_recovery_pos[1], avg_recovery_pos[0])

        # Create points for the "pizza slice"
        theta = np.linspace(angle_push, angle_recovery, num=100)
        x = np.concatenate([[0], wheel_radius * np.cos(theta), [0]])
        y = np.concatenate([[0], wheel_radius * np.sin(theta), [0]])

        # Remove the previous "pizza slice" area
        for collection in ax1.collections:
            collection.remove()

        # Plot the lines of average push and recovery positions
        ax1.plot(
            [0, avg_push_pos[0] * push_ratio],
            [0, avg_push_pos[1] * push_ratio],
            linestyle="-",
            color="b",
            linewidth=3,
            label="Push line",
        )
        ax1.plot(
            [0, avg_recovery_pos[0] * recovery_ratio],
            [0, avg_recovery_pos[1] * recovery_ratio],
            linestyle="-",
            color="g",
            linewidth=3,
            label="Recovery line",
        )

    # Set axes and legends for the plot
    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    ax1.set_title("Push angle biofeedback")
    ax1.legend()

    # Display the plot
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)


def update_min_distance_plot(ax, minimum_recovery_distances):
    ax.clear()
    plot_min_distance_setup(ax)

    if minimum_recovery_distances:
        ax.axhline(
            y=minimum_recovery_distances[-1],
            color="r",
            linestyle="-",
            linewidth=2,
            label=f"Average minimum distance over {frequency_average} recoveries: {minimum_recovery_distances[-1]:.2f}",
        )

        # Add legend only if there is at least one distance to display
        ax.legend(loc="lower left")

    plt.draw()
    plt.pause(0.1)


def main():
    # Ask which graph to display
    print("Which graph would you like to display?")
    print("1: Push Angles")
    print("2: Minimum Distances")
    choice = input("Enter 1 or 2: ")

    # Initialize variables
    ot.start()
    ts = ktk.TimeSeries()
    added_events = set()
    pushes_indices = []
    average_push_angles = []
    minimum_recovery_distances = []
    push_positions = []
    recovery_positions = []

    plt.ion()  # Enable interactive mode

    # Create figures and axes
    fig1, ax1 = plt.subplots(figsize=(8, 6))  # Push Angles
    fig2, ax2 = plt.subplots(figsize=(8, 6))  # Minimum Distances

    plot_wheel_setup(ax1, average_push_position, average_recovery_position)
    plot_min_distance_setup(ax2)

    # Determine which graph to display based on user choice
    if choice == "1":
        current_fig = fig1
        current_ax = ax1
        plt.close(fig2)  # Close the second figure
    elif choice == "2":
        current_fig = fig2
        current_ax = ax2
        plt.close(fig1)  # Close the first figure
    else:
        print("Invalid choice. Displaying both graphs.")
        current_fig = None

    try:
        while True:
            new_ts = ot.fetch()
            if new_ts is None:
                continue

            if new_ts.time.size > 0:
                if ts.time.size > 0:
                    ts.data["Position"] = np.vstack(
                        (ts.data["Position"], new_ts.data["Position"])
                    )
                    ts.time = np.concatenate((ts.time, new_ts.time))
                else:
                    ts.data["Position"] = new_ts.data["Position"]
                    ts.time = new_ts.time

                # Ensure unique time entries
                unique_times, unique_indices = np.unique(
                    ts.time, return_index=True
                )
                ts.data["Position"] = ts.data["Position"][unique_indices]
                ts.time = unique_times

                ts_resampled = ts.resample(60.0)

                # Handle non-constant sample rate exception
                if not ts_resampled:
                    continue

                (
                    ts_resampled,
                    added_events,
                    pushes_indices,
                    reco_indices,
                    push_positions,
                    recovery_positions,
                    average_push_angles,
                    minimum_recovery_distances,
                ) = detect_events(ts_resampled, added_events)

                # Update push and recovery positions on the plot
                push_positions = ts_resampled.data["Position"][pushes_indices][
                    :, :2
                ]
                recovery_positions = ts_resampled.data["Position"][
                    reco_indices
                ][:, :2]

                if current_ax == ax1:
                    update_push_angle_plot(
                        current_ax,
                        push_positions,
                        recovery_positions,
                        wheel_radius,
                    )
                elif current_ax == ax2:
                    update_min_distance_plot(
                        current_ax, minimum_recovery_distances
                    )

                if current_fig:
                    current_fig.canvas.draw()
                    current_fig.canvas.flush_events()

            plt.pause(1)

    except KeyboardInterrupt:
        pass

    finally:
        # Ensure unique timestamps in new_ts
        if new_ts is not None and new_ts.time.size > 0:
            unique_times, unique_indices = np.unique(
                new_ts.time, return_index=True
            )
            new_ts.data["Position"] = new_ts.data["Position"][unique_indices]
            new_ts.time = unique_times

        # Calculate push rate
        push_rate = calculate_push_rate(ts, pushes_indices)

        if len(average_push_angles) > 0 or len(minimum_recovery_distances) > 0:
            # Ensure both lists have the same length by filling the shorter one with NaNs
            min_length = min(
                len(average_push_angles), len(minimum_recovery_distances)
            )
            average_push_angles = average_push_angles[:min_length]
            minimum_recovery_distances = minimum_recovery_distances[
                :min_length
            ]

            # Calculate average positions of pushes and recoveries
            if len(push_positions) > 0:
                avg_push_position = np.mean(push_positions, axis=0)
            else:
                avg_push_position = [np.nan, np.nan]

            if len(recovery_positions) > 0:
                avg_recovery_position = np.mean(recovery_positions, axis=0)
            else:
                avg_recovery_position = [np.nan, np.nan]

            # Print average positions
            print(f"Average push position: {avg_push_position}")
            print(f"Average recovery position: {avg_recovery_position}")

            if minimum_recovery_distances:
                average_min_distance = np.mean(minimum_recovery_distances)
            else:
                average_min_distance = 0
            average_min_distance = round(average_min_distance, 2)
            print(f"Average min distance: {average_min_distance}")

            # Create a DataFrame
            data = {
                "Average push angles (degrees)": average_push_angles,
                "Minimum distances (meters)": minimum_recovery_distances,
                "Push rate (pushes/sec)": [push_rate]
                * len(average_push_angles),
            }
            df = pd.DataFrame(data)
            df = df.round(
                2
            )  # Round all values in the DataFrame to two decimal places
            print(df)

            # Save DataFrame to CSV
            df.to_csv("output_data.csv", index=False)

            # Plot new_ts if it has unique timestamps
            if new_ts is not None:
                new_ts.plot()

        ot.stop()


if __name__ == "__main__":
    main()
