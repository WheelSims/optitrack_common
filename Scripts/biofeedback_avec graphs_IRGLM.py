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
wheel_center_position = np.array([0.444269, 0.658312, -0.209179, 1])
# Initialize the wheel radius
wheel_radius = 0.255
# Initialize the user arm extended distance between his hand and the wheel center
user_arm_extended_distance = 0.150
threshold_min = -8
threshold_max = 7


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
        average_angle = np.mean(push_angles[i - 2 : i + 1])
        average_push_angles.append(average_angle)
        three_push_time = ts.time[thrusts_indices[i][0]]
        if (three_push_time, "three_pushes") not in added_events:
            print(
                f"Average push angle for 3 pushes: {average_angle:.2f} degrees"
            )
            ts = ts.add_event(three_push_time, "three_pushes")
            added_events.add((three_push_time, "three_pushes"))

    for i in range(2, len(min_recovery_distances), 3):
        distances_for_three_recoveries = min_recovery_distances[i - 2 : i + 1]
        avg_min_distance = np.mean(distances_for_three_recoveries)
        minimum_recovery_distances.append(avg_min_distance)
        three_recovery_time = ts.time[recoveries_indices[i][0]]
        if (three_recovery_time, "three_recoveries") not in added_events:
            print(
                f"Average minimum distance for 3 recoveries: {avg_min_distance:.2f} m"
            )
            ts = ts.add_event(three_recovery_time, "three_recoveries")
            added_events.add((three_recovery_time, "three_recoveries"))

    return (
        ts,
        added_events,
        pushes_indices,
        reco_indices,
        push_positions,
        recovery_positions,
        average_push_angles,
        minimum_recovery_distances,
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


def plot_wheel_setup(ax):
    # Draw the wheel
    wheel_circle = plt.Circle(
        (0, 0), wheel_radius, color="b", fill=False, linestyle="-", linewidth=2
    )
    ax.add_patch(wheel_circle)

    # Define the reference angle and valid angle zone
    reference_angle = 5 * np.pi / 6
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
    ax.fill(x1, y1, color="green", alpha=0.3, label=f"Valid zone")

    # Define the angles for the second valid angle zone (symmetric)
    valid_angle_start_rad_sym = np.pi - valid_angle_start_rad
    valid_angle_end_rad_sym = np.pi - valid_angle_end_rad

    # Generate arc points for the symmetric valid angle zone
    arc2 = np.linspace(valid_angle_start_rad_sym, valid_angle_end_rad_sym, 100)
    x2 = np.concatenate([[0], wheel_radius * np.cos(arc2), [0]])
    y2 = np.concatenate([[0], wheel_radius * np.sin(arc2), [0]])

    # Plot the symmetric valid angle zone
    ax.fill(x2, y2, color="green", alpha=0.3)

    ax.set_xlim(-wheel_radius - 0.1, wheel_radius + 0.1)
    ax.set_ylim(-wheel_radius - 0.1, wheel_radius + 0.1)
    ax.set_aspect("equal", "box")
    ax.set_title("Wheelchair Wheel with Push Angles")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.legend()

    # Enable grid lines
    ax.grid(True, which="both")  # Major and minor grid lines
    ax.minorticks_on()  # Enable minor ticks


def plot_min_distance_setup(ax):
    # Draw the vertical rectangle representing the wheel radius
    wheel_rect = plt.Rectangle(
        (-0.05, 0), 0.1, wheel_radius, color="blue", alpha=0.1
    )  # Expanded to 0.1 width
    ax.add_patch(wheel_rect)

    # Draw the ideal hand position zone
    ideal_zone_bottom = user_arm_extended_distance
    ideal_zone_top = (
        user_arm_extended_distance + 0.05
    )  # 15 cm above maximum extension
    ideal_zone_rect = plt.Rectangle(
        (-0.05, ideal_zone_bottom),
        0.1,
        ideal_zone_top - ideal_zone_bottom,
        color="green",
        alpha=0.3,
    )  # Green color
    ax.add_patch(ideal_zone_rect)

    ax.set_title("Ideal Hand Position Zone")
    ax.set_xlim(-0.1, 0.1)  # Adjusted to better fit the rectangle width
    ax.set_ylim(0, wheel_radius)  # Bottom of the y-axis at 0
    ax.set_aspect("equal", "box")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.grid(True, which="both")  # Major and minor grid lines
    ax.minorticks_on()  # Enable minor ticks


def update_push_angle_plot(
    ax1, push_positions, recovery_positions, wheel_radius
):
    ax1.clear()
    plot_wheel_setup(ax1)  # Correct call with a single argument

    # Calculate average positions of pushes every 3 pushes
    average_push_positions = []
    average_recovery_positions = []
    for i in range(2, len(push_positions), 3):
        avg_push_position = np.mean(push_positions[i - 2 : i + 1], axis=0)
        avg_recovery_position = np.mean(
            recovery_positions[i - 2 : i + 1], axis=0
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

        # Fill the "pizza slice" area with color
        ax1.fill(
            x,
            y,
            color="yellow",
            alpha=0.3,
            label="Push angle zone",
        )

        # Plot the lines of average push and recovery positions
        ax1.plot(
            [0, avg_push_pos[0] * push_ratio],
            [0, avg_push_pos[1] * push_ratio],
            linestyle="-",
            color="b",
            label="Push Line",
        )
        ax1.plot(
            [0, avg_recovery_pos[0] * recovery_ratio],
            [0, avg_recovery_pos[1] * recovery_ratio],
            linestyle="-",
            color="g",
            label="Recovery Line",
        )

    # Set axes and legends for the plot
    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    ax1.set_title("Push angle biofeedback")
    ax1.legend()

    # Display the plot
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)


def update_min_distance_plot(ax, minimum_recovery_distances):

    ax.clear()
    plot_min_distance_setup(ax)

    averaged_distances = []

    if len(minimum_recovery_distances) >= 3:
        for i in range(2, len(minimum_recovery_distances), 3):
            distances_for_three_recoveries = minimum_recovery_distances[
                i - 2 : i + 1
            ]
            avg_min_distance = np.mean(distances_for_three_recoveries)
            averaged_distances.append(avg_min_distance)

        # Plot only the last calculated average
        if averaged_distances:
            ax.axhline(
                y=averaged_distances[-1],
                color="r",
                linestyle="-",
                linewidth=2,
                label=f"Real-time Average: {averaged_distances[-1]:.2f}",
            )

    # Add the legend only once after plotting the last average
    if averaged_distances:
        ax.legend()

    plt.draw()
    plt.pause(0.01)


def update_min_distance_plot(ax, minimum_recovery_distances):
    ax.clear()
    plot_min_distance_setup(ax)

    averaged_distances = []

    if len(minimum_recovery_distances) >= 3:
        for i in range(2, len(minimum_recovery_distances), 3):
            distances_for_three_recoveries = minimum_recovery_distances[
                i - 2 : i + 1
            ]
            avg_min_distance = np.mean(distances_for_three_recoveries)
            averaged_distances.append(avg_min_distance)

        # Tracer uniquement la dernière moyenne calculée
        if averaged_distances:
            ax.axhline(
                y=averaged_distances[-1],
                color="r",
                linestyle="-",
                linewidth=2,
                label=f"Moyenne en temps réel: {averaged_distances[-1]:.2f}",
            )

    # Ajouter la légende une seule fois après avoir tracé la dernière moyenne
    if averaged_distances:
        ax.legend()

    plt.draw()
    plt.pause(0.01)


def main():
    ot.start()
    ts = ktk.TimeSeries()
    added_events = set()
    minimum_recovery_distances = []
    average_push_positions = []
    average_recovery_positions = []
    pushes_indices = []
    avg_push_positions = []
    avg_recovery_positions = []
    average_min_distances = []
    average_push_angles = []

    plt.ion()  # Enable interactive mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
    plot_wheel_setup(ax1)
    plot_min_distance_setup(ax2)

    push_scatter = ax1.scatter([], [], color="blue", label="Pushes", zorder=5)
    recovery_scatter = ax1.scatter(
        [], [], color="red", label="Recoveries", zorder=5
    )
    avg_push_scatter = ax1.scatter(
        [], [], color="green", label="Average Pushes", zorder=10
    )
    avg_recovery_scatter = ax1.scatter(
        [], [], color="orange", label="Average Recoveries", zorder=10
    )

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

                # Adjust the positions so that they fit into the plot with the center at (0,0)
                push_positions_adjusted = (
                    push_positions - wheel_center_position[:2]
                )
                recovery_positions_adjusted = (
                    recovery_positions - wheel_center_position[:2]
                )

                update_push_angle_plot(
                    ax1, push_positions, recovery_positions, wheel_radius
                )
                update_min_distance_plot(ax2, minimum_recovery_distances)
                plt.show()

            # Sleep for a short duration to avoid overloading the loop
            time.sleep(0.01)

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

        # Create a DataFrame
        data = {
            "Average Push Angles (degrees)": average_push_angles,
            "Minimum Distances (meters)": minimum_recovery_distances,
            "Push Rate (pushes/sec)": [push_rate]
            * len(average_push_angles),  # Repeat the push rate for each row
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
