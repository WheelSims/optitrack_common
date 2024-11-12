# Biofeedback - Félix
# Editted- Ateayeh

import optitrack as ot
import numpy as np
import pandas as pd
import kineticstoolkit.lab as ktk
import matplotlib.pyplot as plt
import time

plt.close("all")
# Positions and radii, recalculate for each user
USER_INITIAL_MINIMAL_POSITION = 0.7468357046445211
USER_ARM_EXTENDED_HAND_POSITION = 0.683444376975771
WHEEL_RADIUS = 0.265
WHEEL_CENTER_POSITION = 0.659715  # y-coordinate of the wheel centre
name_file = "Aftertrain-P08"

# Create lists to store data for Excel
cadence_data = []
hand_position_data = []

# Number of last cycles to calculate cadence and lowest position
# for biofeedback:
BIOFEEDBACK_CALCULATE_CADENCE_ON_N_CYCLES = 1
BIOFEEDBACK_CALCULATE_LOWEST_HAND_POSITION_ON_N_CYCLES = 1
# for calculating initial parameters:
AVERAGE_CALCULATE_CADENCE_ON_N_CYCLES = 30
AVERAGE_CALCULATE_LOWEST_HAND_POSITION_ON_N_CYCLES = 30

# Misc graphic constants
BIOFEEDBACK_RECTANGLE_WIDTH = 0.1


# -----------------------------------


def update_plot(current_lowest_position):
    # Prepare the figure for lowest hand position
    plt.clf()

    # Ideal zone
    bounds = [
        USER_ARM_EXTENDED_HAND_POSITION,
        USER_INITIAL_MINIMAL_POSITION
        - 0.1
        * (USER_ARM_EXTENDED_HAND_POSITION - USER_INITIAL_MINIMAL_POSITION),
    ]
    plt.fill_between(
        [0, 1],
        [bounds[0], bounds[0]],
        [bounds[1], bounds[1]],
        color="green",
        alpha=1,
    )

    plt.plot(
        [0, 1],
        [current_lowest_position, current_lowest_position],
        color="k",
        linewidth=3,
    )

    plt.plot([0.5], [WHEEL_CENTER_POSITION], "ko")
    plt.plot(
        [0, 1],
        [
            WHEEL_CENTER_POSITION + WHEEL_RADIUS,
            WHEEL_CENTER_POSITION + WHEEL_RADIUS,
        ],
        "k--",
    )
    plt.plot(
        [0, 1],
        [
            WHEEL_CENTER_POSITION - WHEEL_RADIUS,
            WHEEL_CENTER_POSITION - WHEEL_RADIUS,
        ],
        "k--",
    )

    plt.axis(
        [
            0,
            1,
            WHEEL_CENTER_POSITION - 1.5 * WHEEL_RADIUS,
            WHEEL_CENTER_POSITION + 1.5 * WHEEL_RADIUS,
        ]
    )
    plt.gcf().set_size_inches(2, 4)
    plt.xticks([])
    plt.yticks([])


# Start the OptiTrack connection
ot.start()

try:
    while True:
        plt.pause(0.5)

        # Fetch the position data from OptiTrack
        ts = ot.fetch()

        # Ensure we have something to process
        if len(ts.time) == 0:
            continue

        # Calculate the mean and std of x in the last 10 seconds (to avoid
        # accumulating data where the participant is not propelling)
        n_seconds = 10
        if ts.time[-1] - ts.time[0] > n_seconds:
            temp_ts = ts.get_ts_after_time(ts.time[-1] - n_seconds)
        else:
            temp_ts = ts
        mean_x = np.mean(temp_ts.data["Position"][:, 0])
        std_x = np.std(temp_ts.data["Position"][:, 0])

        # Calculate the "front" (1) and "back" (-1) states of the hand
        x = ts.data["Position"][:, 0]
        ts.data["State"] = np.zeros(ts.time.shape)
        ts.data["State"][x > mean_x + std_x / 2] = 1
        ts.data["State"][x < mean_x - std_x / 2] = -1

        # Zeros are "in-between" states. Replace them by the last 1 or -1.
        temp_ts = ts.get_subset("State")
        not_zero = temp_ts.data["State"] != 0
        temp_ts.time = temp_ts.time[not_zero]
        temp_ts.data["State"] = temp_ts.data["State"][not_zero]
        temp_ts = temp_ts.resample(ts.time, kind="nearest")
        ts.data["State"] = temp_ts.data["State"]

        # Convert these transitions to events
        transitions = np.zeros(ts.time.shape)
        transitions[1:] = ts.data["State"][1:] - ts.data["State"][0:-1]
        cycle_times = ts.time[transitions > 0]
        for cycle_time in cycle_times:
            ts.add_event(cycle_time, "cycle", in_place=True)

        # ----- Do the calculations -----
        n_events = ts.count_events("cycle")

        # Calculate the cycle time based on the N last events
        for calculate_on_n_events in [
            BIOFEEDBACK_CALCULATE_CADENCE_ON_N_CYCLES,
            AVERAGE_CALCULATE_CADENCE_ON_N_CYCLES,
        ]:

            if n_events > calculate_on_n_events:
                temp_ts = ts.get_ts_between_events(
                    "cycle",
                    "cycle",
                    n_events - calculate_on_n_events - 1,
                    n_events - 1,
                    inclusive=True,
                )
                cycle_time = (
                    temp_ts.time[-1] - temp_ts.time[0]
                ) / calculate_on_n_events
                cadence = 60 / cycle_time
                print(
                    "Cadence for the last "
                    + str(calculate_on_n_events)
                    + " cycles = "
                    + str(cadence)
                )
                if (
                    calculate_on_n_events
                    == BIOFEEDBACK_CALCULATE_CADENCE_ON_N_CYCLES
                ):
                    cadence_data.append(cadence)

        # Calculate the lowest hand position based on the N last events
        for i, calculate_on_n_events in enumerate(
            [
                BIOFEEDBACK_CALCULATE_LOWEST_HAND_POSITION_ON_N_CYCLES,
                AVERAGE_CALCULATE_LOWEST_HAND_POSITION_ON_N_CYCLES,
            ]
        ):

            if n_events > calculate_on_n_events:

                hand_positions = []
                for i_cycle in range(
                    n_events - calculate_on_n_events - 1,
                    n_events - 1,
                ):
                    temp_ts = ts.get_ts_between_events(
                        "cycle",
                        "cycle",
                        i_cycle,
                        i_cycle + 1,
                        inclusive=True,
                    )
                    hand_positions.append(
                        np.min(temp_ts.data["Position"][:, 1])
                    )

                hand_position = np.mean(hand_positions)
                print(
                    "Hand position for "
                    + str(calculate_on_n_events)
                    + " cycles = "
                    + str(hand_position)
                )

                if i == 0:
                    hand_position_data.append(hand_position)
                    update_plot(hand_position)

except KeyboardInterrupt:
    pass

finally:
    # Save data to Excel at the end of the run
    data = {"Cadence": cadence_data, "Hand Position": hand_position_data}
    df = pd.DataFrame(data)
    df.to_excel(f"{name_file}.xlsx", index=False)
    print(f"Data saved to {name_file}.xlsx")

    # Stop the OptiTrack connection
    ot.stop()
