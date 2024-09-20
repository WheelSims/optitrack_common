# This prints the current position of the arm.

import optitrack as ot
import matplotlib.pyplot as plt
import numpy as np


WHEEL_CENTER_POSITION = 0.656970  # y-coordinate of the wheel centre


ot.start()

plt.figure()
plt.pause(1)
ts = ot.fetch()

while len(ts.time) == 0 or (ts.time[-1] - ts.time[0] < 1):
    plt.pause(1)
    ts = ot.fetch()


ts.data["WheelCenter"] = np.ones(ts.time.shape) * WHEEL_CENTER_POSITION
ts.plot()

print(
    "Position of the hand in full extended arm = "
    + str(np.mean(ts.data["Position"][:, 1]))
)
ot.stop()
