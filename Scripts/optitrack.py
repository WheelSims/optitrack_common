# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:12:24 2024

@author: User
"""

import sys
import time
from NatNetClient import NatNetClient
import kineticstoolkit.lab as ktk
import numpy as np

# Initialize empty lists to store positions and timestamps
_positions = []  # Initialize empty lists to store positions
_times = []      # and timestamps
# Frame limit to retain
frame_limit = 1000
# First system time
first_system_time = time.time()
# Create a global variable for the NatNet client
_streaming_client = NatNetClient()


def receive_rigid_body_frame(new_id: int, position: np.ndarray, orientation: np.ndarray) -> None:
    """
    Callback function to receive rigid body data.

    This function is called whenever a new rigid body frame is received.

    Parameters
    ----------
    new_id: int
        The ID of the rigid body.
    position: np.ndarray
        The position of the rigid body as a NumPy array.
    orientation: np.ndarray
        The orientation of the rigid body as a NumPy array.

    Returns
    -------
    None

    """ 
    # Calculate elapsed time since the first frame
    relative_time = time.time() - first_system_time

    # Add position and timestamp
    position_with1 = np.append(position, 1.0)
    _positions.append(position_with1)
    _times.append(relative_time)

    # If frame count exceeds limit, remove oldest frames
    if len(_positions) > frame_limit:
        _positions.pop(0)
        _times.pop(0)
        
        

def fetch() -> ktk.TimeSeries:
    """
    Create and return a timeseries from the current state of positions and times lists.

    Returns
    -------
    ktk.TimeSeries
        A time series object containing the positions of rigid bodies over time.

    """
    # Convert lists of times and positions to numpy arrays
    times_np = np.array(_times)
    positions_np = np.array(_positions)

    # Create the timeseries
    ts = ktk.TimeSeries(data={'Position': positions_np}, time=times_np)

    return ts



def start() -> None:
    """
    Start receiving data from NatNet.

    Returns
    -------
    None

    """
    # Configure client to receive rigid body data
    _streaming_client.rigid_body_listener = receive_rigid_body_frame

    # Start NatNet client
    is_running = _streaming_client.run()
    if not is_running:
        print("ERROR: Could not start streaming client.")  # Indicates an error if the streaming client fails to start
        sys.exit(1)

    # Wait for client to connect
    while not _streaming_client.connected():
        time.sleep(1)

    print("Connected to the server.")  # Indicates successful connection to the NatNet server



def stop() -> None:
    """
    Stop receiving data from NatNet

    Returns
    -------
    None

    """
    # Stop NatNet client
    _streaming_client.shutdown()

def clear() -> None:
    """
    Clear positions and times lists
    
    Returns
    -------
    None
    
    """
    _positions.clear()
    _times.clear()

