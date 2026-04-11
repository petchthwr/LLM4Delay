import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_traj_file(file_path: str) -> pd.DataFrame:
    """
    Load trajectory data from a CSV file.
    Input: CSV file path
    Output: pd.DataFrame containing trajectory data
    """
    return pd.read_csv(file_path)

def find_active_traj(traj_data, current_time, focus_id):
    """
    Find and create the dataframe of focusing trajectory, active trajectories, and affecting trajectories.
    Input:
        traj_data: processed pd.DataFrame, trajectory data containing columns ['id', 'time', 'x', 'y', 'z']
        current_time: pd.Timestamp or str or float or int, the current time to find active and affecting trajectories
        focus_id: str, the id of the focusing trajectory
    Output:
        focusing_traj: pd.DataFrame, the trajectory data of the focusing trajectory
        active_traj: list of pd.DataFrame, the trajectory data of the active trajectories
        affecting_traj: list of pd.DataFrame, the trajectory data of the affecting trajectories
    Meaning:
        Focusing trajectory: the trajectory of the specified focus_id.
        Active trajectories: the trajectories that comprise all the states at the current time back to their first time of appearance.
        Affecting (Prior) trajectories: the trajectories that contain min_time_of_active_set comprising all their states from the first time of appearance to their last state.
    """

    # Time Conversion
    if isinstance(current_time, str):
        current_time = pd.to_datetime(current_time, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    elif isinstance(current_time, (int, float)):
        current_time = pd.to_datetime(current_time, unit='s')
    elif not isinstance(current_time, pd.Timestamp):
        raise ValueError("current_time must be str, float, int, or pd.Timestamp")
    if pd.isnull(current_time):
        raise ValueError("Invalid current_time after conversion.")

    # Ensure 'time' column is in datetime format
    traj_data = traj_data.copy()
    traj_data['time'] = pd.to_datetime(traj_data['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    traj_data = traj_data.dropna(subset=['time'])

    # Filter traj_data within 12 hours before current_time, avoid unnecessary data
    time_window_start = current_time - pd.Timedelta(hours=12)
    traj_data = traj_data[(traj_data['time'] <= current_time) & (traj_data['time'] >= time_window_start)]

    # Find active trajectories
    current_states = traj_data[traj_data['time'] == current_time] # States at the current time
    active_traj_id = current_states['id'].unique() # Find id of active trajectories
    active_traj = [] # Collect flight trajectories of active trajectories
    for i in range(len(active_traj_id)):
        traj_id = active_traj_id[i] # Find id of active trajectory
        traj = traj_data[traj_data['id'] == traj_id] # Find flight trajectory of active trajectory
        traj = traj[traj['time'] <= current_time] # Find states of active trajectory at the current time back to their first time of appearance
        active_traj.append(traj)

    # Find min_time_of_active_set
    min_time_of_active_set = active_traj[0]['time'].min()
    for i in range(1, len(active_traj)):
        min_time_of_active_set = min(min_time_of_active_set, active_traj[i]['time'].min())

    # Raise error if min_time_of_active_set is greater than current_time
    if min_time_of_active_set > current_time:
        raise ValueError("min_time_of_active_set is greater than current_time")

    # Find affecting trajectories
    affecting_states = traj_data[(traj_data['time'] >= min_time_of_active_set) & (traj_data['time'] <= current_time)]
    affecting_traj_id = affecting_states['id'].unique()
    affecting_traj_id = [traj_id for traj_id in affecting_traj_id if traj_id not in active_traj_id]

    # Collect flight trajectories of affecting trajectories
    affecting_traj = []
    for i in range(len(affecting_traj_id)):
        traj_id = affecting_traj_id[i]
        traj = traj_data[traj_data['id'] == traj_id]
        affecting_traj.append(traj)

    # find index of focusing trajectory
    idx_focusing_traj = np.where(active_traj_id == focus_id)[0][0]
    focusing_traj = active_traj.pop(idx_focusing_traj)

    # Optional: Plot the trajectories
    # plot_traj(focusing_traj, active_traj, affecting_traj, current_time.strftime('%Y-%m-%d_%H-%M-%S'), focus_id)

    return focusing_traj, active_traj, affecting_traj

def plot_traj(focusing_traj, active_traj, affecting_traj, current_time, focus_id):
    """
    Plot the focusing trajectory, active trajectories, and affecting trajectories.
    Input:
        focusing_traj: pd.DataFrame, the trajectory data of the focusing trajectory
        active_traj: list of pd.DataFrame, the trajectory data of the active trajectories
        affecting_traj: list of pd.DataFrame, the trajectory data of the affecting trajectories
        current_time: str, the current time for labeling the plot
        focus_id: str, the id of the focusing trajectory
    Output:
        None (saves plots to ../figs/)
    """
    # 0. Plot All Trajectories
    plt.figure(figsize=(6, 6))
    plt.plot(focusing_traj['x'], focusing_traj['y'], c='g')
    plt.scatter(focusing_traj['x'].iloc[0], focusing_traj['y'].iloc[0], c='g', marker='o')
    plt.scatter(focusing_traj['x'].iloc[-1], focusing_traj['y'].iloc[-1], c='g', marker='x')
    for traj in active_traj:
        plt.plot(traj['x'], traj['y'], c='r')
        plt.scatter(traj['x'].iloc[0], traj['y'].iloc[0], c='r', marker='o')
        plt.scatter(traj['x'].iloc[-1], traj['y'].iloc[-1], c='r', marker='x')
    for traj in affecting_traj:
        plt.plot(traj['x'], traj['y'], linestyle='dashed', c='b')
        plt.scatter(traj['x'].iloc[0], traj['y'].iloc[0], c='b', marker='o')
        plt.scatter(traj['x'].iloc[-1], traj['y'].iloc[-1], c='b', marker='x')
    plt.title(f"All Trajectories at {current_time}")
    plt.savefig(f'../figs/{current_time}_{focus_id}.png')
    plt.close()

    # 1. Focusing Trajectory
    plt.figure(figsize=(6, 6))
    plt.plot(focusing_traj['x'], focusing_traj['y'], c='g')
    plt.scatter(focusing_traj['x'].iloc[0], focusing_traj['y'].iloc[0], c='g', marker='o', label='Start')
    plt.scatter(focusing_traj['x'].iloc[-1], focusing_traj['y'].iloc[-1], c='g', marker='x', label='End')
    plt.title("Focusing Trajectory")
    plt.savefig(f'../figs/{current_time}_{focus_id}_focusing.png')
    plt.close()

    # 2. Active Trajectories
    plt.figure(figsize=(6, 6))
    for traj in active_traj:
        plt.plot(traj['x'], traj['y'], c='r')
        plt.scatter(traj['x'].iloc[0], traj['y'].iloc[0], c='r', marker='o')
        plt.scatter(traj['x'].iloc[-1], traj['y'].iloc[-1], c='r', marker='x')
    plt.title("Active Trajectories")
    plt.savefig(f'../figs/{current_time}_{focus_id}_active.png')
    plt.close()

    # 3. Affecting Trajectories
    plt.figure(figsize=(6, 6))
    for traj in affecting_traj:
        plt.plot(traj['x'], traj['y'], linestyle='dashed', c='b')
        plt.scatter(traj['x'].iloc[0], traj['y'].iloc[0], c='b', marker='o')
        plt.scatter(traj['x'].iloc[-1], traj['y'].iloc[-1], c='b', marker='x')
    plt.title("Affecting Trajectories")
    plt.savefig(f'../figs/{current_time}_{focus_id}_affecting.png')
    plt.close()

# For ATSCC Encoding Geometric Features Extraction is done here
def get_velocity(x):
    vel = np.diff(x, axis=0)
    vel = np.concatenate([vel, vel[-1:]], axis=0)
    return vel

def get_directional_vec(x):
    vel = get_velocity(x)
    vel_norm = np.linalg.norm(vel, axis=1, keepdims=True)
    vel_norm = np.where(vel_norm > 1e-9, vel_norm, 1e-9) # Replace near-zero norms with 1e-9 to avoid division by zero
    directional_vec = vel / vel_norm # Perform division
    return directional_vec

def get_polar(x):
    # Calculate r and theta from x, y
    r = np.linalg.norm(x[:, :2], axis=1, keepdims=True)
    theta = np.arctan2(x[:, 1:2], x[:, :1])
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    polar = np.concatenate([r, sin_theta, cos_theta], axis=1)
    return polar

def pad_stack_traj(focusing_traj, active_traj, affecting_traj, dir=True, polar=True): # Divide by 120 km
    """
    Prepare the trajectory data for ATSCC encoding.
    Input:
        focusing_traj: pd.DataFrame, the trajectory data of the focusing trajectory
        active_traj: list of pd.DataFrame, the trajectory data of the active trajectories
        affecting_traj: list of pd.DataFrame, the trajectory data of the affecting trajectories
        dir: bool, whether to include velocity and directional vector features
        polar: bool, whether to include polar coordinate features
    Output:
        focusing_traj: np.ndarray, the padded trajectory data of the focusing trajectory
        active_traj: np.ndarray or None, the padded trajectory data of the active trajectories
        affecting_traj: np.ndarray or None, the padded trajectory data of the affecting trajectories
    """

    # Only use x,y,z columns
    focusing_traj = focusing_traj[['x', 'y', 'z']]
    focusing_traj = focusing_traj.to_numpy()[::5, :] / 120 / 1000 # Downsample to 5Hz and convert to km

    if len(active_traj) == 0:
        active_traj = None
    else:
        active_traj = [traj[['x', 'y', 'z']] for traj in active_traj]
        active_traj = [traj.to_numpy()[::5, :] / 120 / 1000 for traj in active_traj]

    if len(affecting_traj) == 0:
        affecting_traj = None
    else:
        affecting_traj = [traj[['x', 'y', 'z']] for traj in affecting_traj]
        affecting_traj = [traj.to_numpy()[::5, :] / 120 / 1000 for traj in affecting_traj]

    if dir: # Get the velocity and directional vector
        focusing_traj = np.concatenate([focusing_traj, get_velocity(focusing_traj)], axis=1)
        if active_traj is not None:
            for i in range(len(active_traj)):
                active_traj[i] = np.concatenate([active_traj[i], get_velocity(active_traj[i])], axis=1)
        if affecting_traj is not None:
            for i in range(len(affecting_traj)):
                affecting_traj[i] = np.concatenate([affecting_traj[i], get_velocity(affecting_traj[i])], axis=1)

    if polar: # Get the polar coordinates
        focusing_traj = np.concatenate([focusing_traj, get_polar(focusing_traj)], axis=1)
        if active_traj is not None:
            for i in range(len(active_traj)):
                active_traj[i] = np.concatenate([active_traj[i], get_polar(active_traj[i])], axis=1)
        if affecting_traj is not None:
            for i in range(len(affecting_traj)):
                affecting_traj[i] = np.concatenate([affecting_traj[i], get_polar(affecting_traj[i])], axis=1)

    if active_traj is not None: # Pad the active trajectories
        max_len_active = max([traj.shape[0] for traj in active_traj])
        for i in range(len(active_traj)):
            active_traj[i] = np.pad(active_traj[i], ((0, max_len_active - active_traj[i].shape[0]), (0, 0)), mode='constant', constant_values=np.nan)
        active_traj = np.stack(active_traj)

    if affecting_traj is not None: # Pad the affecting trajectories
        max_len_affecting = max([traj.shape[0] for traj in affecting_traj]) if affecting_traj is not None else 0
        # Pad the affecting trajectories
        for i in range(len(affecting_traj)):
            affecting_traj[i] = np.pad(affecting_traj[i], ((0, max_len_affecting - affecting_traj[i].shape[0]), (0, 0)), mode='constant', constant_values=np.nan)
        affecting_traj = np.stack(affecting_traj)

    return focusing_traj, active_traj, affecting_traj





