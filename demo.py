from dataloader import *
from model.DelayLLM import LlamaDelayPredModel
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from datetime import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def inference(model, test_loader, task='dt_airspace', mean=None, std=None):
    """
    Evaluate the model on a test set and return regression metrics.

    Args:
        model: the model to evaluate
        test_loader: DataLoader for the test set
        device: torch.device
        task: optional, specify if rescaling is needed (e.g., 'dt_airspace')
        mean: mean used for standardization (if any)
        std: std used for standardization (if any)

    Returns:
        Dictionary of regression metrics
    """
    device = next(model.parameters()).device

    def maybe_rescale(tensor):
        return tensor * std + mean if task == 'dt_airspace' and mean is not None and std is not None else tensor

    model_train_flag = model.training
    model.eval()

    y_pred = []
    y_real = []
    dt_dummy = []
    delay = []

    with torch.no_grad():
        for flight_prompts, z, (y, dtd, d) in test_loader:
            flight_prompts = flight_prompts.to(device)
            z = {k: v.to(device) for k, v in z.items()}
            y = y.to(device).float()

            out = model(flight_prompts, z)
            y_pred.append(out.detach().cpu())
            y_real.append(y.detach().cpu())
            dt_dummy.append(dtd.detach().cpu())
            delay.append(d.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)
    y_real = torch.cat(y_real, dim=0)
    dt_dummy = torch.cat(dt_dummy, dim=0)
    delay_real = torch.cat(delay, dim=0)

    y_pred = maybe_rescale(y_pred)
    y_real = maybe_rescale(y_real)

    delay_pred = dt_dummy + y_pred

    model.train(model_train_flag)

    return delay_pred, delay_real, y_pred, y_real

def inference_realtime(model, test_loader, mean, std, task='dt_airspace'):
    delay_pred, delay_real, dt_pred, dt_real = inference(model, test_loader, task, mean, std)
    error = delay_pred - delay_real
    return delay_pred, delay_real, dt_pred, dt_real, error

def split_data_by_flight_id(data):
    unique_flight_ids = []
    for entry in data:
        flight_id = entry['flight_id']
        if flight_id not in unique_flight_ids:
            unique_flight_ids.append(flight_id)

    data_list = []
    for flight_id in unique_flight_ids:
        flight_data = [entry for entry in data if entry['flight_id'] == flight_id]
        data_list.append(flight_data)

    return data_list

def load_model(model_name, hf_token, model_path, device):
    model = LlamaDelayPredModel(model_name, hf_token, ablation_tag=None).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.z_projection.load_state_dict(checkpoint['z_projection'])
    model.linear_proj.load_state_dict(checkpoint['linear_proj'])
    return model, checkpoint['mean'], checkpoint['std']

# CUDA Device setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

hf_token = "YOUR HUGGINGFACE TOKEN"
model_name = 'EleutherAI/pythia-1b'
traj_encoder = 'ts2vec'

# Load pre-trained model
modelpath = 'SAVED_PATH'
model, mean, std = load_model(model_name, hf_token, modelpath, device='cuda')
tokenizer = model.tokenizer

# Load data
datapath = "DATA_PATH"
data_list = split_data_by_flight_id(load_scenarios(datapath))

for data in data_list:
    data.sort(key=lambda x: x['current_time'])

    # Operationl Time Range
    max_current_time = max([x['current_time'] for x in data])
    min_current_time = min([x['current_time'] for x in data])
    print(f"Flight ID: {data[0]['flight_id']}, Schedule date: {data[0]['flight_schedule']['sched_time_utc'][:10]}, "
          f"Current time range: {min_current_time} to {max_current_time}, Total data points: {len(data)}")

    # Demonstrate for flight_id 83408
    if data[0]['flight_id'] != 83408:
        continue

    # Collect current_time for reference
    current_time_list = [entry['current_time'] for entry in data]

    # The last data is the full trajectory, use it to plot x and y top-down view
    full_trajectory = data[-1]['traj_focusing']
    active_trajectory = data[-1]['traj_active']  # (N_active, T, 2)
    prior_trajectory = data[-1]['traj_affecting']  # (N_active, T, 2)

    # if active_trajectory or prior_trajectory is None, skip
    if active_trajectory is None or prior_trajectory is None:
        print(f"Skipping flight {data[0]['flight_id']} due to missing active or prior trajectory.")
        continue

    # if no active or prior trajectory, skip
    if len(active_trajectory) <= 2 or len(prior_trajectory) <= 2:
        print(f"Skipping flight {data[0]['flight_id']} due to no active or prior trajectory.")
        continue

    x = [point[0] for point in full_trajectory]
    y = [point[1] for point in full_trajectory]

    # Set global font sizes
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

    # Plot focusing trajectory in a separate figure
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, c='g')
    plt.scatter(x[0], y[0], c='g', marker='o')
    plt.scatter(x[-1], y[-1], c='g', marker='x')
    plt.title(f"Focusing Trajectory")
    plt.axis('equal')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"realtime_results/trajectory_{data[0]['flight_id']}.png", dpi=300)
    plt.show()

    # Plot multiple active trajectories
    print(active_trajectory.shape)
    plt.figure(figsize=(6, 6))
    for traj in active_trajectory:
        x_active = [point[0] for point in traj]
        y_active = [point[1] for point in traj]

        # Drop nan or zero padded points
        x_active = [x for x in x_active if not (isinstance(x, float) and (np.isnan(x) or x == 0.0))]
        y_active = [y for y in y_active if not (isinstance(y, float) and (np.isnan(y) or y == 0.0))]

        plt.plot(x_active, y_active, c='r')
        plt.scatter(x_active[0], y_active[0], c='r', marker='o')  # Start point
        plt.scatter(x_active[-1], y_active[-1], c='r', marker='x')  # End point
    plt.title(f"Active Trajectories")
    plt.axis('equal')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"realtime_results/active_trajectories_{data[0]['flight_id']}.png", dpi=300)
    plt.show()

    # Plot multiple prior trajectories
    print(prior_trajectory.shape)
    plt.figure(figsize=(6, 6))
    for traj in prior_trajectory:
        x_prior = [point[0] for point in traj]
        y_prior = [point[1] for point in traj]

        # Drop nan or zero padded points
        x_prior = [x for x in x_prior if not (isinstance(x, float) and (np.isnan(x) or x == 0.0))]
        y_prior = [y for y in y_prior if not (isinstance(y, float) and (np.isnan(y) or y == 0.0))]

        plt.plot(x_prior, y_prior, c='b')
        plt.scatter(x_prior[0], y_prior[0], c='b', marker='o')  # Start point
        plt.scatter(x_prior[-1], y_prior[-1], c='b', marker='x')  # End point
    plt.title(f"Prior Trajectories")
    plt.axis('equal')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"realtime_results/prior_trajectories_{data[0]['flight_id']}.png", dpi=300)
    plt.show()

    # Print prompt
    print("Sample Prompt:")
    print(data[-1]['flight_prompt'])
    print(data[-1]['notam_prompt'])

    # Prepare predictions
    test_loader = prepare_test_loader_for_realtime(
        data, tokenizer, mean, std, batch_size=32, task='dt_airspace', ablation_tag=None, traj_encoder=traj_encoder
    )
    delay_pred, delay_real, dt_pred, dt_real, error = inference_realtime(
        model, test_loader, mean, std, task='dt_airspace'
    )
    current_time_list = [datetime.strptime(c, '%Y-%m-%d %H:%M:%S') for c in current_time_list]

    # --- Create figure with GridSpec ---
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.5])  # left narrower, right wider

    # --- (1) Trajectory (big square, spans 2 rows) ---
    ax0 = fig.add_subplot(gs[:, 0])  # span both rows
    ax0.plot(x, y, c='g')
    ax0.scatter(x[0], y[0], c='g', marker='o', label='Start')
    ax0.scatter(x[-1], y[-1], c='g', marker='x', label='End')
    ax0.set_xlabel('X Coordinate')
    ax0.set_ylabel('Y Coordinate')
    ax0.set_title(f"Trajectory\nFlight {data[0]['flight_id']}")
    ax0.axis('equal')
    ax0.grid(True)
    ax0.set_xlim(-1, 1)
    ax0.set_ylim(-1, 1)
    ax0.legend()

    # --- (2) Delay vs. Time (top-right) ---
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(current_time_list, delay_real.numpy(), label='Real Delay', color='blue')
    ax1.plot(current_time_list, delay_pred.numpy(), label='Predicted Delay', color='orange')
    ax1.set_xlabel('Current Time')
    ax1.set_ylabel('Delay (minutes)')
    ax1.set_title(f"Delay vs Time\n({data[0]['flight_schedule']['sched_time_utc'][:10]})")
    ax1.tick_params(axis='x', rotation=30)
    ax1.legend()
    ax1.grid(True)

    # --- (3) Absolute Error vs. Time (bottom-right) ---
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(current_time_list, torch.abs(error).numpy(), label='Absolute Error', color='red')
    ax2.axhline(y=1, color='green', linestyle='--', label='1 Minute Error')
    ax2.axhline(y=0, color='purple', linestyle='--', label='Perfect Prediction')
    ax2.set_xlabel('Current Time')
    ax2.set_ylabel('Absolute Error (minutes)')
    ax2.set_title("Absolute Error vs Time")
    ax2.tick_params(axis='x', rotation=30)
    ax2.legend()
    ax2.grid(True)

    # --- Adjust layout ---
    plt.tight_layout()
    plt.savefig("realtime_results/overview_grid.png", dpi=600)
    plt.show()

    # Inferencing.


