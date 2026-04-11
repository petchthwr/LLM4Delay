import pandas as pd
import numpy as np
import pickle
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from ml_utils import evaluate_delay, scenarios_to_df, extract_labels, label_encode_features, transform_with_label_encoders
from model.traj_encoder import Regressor

DAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2,
    "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
}

def reproducibility(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def load_scenarios(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    with open(file_path, 'rb') as f:
        scenarios = pickle.load(f)
    return scenarios

# Trajectory Feature Extraction
def scenarios_to_traj(scenarios: list) -> np.ndarray:
    """
    Convert a list of scenarios (each containing a trajectory of shape (T, F))
    into a padded 3D NumPy array of shape (N, max_T, F),
    padding missing values with NaN.
    """

    # Determine max sequence length and feature dimension
    max_len = max(s['traj_focusing'].shape[0] for s in scenarios)
    feat_dim = scenarios[0]['traj_focusing'].shape[1]

    # Initialize padded array
    traj_array = np.full((len(scenarios), max_len, feat_dim), np.nan)

    # Fill each trajectory
    for i, s in enumerate(scenarios):
        traj = s['traj_focusing']
        T = traj.shape[0]
        traj_array[i, :T, :] = traj

    return traj_array

# Train/Val/Test Split
def split_train_val_test(df: pd.DataFrame, traj: np.ndarray, train_ratio: float = 0.8):
    """
    Split a DataFrame and aligned trajectory array into train/val/test.

    Behavior:
      - Test = last part of the ORIGINAL df (no shuffle; e.g., chronological)
      - Val = last part of the REMAINING front portion (no shuffle)
        - Train = front part of the REMAINING portion (no shuffle)
      - Trajectory array is split with matching indices.

    Args:
        df: DataFrame of size N
        traj: np.ndarray of shape (N, T, F) aligned row-wise with df
        train_ratio: fraction of all data used for train
        seed: RNG seed for shuffling the train+val portion

    Returns:
        train_df, val_df, test_df, train_traj, val_traj, test_traj
    """
    assert len(df) == traj.shape[0], "df and traj must have same length (N)."

    total_size = len(df)
    remaining_ratio = 1.0 - train_ratio
    test_ratio = val_ratio = remaining_ratio / 2.0

    test_size = int(test_ratio * total_size)
    val_size  = int(val_ratio * total_size)
    train_size = total_size - test_size - val_size  # ensures sum = total_size

    # ---- 1) TEST = last part of ORIGINAL df (no shuffle) ----
    test_start = train_size + val_size
    test_df   = df.iloc[test_start:].reset_index(drop=True)
    test_traj = traj[test_start:]

    # ---- 2) REMAINING FRONT PART (for train + val) ----
    train_val_df   = df.iloc[:test_start]
    train_val_traj = traj[:test_start]

    # ---- 3) VAL = LAST PART of REMAINING (no shuffle) ----
    val_start = train_size
    val_df    = train_val_df.iloc[val_start:].reset_index(drop=True)
    val_traj  = train_val_traj[val_start:]

    # ---- 4) TRAIN = FRONT PART of REMAINING (no shuffle) ----
    train_df   = train_val_df.iloc[:val_start].reset_index(drop=True)
    train_traj = train_val_traj[:val_start]

    # print sizes for sanity check
    print(f"Total samples: {total_size}")
    print(f"Train size: {len(train_df)} ({len(train_traj)})")
    print(f"Val size: {len(val_df)} ({len(val_traj)})")
    print(f"Test size: {len(test_df)} ({len(test_traj)})")

    return train_df, val_df, test_df, train_traj, val_traj, test_traj

# Standardization for each feature type
def standardize_traj(train_traj, val_traj, test_traj, eps=1e-8):
    """
    Standardize trajectory arrays:
        (N_train, T, F), (N_val, T, F), (N_test, T, F)

    - Computes mean/std on TRAIN data only
    - Ignores NaN padding during mean/std computation
    - Applies identical transform to val/test
    - Returns standardized traj + statistics
    """

    # ---- Compute mean and std over TRAIN trajectories (ignoring NaNs) ----
    # Shape: (F,)
    mean = np.nanmean(train_traj, axis=(0, 1))  # mean across all time steps & samples
    std = np.nanstd(train_traj, axis=(0, 1))
    std = np.where(std < eps, 1.0, std)  # avoid division by zero

    # ---- Function to apply standardization to any trajectory set ----
    def apply_standardization(traj):
        traj_std = (traj - mean) / std
        # Keep NaNs as NaNs (important for masking)
        return traj_std

    # ---- Apply to splits ----
    train_traj_std = apply_standardization(train_traj)
    val_traj_std = apply_standardization(val_traj)
    test_traj_std = apply_standardization(test_traj)

    return train_traj_std, val_traj_std, test_traj_std, mean, std

def standardize_features(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, eps: float = 1e-8):
    """
    Standardize tabular features:
        X' = (X - mean) / std

    - mean/std computed on TRAIN only (no leakage)
    - std values that are ~0 are set to 1 to avoid division by zero
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < eps, 1.0, std)

    X_train_std = (X_train - mean) / std
    X_val_std   = (X_val   - mean) / std
    X_test_std  = (X_test  - mean) / std

    return X_train_std, X_val_std, X_test_std, mean, std

def standardize_targets(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, eps: float = 1e-8):
    """
    Standardize target values (dt):

        y' = (y - mean) / std

    - mean/std computed **only on train** (no leakage)
    - std < eps is set to 1 to avoid division by zero
    """
    mean = y_train.mean()
    std = y_train.std()
    if std < eps:
        std = 1.0

    y_train_std = (y_train - mean) / std
    y_val_std   = (y_val   - mean) / std
    y_test_std  = (y_test  - mean) / std

    return y_train_std, y_val_std, y_test_std, mean, std

# Main DataLoader Preparation Function
def prepare_dataloaders(scenarios: list, batch_size: int = 64, train_ratio: float = 0.8):
    """
    Prepare PyTorch DataLoaders for (X, traj, dt, plus4delay) from scenarios.

    Steps:
        1. scenarios -> df (tabular) + traj (N, T, F)
        2. split into train/val/test (df + traj)
        3. extract labels (dt, plus4delay), drop from X
        4. label-encode categorical features (fit on train only)
        5. standardize X (tabular) using train stats
        6. standardize trajectories using train stats
        7. standardize dt (targets) using train stats
        8. convert to torch tensors
        9. wrap into TensorDatasets & DataLoaders

    Returns:
        train_loader, val_loader, test_loader,
        encoders, x_mean, x_std, traj_mean, traj_std, y_mean, y_std
    """

    # 1) Build df and traj from scenarios
    df = scenarios_to_df(scenarios)          # must include dt + plus4delay
    traj = scenarios_to_traj(scenarios)      # shape (N, T, F)

    # 2) Split df + traj together
    train_df, val_df, test_df, train_traj, val_traj, test_traj = split_train_val_test(
        df, traj, train_ratio=train_ratio
    )

    # 3) Extract labels for each split
    X_train_df, y_train, Dminusy_train = extract_labels(train_df)
    X_val_df,   y_val,   Dminusy_val   = extract_labels(val_df)
    X_test_df,  y_test,  Dminusy_test  = extract_labels(test_df)

    # 4) Label-encode categorical features on training set only
    X_train_enc_df, encoders = label_encode_features(X_train_df)
    X_val_enc_df  = transform_with_label_encoders(X_val_df,  encoders)
    X_test_enc_df = transform_with_label_encoders(X_test_df, encoders)

    # 5) Convert X to NumPy
    X_train_np = X_train_enc_df.to_numpy(dtype=np.float32)
    X_val_np   = X_val_enc_df.to_numpy(dtype=np.float32)
    X_test_np  = X_test_enc_df.to_numpy(dtype=np.float32)

    # 5.5) Standardize X using train stats
    X_train_np, X_val_np, X_test_np, x_mean, x_std = standardize_features(
        X_train_np, X_val_np, X_test_np
    )

    # 6) Standardize trajectories (NaN-safe, train-only stats)
    train_traj, val_traj, test_traj, traj_mean, traj_std = standardize_traj(
        train_traj, val_traj, test_traj
    )

    # 7) Convert labels and plus4delay to NumPy
    y_train_np       = np.asarray(y_train, dtype=np.float32)
    y_val_np         = np.asarray(y_val,   dtype=np.float32)
    y_test_np        = np.asarray(y_test,  dtype=np.float32)

    Dminusy_train_np = np.asarray(Dminusy_train, dtype=np.float32)
    Dminusy_val_np   = np.asarray(Dminusy_val,   dtype=np.float32)
    Dminusy_test_np  = np.asarray(Dminusy_test,  dtype=np.float32)

    # 7.5) Standardize dt (targets) using train stats
    y_train_np, y_val_np, y_test_np, y_mean, y_std = standardize_targets(
        y_train_np, y_val_np, y_test_np
    )

    train_traj_np = train_traj.astype(np.float32)
    val_traj_np   = val_traj.astype(np.float32)
    test_traj_np  = test_traj.astype(np.float32)

    # 8) Convert to torch tensors
    X_train_t = torch.from_numpy(X_train_np)
    X_val_t   = torch.from_numpy(X_val_np)
    X_test_t  = torch.from_numpy(X_test_np)

    y_train_t = torch.from_numpy(y_train_np)
    y_val_t   = torch.from_numpy(y_val_np)
    y_test_t  = torch.from_numpy(y_test_np)

    Dminusy_train_t = torch.from_numpy(Dminusy_train_np)
    Dminusy_val_t   = torch.from_numpy(Dminusy_val_np)
    Dminusy_test_t  = torch.from_numpy(Dminusy_test_np)

    train_traj_t = torch.from_numpy(train_traj_np)  # (N_train, T, F)
    val_traj_t   = torch.from_numpy(val_traj_np)
    test_traj_t  = torch.from_numpy(test_traj_np)

    # 9) Build TensorDatasets: (X, traj, dt_std, Dminusy)
    train_ds = TensorDataset(X_train_t, train_traj_t, y_train_t, Dminusy_train_t)
    val_ds   = TensorDataset(X_val_t,   val_traj_t,   y_val_t,   Dminusy_val_t)
    test_ds  = TensorDataset(X_test_t,  test_traj_t,  y_test_t,  Dminusy_test_t)

    # 10) Build DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return (
        train_loader,
        val_loader,
        test_loader,
        encoders,
        x_mean,
        x_std,
        traj_mean,
        traj_std,
        y_mean,
        y_std,
    )

def epoch_runner(model, data_loader, criterion, optimizer=None, device="cpu"):
    """
    Run one epoch of training or evaluation.
    Args:
        model: PyTorch model
        data_loader: DataLoader providing (X, traj, y_std, Dminusy)
        criterion: loss function
        optimizer: if provided, perform training step; else eval only
        device: computation device
    Returns:
        avg_loss: average loss over the epoch
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for X_batch, traj_batch, y_true, _ in data_loader:
            X_batch = X_batch.to(device)
            traj_batch = traj_batch.to(device)
            y_true = y_true.to(device)

            # forward: model takes both X and traj
            y_pred = model(X_batch, traj_batch).view_as(y_true)

            loss = criterion(y_pred, y_true)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = y_true.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss

def evaluate_model(model, data_loader, y_mean, y_std, device="cpu"):
    """
    Evaluate the model on a DataLoader and compute delay-based metrics.
    Assumes:
        - loader yields (X, traj, y_std, Dminusy)
        - y_std is standardized dt: (dt - y_mean) / y_std
    Returns:
        metrics: dict from evaluate_delay (mae, mse, rmse, smape_dt, r2_dt, smape_delay, r2_delay)
    """
    model.eval()
    y_trues_std = []
    y_preds_std = []
    D_list = []

    with torch.no_grad():
        for X_batch, traj_batch, y_true_std, D_batch in data_loader:
            X_batch = X_batch.to(device)
            traj_batch = traj_batch.to(device)
            y_true_std = y_true_std.to(device)
            D_batch = D_batch.to(device)

            y_pred_std = model(X_batch, traj_batch).view_as(y_true_std)

            y_trues_std.append(y_true_std.cpu().numpy())
            y_preds_std.append(y_pred_std.cpu().numpy())
            D_list.append(D_batch.cpu().numpy())

    y_trues_std = np.concatenate(y_trues_std, axis=0).reshape(-1)
    y_preds_std = np.concatenate(y_preds_std, axis=0).reshape(-1)
    Dminusy = np.concatenate(D_list, axis=0).reshape(-1)

    # Destandardize dt
    y_true_dt = y_trues_std * y_std + y_mean
    y_pred_dt = y_preds_std * y_std + y_mean

    # IMPORTANT: evaluate_delay(ypred_dt, ytrue_dt, Dminusy)
    metrics = evaluate_delay(y_pred_dt, y_true_dt, Dminusy)
    return metrics

def fit(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, y_mean, y_std, device="cpu", verbose=False):
    """
    Train the model for a number of epochs, evaluating on validation set.
    Picks the epoch with best validation loss and returns test metrics at that epoch.

    Args:
        model: PyTorch model
        train_loader, val_loader, test_loader
        criterion: loss function (e.g., SmoothL1Loss on standardized dt)
        optimizer: optimizer
        num_epochs: number of epochs
        y_mean, y_std: scalars for destandardizing dt
        device: 'cpu' or 'cuda'
    Returns:
        train_losses: list of training losses per epoch
        val_losses: list of validation losses per epoch
        best_epoch: epoch index (1-based) with best val loss
        best_val_loss: best validation loss
        best_test_metrics: evaluate_delay dict on test set at best epoch
    """
    model.to(device)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_model_state = None
    best_test_metrics = None

    for epoch in range(1, num_epochs + 1):
        train_loss = epoch_runner(model, train_loader, criterion, optimizer, device)
        val_loss = epoch_runner(model, val_loader, criterion, optimizer=None, device=device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if verbose:
            print(
                f"Epoch {epoch}/{num_epochs} "
                f"- Train Loss: {train_loss:.4f} "
                f"- Val Loss: {val_loss:.4f}"
            )

        # Check if this is the best model so far (by val loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # Save model state
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # Evaluate on test set at this epoch
            best_test_metrics = evaluate_model(model, test_loader, y_mean, y_std, device=device)

    # Restore best model weights (optional but usually good)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if verbose:
        print(f"\nBest epoch: {best_epoch} with Val Loss = {best_val_loss:.4f}")
        print(f"Test metrics at best epoch: {best_test_metrics}")

    return train_losses, val_losses, best_epoch, best_val_loss, best_test_metrics

def main(scenarios, model_type, device, verbose=False):
    # Prepare DataLoaders
    (
        train_loader,
        val_loader,
        test_loader,
        encoders,
        x_mean, x_std,
        traj_mean, traj_std,
        y_mean, y_std,
    ) = prepare_dataloaders(
        scenarios, batch_size=32, train_ratio=0.8
    )

    model = Regressor(
        tabular_input_size=train_loader.dataset.tensors[0].shape[1],
        traj_input_size=train_loader.dataset.tensors[1].shape[2],
        traj_emb_size=320,
        hidden_size=1024,
        output_size=1,
        encoder_type=model_type,
    ).to(device)

    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    train_losses, val_losses, best_epoch, best_val_loss, best_test_metrics = fit(
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=15,
        y_mean=y_mean,
        y_std=y_std,
        device=device,
        verbose=verbose
    )
    del model

    best_test_metrics['best_epoch'] = best_epoch
    return best_test_metrics

# Plot all trajectories
def plot_trajectories(scenarios):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    for i, s in enumerate(scenarios[::max(1, len(scenarios)//100)]):  # Plot at most 10 trajectories for clarity
        traj = s['traj_focusing']
        plt.plot(traj[:, 0], traj[:, 1], label=f'Scenario {i+1}')
        # Mark start and end points
        plt.scatter(traj[0, 0], traj[0, 1], color='green', marker='X', s=100, label=f'Start {i+1}')
        plt.scatter(traj[-1, 0], traj[-1, 1], color='red', marker='X', s=100, label=f'End {i+1}')
    plt.title('Sample Trajectories')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid()
    plt.show()

