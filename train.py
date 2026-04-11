from dataloader import *
from model.DelayLLM import LlamaDelayPredModel
import warnings
import time
import os
from accelerate import Accelerator
warnings.simplefilter(action='ignore', category=FutureWarning)

def reproducibility(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def regression_metrics(y_true, y_pred, eps=1e-8):
    mae = torch.mean(torch.abs(y_true - y_pred))

    mse = torch.mean((y_true - y_pred) ** 2)
    rmse = torch.sqrt(mse)

    # R2 score
    y_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_mean) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_res / (ss_tot + eps)

    # MAPE
    mape = torch.mean(torch.abs((y_true - y_pred) / (y_true + eps))) * 100

    # SMAPE
    smape = 100 * torch.mean(2 * torch.abs(y_pred - y_true) / (torch.abs(y_true) + torch.abs(y_pred) + eps))

    return {
        'MAE': mae.item(),
        'MSE': mse.item(),
        'RMSE': rmse.item(),
        'R2': r2.item(),
        'MAPE': mape.item(),
        'SMAPE': smape.item(),
    }

def epoch_runner(model, train_loader, val_loader, criterion, optimizer, accelerator, task='dt_airspace', mean=None, std=None):
    device = accelerator.device

    def maybe_rescale(tensor):
        return tensor * std + mean if task == 'dt_airspace' and mean is not None and std is not None else tensor

    def run_epoch(loader, train_mode=True):
        losses = 0.0
        y_pred, y_real, dt_dummy, delay = [], [], [], []

        model.train() if train_mode else model.eval()
        context = torch.enable_grad() if train_mode else torch.no_grad()

        with context:
            for flight_prompts, z, (y, dtd, d) in loader:
                optimizer.zero_grad()
                flight_prompts = flight_prompts.to(device)
                z = {k: v.to(device) for k, v in z.items()}
                y = y.to(device).float()

                out = model(flight_prompts, z)
                loss = criterion(out.float(), y)
                losses += loss.item()

                if train_mode:
                    accelerator.backward(loss)
                    optimizer.step()

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

        return losses / len(loader), y_real, y_pred, delay_real, delay_pred

    # ---- Train ----
    train_loss, y_train_real, y_train_pred, delay_train_real, delay_train_pred = run_epoch(train_loader, train_mode=True)
    train_metrics = regression_metrics(delay_train_real, delay_train_pred)
    dt_train_metrics = regression_metrics(y_train_real, y_train_pred)
    dt_train_metrics = {f"dt_{k}": v for k, v in dt_train_metrics.items()}
    train_metrics.update(dt_train_metrics)

    # ---- Validation ----
    val_loss, y_val_real, y_val_pred, delay_val_real, delay_val_pred = run_epoch(val_loader, train_mode=False)
    val_metrics = regression_metrics(delay_val_real, delay_val_pred)
    dt_val_metrics = regression_metrics(y_val_real, y_val_pred)
    dt_val_metrics = {f"dt_{k}": v for k, v in dt_val_metrics.items()}
    val_metrics.update(dt_val_metrics)

    return train_loss, val_loss, train_metrics, val_metrics

def test_model(model, test_loader, accelerator, task='dt_airspace', mean=None, std=None):
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

    device = accelerator.device

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

    test_metrics = regression_metrics(delay_real, delay_pred)
    dt_test_metrics = regression_metrics(y_real, y_pred)
    dt_test_metrics = {f"dt_{k}": v for k, v in dt_test_metrics.items()}
    test_metrics.update(dt_test_metrics)

    return test_metrics

def model_runner(model_name, token, data, batch_size, train_data_size, learning_rate, weight_decay, num_epochs, verbose, test_every, seed=0, ablation_tag=None, traj_encoder=None):
    reproducibility(seed)

    model_ablation_tag = None
    dataset_ablation_tag = None

    # Context Removal Settings
    if ablation_tag == 'exclude_focusing':
        model_ablation_tag = 'exclude_focusing'
    elif ablation_tag == 'exclude_active':
        model_ablation_tag = 'exclude_active'
    elif ablation_tag == 'exclude_affecting':
        model_ablation_tag = 'exclude_affecting'
    elif ablation_tag == 'exclude_trajectory':
        model_ablation_tag = 'exclude_trajectory'
    elif ablation_tag == 'exclude_text':
        model_ablation_tag = 'exclude_text'
    elif ablation_tag == 'exclude_flt_plan':
        dataset_ablation_tag = 'exclude_flt_plan'
    elif ablation_tag == 'exclude_notam':
        dataset_ablation_tag = 'exclude_notam'
    elif ablation_tag == 'exclude_metar':
        dataset_ablation_tag = 'exclude_metar'
    elif ablation_tag == 'exclude_taf':
        dataset_ablation_tag = 'exclude_taf'
    # Data Fusion Baselines
    elif ablation_tag == 'LLMTIME':
        dataset_ablation_tag = 'LLMTIME'
        model_ablation_tag = 'LLMTIME'
    elif ablation_tag == 'TimeLLM':
        dataset_ablation_tag = 'TimeLLM'
        model_ablation_tag = 'TimeLLM'
    elif ablation_tag == 'AutoTimes':
        dataset_ablation_tag = 'AutoTimes'
        model_ablation_tag = 'AutoTimes'

    model = LlamaDelayPredModel(model_name, token, ablation_tag=model_ablation_tag)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = model.tokenizer

    model.to('cpu') # move model to cpu first
    train_loader, val_loader, test_loader = prepare_dataloaders(data, tokenizer, batch_size=batch_size, train_ratio=train_data_size, seed=seed, ablation_tag=dataset_ablation_tag, traj_encoder=traj_encoder)
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.to(device)

    # Prepare accelerator
    # For TimeLLM and Autotimes use float16 precision
    if model_ablation_tag in ['TimeLLM', 'AutoTimes']:
        accelerator = Accelerator(mixed_precision="fp16")
    else:
        accelerator = Accelerator()
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )
    device = accelerator.device
    model.to(device)

    loss_log = []
    test_matrices_log = []
    model_state_log = []
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        train_loss, val_loss, train_metrics, val_metrics = epoch_runner(model, train_loader, val_loader, criterion, optimizer, accelerator, mean=train_loader.dataset.mean, std=train_loader.dataset.std)
        loss_log.append((train_loss, val_loss))
        epoch_end_time = time.time()
        time_elapsed = (epoch_end_time - epoch_start_time) / 60.0
        model_state_log.append(accelerator.get_state_dict(model))

        if verbose:
            print(f"Epoch {epoch + 1} taken {time_elapsed:.2f} minutes")
            print(f"Train Metrics: Train Loss: {train_loss:.4f}, MAE: {train_metrics['MAE']:.4f}, MSE: {train_metrics['MSE']:.4f}, RMSE: {train_metrics['RMSE']:.4f}, "
                  f"SMAPE: {train_metrics['SMAPE']:.4f}, R2: {train_metrics['R2']:.5f}, dt_SMAPE: {train_metrics['dt_SMAPE']:.4f}, dt_R2: {train_metrics['dt_R2']:.4f}")
            print(f"Validation Metrics: Val Loss: {val_loss:.4f}, MAE: {val_metrics['MAE']:.4f}, MSE: {val_metrics['MSE']:.4f}, RMSE: {val_metrics['RMSE']:.4f}, "
                  f"SMAPE: {val_metrics['SMAPE']:.4f}, R2: {val_metrics['R2']:.5f}, dt_SMAPE: {val_metrics['dt_SMAPE']:.4f}, dt_R2: {val_metrics['dt_R2']:.4f}")

        # Test every test_every epochs
        if epoch + 1 % test_every == 0 or test_every == 1:
            test_metrics = test_model(model, test_loader, accelerator, mean=train_loader.dataset.mean, std=train_loader.dataset.std)
            test_matrices_log.append(test_metrics)
            if verbose:
                print(f"Test Metrics: MAE: {test_metrics['MAE']:.4f}, MSE: {test_metrics['MSE']:.4f}, RMSE: {test_metrics['RMSE']:.4f}, "
                      f"SMAPE: {test_metrics['SMAPE']:.4f}, R2: {test_metrics['R2']:.5f}, dt_SMAPE: {test_metrics['dt_SMAPE']:.4f}, dt_R2: {test_metrics['dt_R2']:.4f}")

    return model, loss_log, test_matrices_log, train_loader.dataset.mean, train_loader.dataset.std, model_state_log
