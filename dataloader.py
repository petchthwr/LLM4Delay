import pickle
import os
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
from functools import partial
from datasets import load_dataset

# Hugging Face Dataset Loading and Processing
def load_month_as_dict_list(repo_id, hf_token, month, split = "train", month_col = "month"):
    """
    Load one month from HF dataset and convert rows to list of dict.
    Returns:
        List[dict] where each dict has keys:
        i, t, label, F_f, P_f, P_m, P_t, P_n, X_f, X_a, X_p
    """
    ds = load_dataset(repo_id, split=split, token=hf_token) # Load dataset
    ds = ds.filter(lambda x: int(x[month_col]) == int(month)) # Filter by month
    output = []

    for row in ds:
        item = {}
        item["i"] = row.get("i") # Flight ID
        item["t"] = row.get("t") # Current Time
        item["label"] = {"y_dt": row.get("y_dt"),"y_delay": row.get("y_delay")} # Regression Labels
        item["F_f"] = row.get("F_f") # Tabular Flight Information Features
        item["P_f"] = row.get("P_f") # Flight Information Prompt
        item["P_m"] = row.get("P_m") # METAR Information Prompt
        item["P_t"] = row.get("P_t") # TAF Information Prompt
        item["P_n"] = row.get("P_n") # NOTAM Prompt
        item["X_f"] = np.asarray(row.get("X_f")) # Focusing Trajectory Data (Shape: T_foc, 9)
        item["X_a"] = None if row.get("X_a") in (None, []) else np.asarray(row.get("X_a")) # Active Trajectory Data (Shape: N_a, T_a, 9)
        item["X_p"] = None if row.get("X_p") in (None, []) else np.asarray(row.get("X_p")) # Prior Trajectory Data (Shape: N_p, T_p, 9)
        output.append(item)
    return output
def hf_to_llm4delay_scenario(scenarios):
    """
    Change dictionary keys to match the expected input format for LLM4Delay.
    i -> flight_id
    t -> current_time
    F_f -> flight_schedule
    flight_prompt = P_f + P_m + P_t
    P_no -> notam_prompt
    X_f -> traj_focusing
    X_a -> traj_active
    X_p -> traj_affecting
    z_focusing ->  zero shape of (320,) since hf_data doesn't have the actual pre-encoded trajectory data. We will set it to zeros for now.
    z_active, z_affecting are not included in the HF dataset, so we will set them to None for now.
    label[y_dt] = label['time_spend_in_airspace']
    label[y_delay] = label['delay_in_mins']
    label['dt_dummy'] = label[y_delay] - label['y_dt']
    label['actual_time_arrival'] = None # Not available
    label['delay_bool'] = None # Not available
    """
    output = []
    for s in scenarios:
        item = {}
        item["flight_id"] = s.get("i")
        item["current_time"] = s.get("t")
        item["label"] = s.get("label")
        item["flight_schedule"] = s.get("F_f")
        item["flight_prompt"] = s.get("P_f") + s.get("P_m") + s.get("P_t")
        item["notam_prompt"] = s.get("P_n")
        item["traj_focusing"] = s.get("X_f")
        item["traj_active"] = s.get("X_a")
        item["traj_affecting"] = s.get("X_p")
        item["z_focusing"] = np.zeros((320,)) # Placeholder for pre-encoded trajectory data
        item["z_active"] = None # Not available in HF dataset
        item["z_affecting"] = None # Not available in HF dataset
        y_dt = item["label"].get("y_dt")
        y_delay = item["label"].get("y_delay")
        item["label"]["time_spend_in_airspace"] = y_dt
        item["label"]["delay_in_mins"] = y_delay
        item["label"]["dt_dummy"] = y_delay - y_dt
        item["label"]["actual_time_arrival"] = None # Not available
        item["label"]["delay_bool"] = None # Not available
        output.append(item)
    return output
def load_scenarios_from_hf(month, hf_token):
    data = load_month_as_dict_list("petchthwr/ICNDelay", hf_token, month)
    scenarios = hf_to_llm4delay_scenario(data)
    return scenarios

# Our Own Pickle Loading and Processing
def load_scenarios(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    with open(file_path, 'rb') as f:
        scenarios = pickle.load(f)
    return scenarios
def drop_zero_padding(traj_T9, pad_value=0.0):
    """
    Remove padded timesteps (all zeros) from a (T,9) trajectory.
    """
    traj_T9 = np.asarray(traj_T9)
    if traj_T9.ndim != 2:
        raise ValueError(f"Expected (T,9), got {traj_T9.shape}")
    mask = ~(np.all(traj_T9 == pad_value, axis=1))
    return traj_T9[mask]

"""
LLMTIME : Implemenatation of the serialization method from:
N. Gruver, M. Finzi, S. Qiu, and A. G. Wilson,
“Large language models are zero-shot time series forecasters,”
in Proceedings of the 37th International Conference on Neural Information Processing Systems,
ser. NIPS ’23. Red Hook, NY, USA: Curran Associates Inc., 2023.
"""
def vec_num2repr(val, base, prec, max_val):
    """
    Convert numbers to a representation in a specified base with precision.

    Parameters:
    - val (np.array): The numbers to represent.
    - base (int): The base of the representation.
    - prec (int): The precision after the 'decimal' point in the base representation.
    - max_val (float): The maximum absolute value of the number.

    Returns:
    - tuple: Sign and digits in the specified base representation.

    Examples:
        With base=10, prec=2:
            0.5   ->    50
            3.52  ->   352
            12.5  ->  1250
    """
    base = float(base)
    bs = val.shape[0]
    sign = 1 * (val >= 0) - 1 * (val < 0)
    val = np.abs(val)
    max_bit_pos = int(np.ceil(np.log(max_val) / np.log(base)).item())

    before_decimals = []
    for i in range(max_bit_pos):
        digit = (val / base ** (max_bit_pos - i - 1)).astype(int)
        before_decimals.append(digit)
        val -= digit * base ** (max_bit_pos - i - 1)

    before_decimals = np.stack(before_decimals, axis=-1)

    if prec > 0:
        after_decimals = []
        for i in range(prec):
            digit = (val / base ** (-i - 1)).astype(int)
            after_decimals.append(digit)
            val -= digit * base ** (-i - 1)

        after_decimals = np.stack(after_decimals, axis=-1)
        digits = np.concatenate([before_decimals, after_decimals], axis=-1)
    else:
        digits = before_decimals
    return sign, digits
def serialize_arr(arr):
    """
    Serialize a 1D array (time series) into a compact string using digit representation.

    Args:
        arr: (T,) array-like (may contain NaNs)
        settings: dict with keys:
            base, prec, signed, fixed_length, max_val,
            time_sep, bit_sep, plus_sign, minus_sign,
            decimal_point, missing_str

    Returns:
        str
    """
    settings = {
        "base": 10,
        "prec": 3,
        "signed": True,
        "fixed_length": False,
        "max_val": 1e7,
        "time_sep": " ,",
        "bit_sep": " ",
        "plus_sign": "",
        "minus_sign": " -",
        "half_bin_correction": True,
        "decimal_point": "",
        "missing_str": " Nan",
    }

    arr = np.asarray(arr)

    # Safety checks
    valid = ~np.isnan(arr)
    if valid.any():
        assert np.all(np.abs(arr[valid]) <= settings["max_val"]), (
            f"abs(arr) must be <= max_val, but abs(arr)={np.abs(arr)}, max_val={settings['max_val']}"
        )

    if not settings["signed"]:
        if valid.any():
            assert np.all(arr[valid] >= 0), "unsigned arr must be >= 0"
        plus_sign = minus_sign = ""
    else:
        plus_sign = settings["plus_sign"]
        minus_sign = settings["minus_sign"]

    vnum2repr = partial(vec_num2repr, base=settings["base"], prec=settings["prec"], max_val=settings["max_val"])
    sign_arr, digits_arr = vnum2repr(np.where(np.isnan(arr), np.zeros_like(arr), arr))
    ismissing = np.isnan(arr)

    def tokenize(digits_1d: np.ndarray) -> str:
        return "".join([settings["bit_sep"] + str(b) for b in digits_1d])

    bit_strs = []
    for sign, digits, missing in zip(sign_arr, digits_arr, ismissing):
        if not settings["fixed_length"]:
            nonzero_indices = np.where(digits != 0)[0]
            if len(nonzero_indices) == 0:
                digits = np.array([0])
            else:
                digits = digits[nonzero_indices[0]:]

            prec = settings["prec"]
            if len(settings["decimal_point"]):
                digits = np.concatenate([digits[:-prec], np.array([settings["decimal_point"]]), digits[-prec:]])

        digits_str = tokenize(digits)
        sign_sep = plus_sign if sign == 1 else minus_sign

        if missing:
            bit_strs.append(settings["missing_str"])
        else:
            bit_strs.append(sign_sep + digits_str)

    out = settings["time_sep"].join(bit_strs) + settings["time_sep"]
    return out
def serialize_time_series(arr: np.ndarray, feat_sep: str = " | ") -> str:
    """
    Serialize (T,) or (T,D) using serialize_arr for consistency.

    - (T,)   -> serialize_arr(arr)
    - (T,D)  -> serialize each scalar with serialize_arr([scalar]) and join within timestep by feat_sep,
               then join timesteps by settings['time_sep'] and end with settings['time_sep'].
    """
    settings = {
        "base": 10,
        "prec": 3,
        "signed": True,
        "fixed_length": False,
        "max_val": 1e7,
        "time_sep": " ,",
        "bit_sep": " ",
        "plus_sign": "",
        "minus_sign": " -",
        "half_bin_correction": True,
        "decimal_point": "",
        "missing_str": " Nan",
    }
    arr = np.asarray(arr)

    if arr.ndim == 1:
        return serialize_arr(arr)

    if arr.ndim == 2:
        T, D = arr.shape
        step_strs = []
        for t in range(T):
            vals = []
            for d in range(D):
                s = serialize_arr(np.array([arr[t, d]]))
                # strip trailing time_sep from scalar serialization
                tsep = settings["time_sep"]
                if len(tsep) > 0 and s.endswith(tsep):
                    s = s[:-len(tsep)]
                vals.append(s)
            step_strs.append(feat_sep.join(vals))
        return settings["time_sep"].join(step_strs) + settings["time_sep"]

    raise ValueError(f"arr must be 1D or 2D. Got shape {arr.shape}")
def truncate_input(arr, tokenizer, max_context_length, downsample, feat_sep=" | "):
    """
    Truncate (T,D) time series so its serialized string fits within `max_context_length` tokens.

    - Keeps the most recent timesteps by dropping from the front.
    - Optionally downsamples along time (T).
    - Returns:
        truncated_str (str), remaining_length (int)
    """
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected (T,D), got shape {arr.shape}")
    if downsample is not None and downsample > 1:
        arr = arr[::downsample]

    # Serialize each timestep separately (cheap to drop from the front)
    T, D = arr.shape
    step_strs = [serialize_time_series(arr[t:t+1], feat_sep=feat_sep) for t in range(T)]
    step_lens = [len(tokenizer(s, add_special_tokens=False)["input_ids"]) for s in step_strs]
    total = sum(step_lens)

    if total <= max_context_length: # Fits already
        s = "".join(step_strs)
        remaining = max_context_length - total
        return s, remaining

    rem, k = total, 0 # Drop from the front until it fits
    while k < T and rem > max_context_length:
        rem -= step_lens[k]
        k += 1

    s = step_strs[-1] if k >= T else "".join(step_strs[k:])
    used = len(tokenizer(s, add_special_tokens=False)["input_ids"]) # Recompute exact remaining once (safe)
    return s, max_context_length - used
def build_prompt_from_trajs(focusing_traj_1T9, active_traj_NT9, affecting_traj_NT9, tokenizer, max_context_length, downsample=None, time_sep=" | ",):
    """
    Minimal wrapper using truncate_input() only.
    """
    # ---- focusing (T,9) -> (T,9) ----
    focusing_traj = np.asarray(focusing_traj_1T9)

    foc_str, remaining = truncate_input(
        focusing_traj, tokenizer, max_context_length, downsample, time_sep
    )

    parts = [
        f'Airspace is described using three trajectory types. '
        f'The focus trajectory: "{foc_str}"'
    ]

    if remaining <= 0:
        return ", ".join(parts)

    # ---- active trajectories ----
    if active_traj_NT9 is not None and remaining > 0:
        active_trajs = np.asarray(active_traj_NT9)
        parts.append("For other active trajectories:")
        for i, traj in enumerate(active_trajs):
            traj = drop_zero_padding(traj)
            if len(traj) == 0:
                continue

            traj_str, remaining = truncate_input(
                traj, tokenizer, remaining, downsample, time_sep
            )
            parts.append(f'"Act{i + 1}": "{traj_str}"')

            if remaining <= 0:
                return ", ".join(parts)

    # ---- affecting trajectories ----
    if affecting_traj_NT9 is not None and remaining > 0:
        affecting_trajs = np.asarray(affecting_traj_NT9)
        parts.append("For past or inactive trajectories that may still matter:")
        for i, traj in enumerate(affecting_trajs):
            traj = drop_zero_padding(traj)
            if len(traj) == 0:
                continue

            traj_str, remaining = truncate_input(
                traj, tokenizer, remaining, downsample, time_sep
            )
            parts.append(f'"Aff{i + 1}": "{traj_str}"')

            if remaining <= 0:
                return ", ".join(parts)

    parts.append(" The predicted total time spent in the airspace is: ")

    return ", ".join(parts)

# Unified Time-series Concatenation for TimeLLM and AutoTimes
def compress_multi_traj_time_concat(focusing_1T9, active_NT9=None, affecting_NT9=None, pad_value=0.0, downsample=None, add_sep_row=True):
    """
    Returns:
        X: (T_total, 9) concatenated along time axis.
    """

    # focusing: (1,T,9) -> (T,9)
    foc = np.asarray(focusing_1T9)
    if foc.ndim == 3 and foc.shape[0] == 1: # if (1,T,9) convert to (T,9) else assume (T,9)
        foc = foc[0]
    foc = np.asarray(foc)
    if downsample is not None and downsample > 1:
        foc = foc[::downsample]

    chunks = [foc]
    segment_length = 96
    if add_sep_row:
        pad_length = (segment_length - (foc.shape[0] % segment_length)) % segment_length
        chunks.append(np.full((pad_length, foc.shape[1]), pad_value)) # Add padding rows if needed

    def handle_batch(batch_NT9, pad_value=pad_value, downsample=downsample, add_sep_row=add_sep_row):
        if batch_NT9 is None:
            return
        batch_NT9 = np.asarray(batch_NT9)  # (N,T,9)
        for i in range(batch_NT9.shape[0]):
            traj = drop_zero_padding(batch_NT9[i], pad_value=pad_value)
            if traj.shape[0] == 0:
                continue
            if downsample is not None and downsample > 1:
                traj = traj[::downsample]
            chunks.append(traj)
            if add_sep_row:
                pad_length = (segment_length - (traj.shape[0] % segment_length)) % segment_length
                chunks.append(np.full((pad_length, traj.shape[1]), pad_value)) # Add padding rows if needed

    # append active then affecting (in that order)
    handle_batch(active_NT9, pad_value=pad_value, downsample=downsample, add_sep_row=add_sep_row)
    handle_batch(affecting_NT9, pad_value=pad_value, downsample=downsample, add_sep_row=add_sep_row)

    X = np.concatenate(chunks, axis=0)  # (T_total, 9)
    return X
def count_traj_and_construct_prompt(scenario, active_only=False):
    num_active_trajs = len(scenario['traj_active']) if scenario['traj_active'] is not None else 0
    num_affecting_trajs = len(scenario['traj_affecting']) if scenario['traj_affecting'] is not None else 0
    prompt_focus = 'The focusing trajectory is provided as the first trajectory.'
    prompt = f"This instance contains {num_active_trajs} active surrounding trajectories and {num_affecting_trajs} completed trajectories that may still matter."
    if active_only:
        prompt = f"This instance contains {num_active_trajs} active surrounding trajectories."
    return prompt_focus + " " + prompt

# LLM4Delay Pre-encoding Function with Different Encoders
def encode_trajectories(data, encoder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if encoder is None:
        traj_encoder = None
    elif encoder == 'infots':
        from baseline_enc.InfoTS.test import load_trained_infots, load_standardization_params
        traj_encoder = load_trained_infots('baseline_enc/InfoTS/models/RKSI_infots.pth', device=device)
        enc_mean, enc_std = load_standardization_params('baseline_enc/InfoTS/models/RKSI_infots_mean.npy',
                                                        'baseline_enc/InfoTS/models/RKSI_infots_std.npy')
        traj_encoder.enc_std = enc_std
        traj_encoder.enc_mean = enc_mean
    elif encoder == 'ts2vec':
        from baseline_enc.ts2vec.fit import load_trained_ts2vec, load_standardization_params
        traj_encoder = load_trained_ts2vec('baseline_enc/ts2vec/models/RKSI_ts2vec.pth', device=device)
        enc_mean, enc_std = load_standardization_params('baseline_enc/ts2vec/models/RKSI_ts2vec_mean.npy',
                                                        'baseline_enc/ts2vec/models/RKSI_ts2vec_std.npy')
        traj_encoder.enc_std = enc_std
        traj_encoder.enc_mean = enc_mean
    elif encoder == 'tcn_autoencoder':
        from baseline_enc.autoencoder.fit import load_trained_tcn_autoencoder, load_standardization_params
        traj_encoder = load_trained_tcn_autoencoder('baseline_enc/autoencoder/models/RKSI_tcn_autoencoder.pth').to(device)
        enc_mean, enc_std = load_standardization_params('baseline_enc/autoencoder/models/RKSI_tcn_autoencoder_mean.npy',
                                                        'baseline_enc/autoencoder/models/RKSI_tcn_autoencoder_std.npy')
        traj_encoder.enc_std = enc_std
        traj_encoder.enc_mean = enc_mean
        traj_encoder.eval()
    elif encoder == 'atscc':
        from preprocess.utils.scenario_utils import load_atscc_encoder
        traj_encoder = load_atscc_encoder("YOUR HUGGING FACE TOKEN HERE", device)
        traj_encoder.enc_std = None
        traj_encoder.enc_mean = None
        traj_encoder.eval()
    else:
        raise ValueError(f"Unknown encoder: {encoder}")

    if traj_encoder is not None:
        print(f'Pre-encoding trajectory representations using {encoder} encoder.')
        for i in tqdm(range(len(data))):
            s = data[i]
            focusing_traj = s['traj_focusing']  # Shape (T, D)
            active_traj = s['traj_active']  # Shape (N_act, T, D) or None
            affecting_traj = s['traj_affecting']  # Shape (N_aff, T, D) or None

            if active_traj is not None:
                active_traj = np.where(np.all(active_traj == 0, axis=-1, keepdims=True), np.nan, active_traj)
            if affecting_traj is not None:
                affecting_traj = np.where(np.all(affecting_traj == 0, axis=-1, keepdims=True), np.nan, affecting_traj)

            # Standardize focusing trajectory for ts2vec, imfots, and tcn_autoencoder
            if traj_encoder.enc_mean is not None and traj_encoder.enc_std is not None:
                focusing_traj = np.expand_dims(focusing_traj, axis=0) # Shape (1, T, D)
                focusing_traj = (focusing_traj - traj_encoder.enc_mean) / (traj_encoder.enc_std + 1e-8)
                if active_traj is not None:
                    active_traj = (active_traj - traj_encoder.enc_mean) / (traj_encoder.enc_std + 1e-8)
                if affecting_traj is not None:
                    affecting_traj = (affecting_traj - traj_encoder.enc_mean) / (traj_encoder.enc_std + 1e-8)

            # If encoder is tcn_autoencoder, convert to torch tensor
            if encoder == 'tcn_autoencoder' or encoder == 'atscc':
                focusing_traj = torch.tensor(focusing_traj, dtype=torch.float32).to(device)  # Shape (1, T, D)
                if active_traj is not None:
                    active_traj = torch.tensor(active_traj, dtype=torch.float32).to(device)  # Shape (N_act, T, D)
                if affecting_traj is not None:
                    affecting_traj = torch.tensor(affecting_traj, dtype=torch.float32).to(device)  # Shape (N_aff, T, D)

            # Encode trajectories
            if encoder == 'ts2vec':
                z_focusing = traj_encoder.encode(focusing_traj, encoding_window='full_series')  # Shape (1, E)
                z_active = traj_encoder.encode(active_traj, encoding_window='full_series') if active_traj is not None else None  # Shape (N_act, E)
                z_affecting = traj_encoder.encode(affecting_traj, encoding_window='full_series') if affecting_traj is not None else None  # Shape (N_aff, E)
            elif encoder == 'atscc':
                with torch.no_grad():
                    z_focusing = traj_encoder.instance_level_encode(focusing_traj.unsqueeze(0))  # Shape (T, D) -> (1, T, D) -> (1, E)
                    z_active = traj_encoder.instance_level_encode(active_traj) if active_traj is not None else None  # Shape (N_act, T, D) -> (N_act, E)
                    z_affecting = traj_encoder.instance_level_encode(affecting_traj) if affecting_traj is not None else None  # Shape (N_aff, T, D) -> (N_aff, E)
            else: # tcn_autoencoder and infots
                z_focusing = traj_encoder.encode(focusing_traj)  # Shape (1, E)
                z_active = traj_encoder.encode(active_traj) if active_traj is not None else None  # Shape (N_act, E)
                z_affecting = traj_encoder.encode(affecting_traj) if affecting_traj is not None else None  # Shape (N_aff, E)

            if encoder == 'tcn_autoencoder' or encoder == 'atscc':
                z_focusing = z_focusing.detach().cpu().numpy() # Shape (1, E)
                if z_active is not None:
                    z_active = z_active.detach().cpu().numpy()  # Shape (N_act, E)
                if z_affecting is not None:
                    z_affecting = z_affecting.detach().cpu().numpy()  # Shape (N_aff, E)

            data[i]['z_focusing'] = z_focusing  # Shape (1, E)
            data[i]['z_active'] = z_active if z_active is not None else None  # Shape (N_act, E)
            data[i]['z_affecting'] = z_affecting if z_affecting is not None else None  # Shape (N_aff, E)

    else:
        print('Using pre-encoded trajectory representations.')
        pass
    return data

# Dataloader Class
class DelayScenarioDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, task='dt_airspace', train=True, mean=None, std=None, ablation_tag=None, encoder=None):

        # Prepare Trajectory Information Based on Adaptation Method
        if ablation_tag == 'LLMTIME':
            # Prepare trajectory prompts using LLMTIME method
            print('Preparing LLMTIME trajectory prompts for each instance.')
            for i in tqdm(range(len(data))):
                s = data[i]
                traj_prompt = build_prompt_from_trajs(s['traj_focusing'],
                                                      s['traj_active'],
                                                      s['traj_affecting'],
                                                      tokenizer, max_context_length=2048, downsample=4, time_sep=" | ")
                data[i]['llmtime_prompt'] = traj_prompt # Using traj_prompt, keep z the same as pre-encoded; will not be used
        elif ablation_tag == 'TimeLLM' or ablation_tag == 'AutoTimes':
            # Prepare Unified Time-series Concatenation trajectory prompts
            for i in tqdm(range(len(data))):
                s = data[i]
                data[i]['traj_guide_prompt'] = count_traj_and_construct_prompt(s)
                compressed_traj = compress_multi_traj_time_concat(s['traj_focusing'],
                                                                  s['traj_active'],
                                                                  s['traj_affecting'],
                                                                  pad_value=0.0,
                                                                  downsample=None,
                                                                  add_sep_row=True)
                data[i]['z_focusing'] = compressed_traj # Store compressed trajectory in z_active for TimeLLM and AutoTimes
                data[i]['z_active'] = None
                data[i]['z_affecting'] = None
        else: # Pre-encoding for LLM4Delay
            data = encode_trajectories(data, encoder)

        self.data = data
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.task = task
        self.z_dim = data[0]['z_focusing'].shape[1] # Assuming all z_focusing have the same shape

        # Raise value for unknown ablation tag
        if ablation_tag not in [None, # Full Configuration
                                'exclude_flt_plan', 'exclude_notam', 'exclude_metar', 'exclude_taf', # Context Removal Configurations
                                'LLMTIME', 'TimeLLM', 'AutoTimes']:
            raise ValueError(f"Unknown ablation tag: {ablation_tag}")

        flight_prompt = [data[i]['flight_prompt'] for i in range(len(data))] # Full Configuration

        # Context Prompt Ablations
        if ablation_tag == 'exclude_flt_plan':
            print('Excluding Flight Information Prompt')
            flight_prompt = ['Flight plan unavailable; METAR' + data[i]['flight_prompt'].split('METAR')[1] + data[i]['flight_prompt'].split('METAR')[2] for i in range(len(data))] # Extract weather prompt
        if ablation_tag == 'exclude_metar': # TAF available METAR unavailable
            print('Excluding METAR Prompt')
            flt_plan_prompt = [data[i]['flight_prompt'].split('METAR')[0] for i in range(len(data))]
            weather_prompts = [data[i]['flight_prompt'].split('METAR')[1] + data[i]['flight_prompt'].split('METAR')[2] for i in range(len(data))] # Extract weather prompt
            taf_prompt = [weather_prompts[i].split('TAF in effect')[1] for i in range(len(data))]
            flight_prompt = [flt_plan_prompt[i] + '; METAR information unavailable; TAF in effect' + taf_prompt[i] for i in range(len(data))]
        if ablation_tag == 'exclude_taf':
            print('Excluding TAF Prompt')
            flt_plan_prompt = [data[i]['flight_prompt'].split('METAR')[0] for i in range(len(data))]
            weather_prompt = [data[i]['flight_prompt'].split('METAR')[1] + data[i]['flight_prompt'].split('METAR')[2] for i in range(len(data))]
            metar_prompt = [weather_prompt[i].split('TAF in effect')[0] for i in range(len(data))]
            flight_prompt = [flt_plan_prompt[i] + '; METAR' + metar_prompt[i] + '; TAF information unavailable' for i in range(len(data))]

        # NOTAM prompt is included by default in all configurations except the ones that explicitly exclude NOTAM
        if ablation_tag == 'exclude_notam':
            print('Excluding NOTAM prompt')
            flight_prompt = [flight_prompt[i] + ' ' + 'Unavailable NOTAM data' for i in range(len(data))]
        else: # all ablation tags except the ones that explicitly exclude NOTAM will include NOTAM prompt as default
            flight_prompt = [flight_prompt[i] + ' ' + data[i]['notam_prompt'] for i in range(len(data))] # Default case

        # Time-series Fusion Baselines
        if ablation_tag == 'LLMTIME':
            flight_prompt = [flight_prompt[i] + ' ' + data[i]['llmtime_prompt'] for i in range(len(data))]
        if ablation_tag == 'TimeLLM' or ablation_tag == 'AutoTimes':
            flight_prompt = [flight_prompt[i] + ' ' + data[i]['traj_guide_prompt'] for i in range(len(data))]

        self.system_prompt = ''
        self.flight_prompt_raw = [self.system_prompt + " " + flight_prompt[i] for i in range(len(data))]
        self.flight_prompt = [self.tokenizer(flight_prompt, return_tensors='pt', truncation=True, add_special_tokens=False) for flight_prompt in self.flight_prompt_raw]

        # Trajectory Components: Remarks they have been pre-encoded as vectors representing the trajectories instead of full trajectories
        self.z_focusing = [data[i]['z_focusing'] for i in range(len(data))] # LLM4Delay: (1, 320), TimeLLM/AutoTimes: (T_total, 9)
        self.z_active = [data[i]['z_active'] if data[i]['z_active'] is not None else np.full((1, self.z_dim), np.nan) for i in range(len(data))] # LLM4Delay: (N_ac, 320), TimeLLM/AutoTimes: None
        self.z_affecting = [data[i]['z_affecting'] if data[i]['z_affecting'] is not None else np.full((1, self.z_dim), np.nan) for i in range(len(data))] # LLM4Delay: (N_af, 320), TimeLLM/AutoTimes: None

        # Convert to tensors
        self.z_focusing = [torch.tensor(z_focusing, dtype=torch.float32) for z_focusing in self.z_focusing]
        self.z_active = [torch.tensor(z_active, dtype=torch.float32) for z_active in self.z_active]
        self.z_affecting = [torch.tensor(z_affecting, dtype=torch.float32) for z_affecting in self.z_affecting]

        # Label processing
        if self.task == 'dt_airspace':
            self.label = [float(data[i]['label']['time_spend_in_airspace']) for i in range(len(data))]
        else:
            raise ValueError(f"Unknown task: {self.task}")

        # Standardization for regression tasks
        if self.task in ['dt_airspace', 'acc_delay']:
            if train:
                self.mean = np.mean(self.label)
                self.std = np.std(self.label)
                self.label = [(x - self.mean) / self.std for x in self.label]
            else:
                if mean is None or std is None:
                    raise ValueError("Mean and std must be provided for test set.")
                self.mean = mean
                self.std = std
                self.label = [(x - self.mean) / self.std for x in self.label]
        else:
            self.mean = None
            self.std = None

        # There will be T_dummy in the data for calculating delay in minutes
        self.dt_dummy = [float(data[i]['label']['dt_dummy']) for i in range(len(data))] # Predicted Delay = dt_dummy + time_spend_in_airspace
        self.min_delay = [float(data[i]['label']['delay_in_mins']) for i in range(len(data))] # Actual Delay in minutes

        for i in range(len(data)): # Save memory by removing raw trajectory data
            if 'traj_focusing' in data[i]:
                del data[i]['traj_focusing']
            if 'traj_active' in data[i]:
                del data[i]['traj_active']
            if 'traj_affecting' in data[i]:
                del data[i]['traj_affecting']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.flight_prompt[idx],
                self.z_focusing[idx],
                self.z_active[idx],
                self.z_affecting[idx],
                self.label[idx],
                self.tokenizer.pad_token_id, # Pad token id for padding in collate_fn
                self.dt_dummy[idx],
                self.min_delay[idx]
                )

# Collate function
def collate_fn(batch):
    flight_prompts, z_focusings, z_actives, z_affectings, labels, pad_token_id, dt_dummy, min_delay = zip(*batch)

    # Pad and stack tokenized flight prompts
    flight_prompts = torch.nn.utils.rnn.pad_sequence([flight_prompt['input_ids'].squeeze(0) for flight_prompt in flight_prompts], batch_first=True, padding_value=pad_token_id[0]) # Shape: (B, L)

    # Trajectory Components
    # Input Shapes:
    # LLM4Delay: Shpae: z_focusing: (1, E), Shape: z_active: (N_ac, E), Shape: z_affecting: (N_af, E)
    # TimeLLM / AutoTimes: Shape: z_focusing: (T_total, 9), z_active: None, z_affecting: None
    # Target: Shape: (B, E), Shape: (B, N_ac, E), Shape: (B, N_af, E)
    # If all tensors have same shape (T, E), stacking works
    if all(z.ndim == 2 and z.shape == z_focusings[0].shape for z in z_focusings): # and z_focusings[0].shape[0] == 1 for LLM4Delay
        # If lengths are the same, stack directly works for LLM4Delay
        z_focusings = torch.stack(z_focusings, dim=0)  # (B, 1, E)
    else:
        # Variable-length sequences (TimeLLM / AutoTimes)
        z_focusings = torch.nn.utils.rnn.pad_sequence(
            z_focusings, batch_first=True, padding_value=0.0
        )  # (B, T_max, E)
    z_actives = torch.nn.utils.rnn.pad_sequence(z_actives, batch_first=True, padding_value=np.nan) # (B, N_ac, E)
    z_affectings = torch.nn.utils.rnn.pad_sequence(z_affectings, batch_first=True, padding_value=np.nan) # (B, N_af, E)

    # Make dictionary for trajectory components
    z_components = {
        'focusing': z_focusings, # Shape: Embedding (B, E) for LLM4Delay; Raw Trajectory (B, T_max, 9) for TimeLLM / AutoTimes
        'active': z_actives, # For LLM4Delay, active will be tensors of shape (B, N_ac, E) with NaN values if no active trajectories
        'affecting': z_affectings # For LLM4Delay, affecting will be tensors of shape (B, N_af, E) with NaN values if no affecting trajectories
    } # For TimeLLM / AutoTimes, active and affecting will be tensors of shape (B, 1, E) with all NaN values

    # Check type of label and convert to tensor accordingly it is either float or boolean
    if isinstance(labels[0], float):
        labels = torch.tensor(labels, dtype=torch.float32)
    elif isinstance(labels[0], bool):
        labels = torch.tensor(labels, dtype=torch.bool)
    else:
        raise ValueError("Label type not supported")

    dt_dummy = torch.tensor(dt_dummy, dtype=torch.float32)
    min_delay = torch.tensor(min_delay, dtype=torch.float32)

    return flight_prompts, z_components, (labels, dt_dummy, min_delay)

# Create dataloader function
def create_dataloader(data, tokenizer, batch_size=32, shuffle=True, task='dt_airspace', train=True, mean=None, std=None, ablation_tag=None, traj_encoder=None):
    dataset = DelayScenarioDataset(data, tokenizer, task=task, train=train, mean=mean, std=std, ablation_tag=ablation_tag, encoder=traj_encoder)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
    return loader

def prepare_dataloaders(data, tokenizer, batch_size=32, task='dt_airspace', train_ratio=0.8, seed=42, ablation_tag=None, traj_encoder=None):
    total_size = len(data)

    # Calculate split sizes
    remaining_ratio = 1.0 - train_ratio
    test_ratio = val_ratio = remaining_ratio / 2

    test_size = int(test_ratio * total_size)
    val_size = int(val_ratio * total_size)
    train_size = total_size - test_size - val_size

    # Step 1: Fix test set (last test_size samples)
    test_data = data[-test_size:]
    train_val_data = data[:-test_size]  # Remaining data for train and val

    # Step 2: Fix Val set like test set (last val_size samples of remaining data)
    val_data = train_val_data[-val_size:]
    train_data = train_val_data[:-val_size] # Remaining data for training

    # Check the final current time of train_data
    train_final_times = [datetime.strptime(s['current_time'], "%Y-%m-%d %H:%M:%S") for s in train_data]
    print(f"Train data final current time range: {min(train_final_times)} to {max(train_final_times)}")

    # Check the final current time of val_data
    val_final_times = [datetime.strptime(s['current_time'], "%Y-%m-%d %H:%M:%S") for s in val_data]
    print(f"Val data final current time range: {min(val_final_times)} to {max(val_final_times)}")

    # Check the final current time of test_data
    test_final_times = [datetime.strptime(s['current_time'], "%Y-%m-%d %H:%M:%S") for s in test_data]
    print(f"Test data final current time range: {min(test_final_times)} to {max(test_final_times)}")

    # Step 3: Create dataloaders
    train_loader = create_dataloader(train_data, tokenizer, batch_size=batch_size, shuffle=True, task=task, train=True,
                                     ablation_tag=ablation_tag, traj_encoder=traj_encoder)

    val_loader = create_dataloader(val_data, tokenizer, batch_size=batch_size, shuffle=False, task=task, train=False,
                                   mean=train_loader.dataset.mean, std=train_loader.dataset.std,
                                   ablation_tag=ablation_tag, traj_encoder=traj_encoder)

    test_loader = create_dataloader(test_data, tokenizer, batch_size=batch_size, shuffle=False, task=task, train=False,
                                    mean=train_loader.dataset.mean, std=train_loader.dataset.std,
                                    ablation_tag=ablation_tag, traj_encoder=traj_encoder)

    # Debug info
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_data)}")
    print(f"Train mean: {train_loader.dataset.mean}, Train std: {train_loader.dataset.std}")

    return train_loader, val_loader, test_loader

def select_samples_for_test_only(data, num_samples=20):
    total_size = len(data)
    indices = np.linspace(0, total_size - 1, num=num_samples, dtype=int)
    selected_data = [data[i] for i in indices]
    return selected_data

def prepare_dataloaders_for_test_only(data, tokenizer, batch_size=32, task='dt_airspace', train_ratio=0.8, seed=42, ablation_tag=None, traj_encoder=None, mean=None, std=None):
    total_size = len(data)

    # Calculate split sizes
    remaining_ratio = 1.0 - train_ratio
    test_ratio = remaining_ratio / 2
    test_size = int(test_ratio * total_size)
    test_data = data[-test_size:]

    test_data = select_samples_for_test_only(test_data, num_samples=40)

    # Check the final current time of test_data
    test_final_times = [datetime.strptime(s['current_time'], "%Y-%m-%d %H:%M:%S") for s in test_data]
    print(f"Test data final current time range: {min(test_final_times)} to {max(test_final_times)}")

    # Create dataloader
    test_loader = create_dataloader(test_data, tokenizer, batch_size=batch_size, shuffle=False, task=task, train=False,
                                    mean=mean, std=std,
                                    ablation_tag=ablation_tag, traj_encoder=traj_encoder)

    return test_loader

def prepare_test_loader_for_realtime(data, tokenizer, mean, std, batch_size=32, task='dt_airspace', ablation_tag=None, traj_encoder=None):
    test_loader = create_dataloader(data, tokenizer, batch_size=batch_size, shuffle=False, task=task, train=False, mean=mean, std=std, ablation_tag=ablation_tag, traj_encoder=traj_encoder)
    print(f"Test size: {len(data)}")
    return test_loader


