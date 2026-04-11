import os
import pickle
import random
from metar import Metar
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import numpy as np
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

DAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2,
    "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
}

def load_scenarios(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    with open(file_path, 'rb') as f:
        scenarios = pickle.load(f)
    return scenarios

# Data Cleaning and Feature Extraction
def attach_weather_from_metar(description: str, flight_schedule: dict) -> dict:
    """
    - Extract METAR from description
    - Extract METAR timestamp
    - Parse with python-metar
    - Replace None weather values with 0
    - Add features to flight_schedule dict
    """
    start_token = "METAR in effect:"
    end_token = "TAF in effect:"

    # --- 1. Extract METAR block ---
    try:
        start_idx = description.index(start_token) + len(start_token)
    except ValueError:
        return flight_schedule  # no METAR found

    try:
        end_idx = description.index(end_token, start_idx)
    except ValueError:
        end_idx = len(description)

    metar_block = description[start_idx:end_idx].strip()
    if not metar_block:
        return flight_schedule

    # --- 2. Extract timestamp (YYYYMMDDHHMM) ---
    parts = metar_block.split()
    metar_time_str = None

    if len(parts) > 0 and parts[0].isdigit() and len(parts[0]) == 12:
        metar_time_str = parts[0]
        parts = parts[1:]

    # Strip trailing "="
    metar_str = " ".join(parts).rstrip("=").strip()

    updated = flight_schedule.copy()
    updated["metar_time_utc"] = metar_time_str
    updated["metar_raw"] = metar_str

    # --- 3. Parse with correct year & month ---
    try:
        if metar_time_str:
            year = int(metar_time_str[:4])
            month = int(metar_time_str[4:6])
            report = Metar.Metar(metar_str, year=year, month=month)
        else:
            report = Metar.Metar(metar_str)
    except Exception as e:
        updated["metar_parse_error"] = str(e)
        return updated

    # Replace None with 0
    def safe_val(v):
        if v is None:
            return 0
        try:
            return v.value() if hasattr(v, "value") else v
        except Exception:
            return 0

    # --- 4. Extract weather features (None → 0) ---
    updated["wx_temp_c"]        = safe_val(report.temp)
    updated["wx_dewpoint_c"]    = safe_val(report.dewpt)
    updated["wx_pressure_hpa"]  = safe_val(report.press)
    updated["wx_visibility_m"]  = safe_val(report.vis)
    updated["wx_wind_dir_deg"]  = safe_val(report.wind_dir)
    updated["wx_wind_speed"]    = safe_val(report.wind_speed)
    updated["wx_wind_gust"]     = safe_val(report.wind_gust)

    updated["wx_temp_dew_diff"] = (
        updated["wx_temp_c"] - updated["wx_dewpoint_c"]
        if updated["wx_temp_c"] and updated["wx_dewpoint_c"]
        else 0
    )

    # --- 5. Weather condition flags ---
    wx_codes = ""
    if getattr(report, "weather", None):
        try:
            wx_codes = "".join(w.code for w in report.weather if getattr(w, "code", None))
        except Exception:
            wx_codes = ""

    updated["wx_rain"]    = 1 if "RA" in wx_codes else 0
    updated["wx_snow"]    = 1 if "SN" in wx_codes else 0
    updated["wx_fog"]     = 1 if "FG" in wx_codes else 0
    updated["wx_thunder"] = 1 if "TS" in wx_codes else 0

    return updated

def convert_datetime_features(f: dict) -> dict:
    """
    Convert datetime-related string features to numeric ML features,
    and remove the original datetime string fields.
    """
    out = f.copy()

    # --- Convert sched_time_utc ---
    if "sched_time_utc" in out and out["sched_time_utc"]:
        dt = datetime.fromisoformat(out["sched_time_utc"])
        out["sched_hour"] = dt.hour
        out["sched_minute"] = dt.minute
        out["sched_day"] = dt.day
        out["sched_month"] = dt.month
        out["sched_weekday"] = dt.weekday()
        out["sched_sec_of_day"] = dt.hour * 3600 + dt.minute * 60 + dt.second

    # --- Convert actual_entry_time ---
    if "actual_entry_time" in out and out["actual_entry_time"]:
        et = datetime.fromisoformat(out["actual_entry_time"])
        out["entry_hour"] = et.hour
        out["entry_minute"] = et.minute
        out["entry_day"] = et.day
        out["entry_month"] = et.month
        out["entry_weekday"] = et.weekday()
        out["entry_sec_of_day"] = et.hour * 3600 + et.minute * 60 + et.second

    # --- Convert date ---
    if "date" in out and out["date"]:
        d = datetime.fromisoformat(out["date"])
        out["date_day"] = d.day
        out["date_month"] = d.month
        out["date_weekday"] = d.weekday()

    # --- Convert day_of_week string to index ---
    if "day_of_week" in out and out["day_of_week"]:
        dow = out["day_of_week"].lower()
        out["day_of_week_idx"] = DAY_MAP.get(dow, -1)

    # --- Remove original string datetime fields ---
    for key in ["sched_time_utc", "actual_entry_time", "date", "day_of_week"]:
        out.pop(key, None)

    return out

def clean_flight_dict_for_ml(base: dict) -> dict:
    """
    Keep *all* base features exactly as used in the prompt.
    Add weather features.
    Remove only labels (delay, ATE).
    """

    clean = base.copy()  # keep everything you use in prompts

    # --- Remove leakage labels ---
    for key in ["delay_gt_15min", "delay_mins", "airspace_ate_min", "act_time_utc"]:
        if key in clean:
            clean.pop(key)

    # --- Weather fields: ensure all exist and are numeric ---
    def wx(name):
        return base.get(name, 0) or 0

    clean.update({
        "wx_temp_c":         wx("wx_temp_c"),
        "wx_dewpoint_c":     wx("wx_dewpoint_c"),
        "wx_temp_dew_diff":  wx("wx_temp_dew_diff"),
        "wx_pressure_hpa":   wx("wx_pressure_hpa"),
        "wx_visibility_m":   wx("wx_visibility_m"),
        "wx_wind_dir_deg":   wx("wx_wind_dir_deg"),
        "wx_wind_speed":     wx("wx_wind_speed"),
        "wx_wind_gust":      wx("wx_wind_gust"),
        "wx_rain":           wx("wx_rain"),
        "wx_snow":           wx("wx_snow"),
        "wx_fog":            wx("wx_fog"),
        "wx_thunder":        wx("wx_thunder"),
    })

    # --- Remove fields you want deleted ---
    REMOVE_KEYS = [
        "aircraft_registration",
        "callsign_code_iata",
        "dep_code_iata",
        "dep_name_english",
        "dest_altitude",
        "dest_code_iata",
        "dest_code_icao",
        "dest_lat",
        "dest_lon",
        "dest_name_english",
        "id",
        "metar_raw",
        "metar_time_utc",
        "metar_parse_error",
    ]

    for key in REMOVE_KEYS:
        clean.pop(key, None)

    clean = convert_datetime_features(clean)
    return clean

def count_num_trajectory(scenario: dict) -> int:
    """
    Count the number of trajectories in a scenario.
    """
    count = 1  # focusing trajectory always exists

    if scenario.get("traj_active") is not None:
        count += scenario["traj_active"].shape[0]

    return count

def scenarios_to_df(scenarios: list) -> pd.DataFrame:
    """
    Convert a list of scenarios to a pandas DataFrame,
    where each row corresponds to a flight in a scenario.
    """
    records = []

    for s in scenarios:
        # print current time of the scenario for debugging
        #print(f"Processing scenario with current_time: {s.get('current_time', 'N/A')}")
        # Feature fields
        flight_schedule_dict = s['flight_schedule']
        description = s['flight_prompt']
        flight_schedule_dict = clean_flight_dict_for_ml(attach_weather_from_metar(description, flight_schedule_dict))

        # Label fields Note that: delay = dt_dummy + dt
        plus4delay = float(s['label']['dt_dummy'])
        dt = float(s['label']['time_spend_in_airspace'])
        flight_schedule_dict['plus4delay'] = plus4delay
        flight_schedule_dict['dt'] = dt

        # Include number of trajectories as a feature
        flight_schedule_dict['num_trajectories'] = count_num_trajectory(s)

        records.append(flight_schedule_dict)
    df = pd.DataFrame.from_records(records)
    return df

# Data Preparation for ML
def split_train_val_test(df: pd.DataFrame, train_ratio=0.8):
    """
    Split a pandas DataFrame into train/val/test without shuffling.

    - train_ratio: fraction of data for training
    - remaining is split equally into val and test
    """
    total_size = len(df)

    remaining_ratio = 1.0 - train_ratio
    test_ratio = val_ratio = remaining_ratio / 2.0

    test_size = int(test_ratio * total_size)
    val_size = int(val_ratio * total_size)
    train_size = total_size - test_size - val_size

    test_df = df.iloc[train_size + val_size:].reset_index(drop=True)
    train_val_df = df.iloc[:train_size + val_size]

    train_df = train_val_df.iloc[:train_size].reset_index(drop=True)
    val_df = train_val_df.iloc[train_size:].reset_index(drop=True)

    print(f"Split sizes → Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df

def extract_labels(df: pd.DataFrame):
    """
    Extract labels from the DataFrame.
    Labels are 'dt'
    plus4delay is D-y for Delay = dt + plus4delay
    """
    y = df['dt'].values
    Dminusy = df['plus4delay'].values
    X = df.drop(columns=['dt', 'plus4delay'])
    return X, y, Dminusy

def label_encode_features(X: pd.DataFrame):
    """
    Label-encode all object/string columns in X.
    Returns:
        X_encoded: DataFrame with label-encoded columns
        encoders: dict[col_name -> LabelEncoder]
    """
    X_encoded = X.copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

    return X_encoded, encoders

def transform_with_label_encoders(X: pd.DataFrame, encoders: dict):
    """
    Apply fitted label encoders to new data (val/test).
    Unseen categories are mapped to -1.
    """
    X_encoded = X.copy()

    for col, le in encoders.items():
        if col not in X_encoded.columns:
            continue

        # map each value to existing class or -1 if unseen
        def encode_val(v):
            v = str(v)
            if v in le.classes_:
                return le.transform([v])[0]
            else:
                return -1

        X_encoded[col] = X_encoded[col].astype(str).map(encode_val)

    return X_encoded

def prepare_data_for_ml(scenarios: list):
    """
    Convert scenarios to DataFrame, split into train/val/test,
    extract labels, and label-encode categorical features.
    Returns:
        (X_train_enc, y_train, Dminusy_train,
         X_val_enc,   y_val,   Dminusy_val,
         X_test_enc,  y_test,  Dminusy_test,
         encoders)
    """
    df = scenarios_to_df(scenarios)

    # Split into train/val/test (make sure this uses a fixed seed internally)
    train_df, val_df, test_df = split_train_val_test(df)

    # Extract labels
    X_train, y_train, Dminusy_train = extract_labels(train_df)
    X_val,   y_val,   Dminusy_val   = extract_labels(val_df)
    X_test,  y_test,  Dminusy_test  = extract_labels(test_df)

    # Label-encode categorical features on training set only
    X_train_encoded, encoders = label_encode_features(X_train)
    X_val_encoded  = transform_with_label_encoders(X_val,  encoders)
    X_test_encoded = transform_with_label_encoders(X_test, encoders)

    return (X_train_encoded, y_train, Dminusy_train,
            X_val_encoded,  y_val,   Dminusy_val,
            X_test_encoded, y_test,  Dminusy_test,
            encoders)

def make_regressor(model_type: str = "rf", random_state: int = 42):
    """
    Create a regression model by name.
    model_type: 'linear', 'rf', 'svm', 'xgb', 'mlp'
    """
    model_type = model_type.lower()

    if model_type == "linear":
        return LinearRegression()

    if model_type == "rf":
        return RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        )

    if model_type == "svm":
        return SVR(kernel="rbf", C=1.0, epsilon=0.1)

    if model_type == "xgb":
        if not _HAS_XGB:
            raise ImportError("xgboost is not installed. Please install xgboost or use another model_type.")
        return XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            tree_method="hist",
        )

    raise ValueError(f"Unknown model_type: {model_type}")

def regression_metrics_np(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae  = np.mean(np.abs(y_true - y_pred))
    mse  = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    # R2 score
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_res / (ss_tot + eps)

    # MAPE
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100

    # SMAPE
    smape = 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)
    )

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "SMAPE": smape,
    }

def evaluate_delay(ypred_dt, ytrue_dt, Dminusy):
    """
    Evaluate on Delay = dt + (D - y) = dt + plus4delay.
    ypred_dt, ytrue_dt, Dminusy are 1D arrays of same length.
    Returns: dict of metrics.
    """
    ypred_dt = np.asarray(ypred_dt)
    ytrue_dt = np.asarray(ytrue_dt)
    Dminusy  = np.asarray(Dminusy)

    # Compute delay prediction and ground truth
    delay_pred = ypred_dt + Dminusy
    delay_true = ytrue_dt + Dminusy

    eval_dt = regression_metrics_np(ytrue_dt, ypred_dt)
    eval_delay = regression_metrics_np(delay_true, delay_pred)

    return {
        "mae": eval_dt["MAE"],
        "mse": eval_dt["MSE"],
        "rmse": eval_dt["RMSE"],
        "smape_dt": eval_dt["SMAPE"],
        "r2_dt": eval_dt["R2"],
        "smape_delay": eval_delay["SMAPE"],
        "r2_delay": eval_delay["R2"],
    }

def train_and_evaluate_package(package, model_type: str = "rf", random_state: int = 42, scale_X: bool = True):
    """
    Train a regressor on dt and evaluate using evaluate_delay().

    For model_type in {"linear", "svm"}:
        - Standardize X (StandardScaler)
        - Standardize y (dt) and inverse-transform predictions

    For {"rf", "xgb"}:
        - Optionally standardize X (trees don't need it, but it's harmless)
        - Do NOT standardize y
    """
    # Unpack data from prepare_data_for_ml
    X_train_df, y_train, Dminusy_train = package[0], package[1], package[2]
    X_val_df, y_val, Dminusy_val = package[3], package[4], package[5]
    X_test_df, y_test, Dminusy_test = package[6], package[7], package[8]
    encoders = package[9]

    # Convert X DataFrames to NumPy
    X_train_np = X_train_df.to_numpy(dtype=np.float32)
    X_val_np = X_val_df.to_numpy(dtype=np.float32)
    X_test_np = X_test_df.to_numpy(dtype=np.float32)

    # Convert y, Dminusy to NumPy 1D
    y_train_np = np.asarray(y_train, dtype=np.float32)
    y_val_np = np.asarray(y_val, dtype=np.float32)
    y_test_np = np.asarray(y_test, dtype=np.float32)

    Dminusy_train_np = np.asarray(Dminusy_train, dtype=np.float32)
    Dminusy_val_np = np.asarray(Dminusy_val, dtype=np.float32)
    Dminusy_test_np = np.asarray(Dminusy_test, dtype=np.float32)

    # ---- 1) Input scaling (X) ----
    X_scaler = None
    if scale_X: # Default: True
        X_scaler = StandardScaler()
        X_train_scaled = X_scaler.fit_transform(X_train_np)
        X_val_scaled = X_scaler.transform(X_val_np)
        X_test_scaled = X_scaler.transform(X_test_np)
    else:
        X_train_scaled, X_val_scaled, X_test_scaled = X_train_np, X_val_np, X_test_np

    # ---- 2) Target scaling (y) for Linear & SVM only ----
    model_type_lower = model_type.lower()
    y_scaler = None
    if model_type_lower in ["linear", "svm"]:
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train_np.reshape(-1, 1)).ravel()
        y_for_fit = y_train_scaled
    else:
        y_for_fit = y_train_np

    # ---- 3) Build model ----
    model = make_regressor(model_type=model_type_lower, random_state=random_state)

    # ---- 4) Fit model ----
    model.fit(X_train_scaled, y_for_fit)

    # ---- 5) Predict dt on each split ----
    ypred_train_dt_scaled = model.predict(X_train_scaled)
    ypred_val_dt_scaled = model.predict(X_val_scaled)
    ypred_test_dt_scaled = model.predict(X_test_scaled)

    # If we scaled y, inverse-transform predictions back to original dt units
    if y_scaler is not None:
        ypred_train_dt = y_scaler.inverse_transform(ypred_train_dt_scaled.reshape(-1, 1)).ravel()
        ypred_val_dt = y_scaler.inverse_transform(ypred_val_dt_scaled.reshape(-1, 1)).ravel()
        ypred_test_dt = y_scaler.inverse_transform(ypred_test_dt_scaled.reshape(-1, 1)).ravel()
    else:
        ypred_train_dt = ypred_train_dt_scaled
        ypred_val_dt = ypred_val_dt_scaled
        ypred_test_dt = ypred_test_dt_scaled

    # ---- 6) Evaluate in dt + delay space using your evaluate_delay() ----
    train_metrics = evaluate_delay(ypred_train_dt, y_train_np, Dminusy_train_np)
    val_metrics = evaluate_delay(ypred_val_dt, y_val_np, Dminusy_val_np)
    test_metrics = evaluate_delay(ypred_test_dt, y_test_np, Dminusy_test_np)

    return {
        "model": model,
        "encoders": encoders,
        "X_scaler": X_scaler,
        "y_scaler": y_scaler,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }

def train_and_evaluate_from_scenarios(scenarios, model_type: str = "rf", random_state: int = 42):
    """
    Full pipeline:
      scenarios -> DataFrame -> split -> label extraction
                -> label encoding -> train -> eval
    """
    package = prepare_data_for_ml(scenarios)
    results = train_and_evaluate_package(package, model_type=model_type, random_state=random_state)

    if model_type in ['rf', 'xgb']:
        feature_names = package[0].columns.tolist()
        feature_importances = tree_based_feature_importance(results['model'], feature_names)
        results['feature_importances'] = feature_importances

    return results

def tree_based_feature_importance(model, feature_names):
    """
    Get feature importance from tree-based models (RF, XGB).
    Returns a dict of feature_name -> importance.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return dict(zip(feature_names, importances))
    else:
        raise ValueError("Model does not have feature_importances_ attribute.")