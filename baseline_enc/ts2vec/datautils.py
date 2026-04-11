import os
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import pickle
from .utils import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

def load_UCR(dataset):
    train_file = os.path.join('datasets/UCR', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('datasets/UCR', dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels
    
    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels


def load_UEA(dataset):
    train_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TEST.arff')[0]
    
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    
    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y
    
    
def load_forecast_npy(name, univar=False):
    data = np.load(f'datasets/{name}.npy')    
    if univar:
        data = data[: -1:]
        
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)

    pred_lens = [24, 48, 96, 288, 672]
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, 0


def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)


def load_forecast_csv(name, univar=False):
    data = pd.read_csv(f'datasets/{name}.csv', index_col='date', parse_dates=True)
    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]
    
    if univar:
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            data = data[['OT']]
        elif name == 'electricity':
            data = data[['MT_001']]
        else:
            data = data.iloc[:, -1:]
        
    data = data.to_numpy()
    if name == 'ETTh1' or name == 'ETTh2':
        train_slice = slice(None, 12*30*24)
        valid_slice = slice(12*30*24, 16*30*24)
        test_slice = slice(16*30*24, 20*30*24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12*30*24*4)
        valid_slice = slice(12*30*24*4, 16*30*24*4)
        test_slice = slice(16*30*24*4, 20*30*24*4)
    else:
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    if name in ('electricity'):
        data = np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
    else:
        data = np.expand_dims(data, 0)
    
    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)
    
    if name in ('ETTh1', 'ETTh2', 'electricity'):
        pred_lens = [24, 48, 168, 336, 720]
    else:
        pred_lens = [24, 48, 96, 288, 672]
        
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols


def load_anomaly(name):
    res = pkl_load(f'datasets/{name}.pkl')
    return res['all_train_data'], res['all_train_labels'], res['all_train_timestamps'], \
           res['all_test_data'],  res['all_test_labels'],  res['all_test_timestamps'], \
           res['delay']


def gen_ano_train_data(all_train_data):
    maxl = np.max([ len(all_train_data[k]) for k in all_train_data ])
    pretrain_data = []
    for k in all_train_data:
        train_data = pad_nan_to_target(all_train_data[k], maxl, axis=0)
        pretrain_data.append(train_data)
    pretrain_data = np.expand_dims(np.stack(pretrain_data), 2)
    return pretrain_data


def load_ATFM_data(path, split_point='auto', downsample=2, size_lim=None, direction=True, polar=True):

    # Load data, takes only the last 3 columns (x, y, z)
    with open(os.path.join(path, 'x_train.pkl'), 'rb') as f:
        x_train = pickle.load(f)[:, -3:, :]
        x_train = np.transpose(x_train, (0, 2, 1))
    with open(os.path.join(path, 'x_test.pkl'), 'rb') as f:
        x_test = pickle.load(f)[:, -3:, :]
        x_test = np.transpose(x_test, (0, 2, 1))

    # Load labels if they exist
    if not os.path.exists(os.path.join(path, 'y_train.pkl')) or not os.path.exists(os.path.join(path, 'y_test.pkl')):
        y_train, y_test = None, None
    else:
        label_encoder = LabelEncoder()
        with open(os.path.join(path, 'y_train.pkl'), 'rb') as f:
            y_train = pickle.load(f)
        with open(os.path.join(path, 'y_test.pkl'), 'rb') as f:
            y_test = pickle.load(f)
        label_encoder.fit(np.concatenate([y_train, y_test], axis=0))
        y_train = label_encoder.transform(y_train)
        y_test = label_encoder.transform(y_test)

        assert x_train.shape[0] == y_train.shape[0], 'Number of samples and labels do not match!'
        assert x_test.shape[0] == y_test.shape[0], 'Number of samples and labels do not match!'

    x_train = x_train[:, ::downsample, :] if downsample != 1 else x_train
    x_test = x_test[:, ::downsample, :] if downsample != 1 else x_test

    if split_point != 'auto':
        x = np.concatenate([x_train, x_test], axis=0)
        x_train, x_test = x[:int(len(x) * split_point)], x[int(len(x) * split_point):]
        if y_train is not None:
            y = np.concatenate([y_train, y_test], axis=0)
            y_train, y_test = y[:int(len(y) * split_point)], y[int(len(y) * split_point):]

    x_train = x_train[:size_lim] if size_lim is not None else x_train
    x_test = x_test[:size_lim] if size_lim is not None else x_test
    y_train = y_train[:size_lim] if size_lim is not None and y_train is not None else y_train
    y_test = y_test[:size_lim] if size_lim is not None and y_test is not None else y_test

    if direction:
        u_train = np.array([get_data_directional_vec(x_i) for x_i in x_train])
        u_test = np.array([get_data_directional_vec(x_i) for x_i in x_test])
        x_train = np.concatenate([x_train, u_train], axis=-1)
        x_test = np.concatenate([x_test, u_test], axis=-1)

    if polar:
        p_train = np.array([get_data_polar(x_i) for x_i in x_train])
        p_test = np.array([get_data_polar(x_i) for x_i in x_test])
        x_train = np.concatenate([x_train, p_train], axis=-1)
        x_test = np.concatenate([x_test, p_test], axis=-1)

    return x_train, x_test, y_train, y_test

def data_to_path(data):
    if data == 'RKSIa_v':
        return '../data_pt/arrival_v'
    elif data == 'RKSId_v':
        return '../data_pt/departure_v'
    else:
        raise ValueError('Invalid data')


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
    #theta = (theta + np.pi) / (2 * np.pi)

    polar = np.concatenate([r, sin_theta, cos_theta], axis=1)

    return polar

def get_data_directional_vec(x):
    original_shape = x.shape
    x = x[~np.isnan(x).any(axis=1)]
    u = get_directional_vec(x)
    u = np.pad(u, ((0, original_shape[0] - u.shape[0]), (0, 0)), mode='constant', constant_values=float('nan'))
    return u

def get_data_polar(x):
    original_shape = x.shape
    x = x[~np.isnan(x).any(axis=1)]
    p = get_polar(x)
    p = np.pad(p, ((0, original_shape[0] - p.shape[0]), (0, 0)), mode='constant', constant_values=float('nan'))
    return p