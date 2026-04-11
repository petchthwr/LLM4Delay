from .ts2vec import TS2Vec
from .datautils import *
import numpy as np
import torch
import warnings
import random
from .utils import pad_nan_to_target
warnings.filterwarnings("ignore")

def reproducibility(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def _pad_to_common_length(*arrays):
    max_len = max(arr.shape[1] for arr in arrays)
    padded = []
    for arr in arrays:
        padded.append(pad_nan_to_target(arr, max_len, axis=1) if arr.shape[1] < max_len else arr)
    return padded

def load_trained_ts2vec(model_path, **ts2vec_kwargs):
    """Instantiate TS2Vec with defaults used here and load pretrained weights."""
    defaults = {
        'input_dims': 9,
        'output_dims': 320,
        'device': 1,
    }
    defaults.update(ts2vec_kwargs)
    model = TS2Vec(**defaults)
    model.load(model_path)
    return model

def load_standardization_params(mean_path, std_path):
    """Load mean and std numpy arrays for data standardization."""
    mean = np.load(mean_path)
    std = np.load(std_path)
    return mean, std

def pretraining():
    seed = 0
    torch.cuda.empty_cache()
    print(f'Running for seed {seed}')
    reproducibility(seed)

    # Load Air Traffic Data
    train_data_a, test_data_a, train_labels_a, test_labels_a = load_ATFM_data(data_to_path('RKSIa_v'), downsample=5, size_lim=None)
    train_data_d, test_data_d, train_labels_d, test_labels_d = load_ATFM_data(data_to_path('RKSId_v'), downsample=5, size_lim=None)
    train_data_a, train_data_d, test_data_a, test_data_d = _pad_to_common_length(train_data_a, train_data_d, test_data_a, test_data_d)
    train_data = np.concatenate((train_data_a, train_data_d, test_data_a, test_data_d), axis=0)

    #Standardize data
    mean = np.nanmean(train_data, axis=(0,1), keepdims=True)
    std = np.nanstd(train_data, axis=(0,1), keepdims=True)
    train_data = (train_data - mean) / (std + 1e-8)

    # Train a TS2Vec model
    model = TS2Vec(input_dims=9, output_dims=320, device=0)
    loss_log = model.fit(train_data, verbose=True)
    model.save('models/RKSI_ts2vec.pth')
    print('Saved pretrained TS2Vec model to models/RKSI_ts2vec.pth')

    # Save mean and std for later use
    np.save('models/RKSI_ts2vec_mean.npy', mean)
    np.save('models/RKSI_ts2vec_std.npy', std)

    # Compute instance-level representations for the combined dataset
    # Encoding demonstration.
    train_repr = model.encode(train_data, encoding_window='full_series')
    print('Train data shape:', train_data.shape)
    print('Train representation shape:', train_repr.shape)