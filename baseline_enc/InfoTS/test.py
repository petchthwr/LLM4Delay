from .infots import InfoTS as MetaInfoTS
from .datautils import load_ATFM_data, data_to_path
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size//2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)

def _pad_to_common_length(*arrays):
    max_len = max(arr.shape[1] for arr in arrays)
    padded = []
    for arr in arrays:
        padded.append(pad_nan_to_target(arr, max_len, axis=1) if arr.shape[1] < max_len else arr)
    return padded

def load_trained_infots(model_path, **infots_kwargs):
    """Instantiate InfoTS with repo defaults and load pretrained weights."""
    defaults = {
        'input_dims': 9,
        'output_dims': 320,
        'device': 'cuda',
    }
    defaults.update(infots_kwargs)
    model = MetaInfoTS(**defaults)
    model.load(model_path)
    return model

def load_standardization_params(mean_path, std_path):
    """Load mean and std numpy arrays for data standardization."""
    mean = np.load(mean_path)
    std = np.load(std_path)
    return mean, std

def pretraining():
    dset = 'RKSI'
    seed = 0

    print(f'Running for {dset}')
    print(f'Running for seed {seed}')

    valid_dataset_a = load_ATFM_data(data_to_path('RKSIa_v'), downsample=5, size_lim=None)
    a_train_data, a_train_labels, a_test_data, a_test_labels = valid_dataset_a
    valid_dataset_d = load_ATFM_data(data_to_path('RKSId_v'), downsample=5, size_lim=None)
    d_train_data, d_train_labels, d_test_data, d_test_labels = valid_dataset_d

    train_data_a, train_data_d, test_data_a, test_data_d = _pad_to_common_length(a_train_data, d_train_data, a_test_data, d_test_data)
    train_data = np.concatenate((train_data_a, train_data_d, test_data_a, test_data_d), axis=0)

    # Standardize data
    mean = np.nanmean(train_data, axis=(0,1), keepdims=True)
    std = np.nanstd(train_data, axis=(0,1), keepdims=True)
    train_data = (train_data - mean) / (std + 1e-8)

    model = MetaInfoTS(input_dims=train_data.shape[-1])
    model.fit(train_data, n_iters=200, n_epochs=400)
    model.save('models/RKSI_infots.pth')
    print('Saved pretrained InfoTS model to models/RKSI_infots.pth')

    # Save standardization parameters
    np.save('models/RKSI_infots_mean.npy', mean)
    np.save('models/RKSI_infots_std.npy', std)

    # Compute instance-level representations for the combined dataset
    # Encoding demonstration.
    train_repr = model.encode(train_data)
    print('Train data shape:', train_data.shape)
    print('Train representation shape:', train_repr.shape)