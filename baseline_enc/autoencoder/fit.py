import numpy as np
import torch
import warnings
import random
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torch.nn import functional as F
from .utils import pad_nan_to_target
from .datautils import load_ATFM_data, data_to_path
import os
from .dilated_conv import DilatedConvEncoder
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

def create_dataloader(data, batch_size=32, shuffle=True):
    tensor_data = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

class TCN_AutoEncoder(nn.Module):
    def __init__(self, input_dims, emb_dims, hidden_dims=64, depth=10):
        super().__init__()
        self.input_dims = input_dims
        self.emb_dims = emb_dims
        self.hidden_dims = hidden_dims

        # Encoder
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.encoder = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [emb_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)

        # Decoder
        self.decoder_dropout = nn.Dropout(p=0.1)
        self.decoder = DilatedConvEncoder(
            emb_dims,
            [hidden_dims] * depth + [hidden_dims],
            kernel_size=3
        )
        self.output_fc = nn.Linear(hidden_dims, input_dims)

    def forward(self, x):  # x: (B, T, input_dims)
        # Encode
        mask = x.isnan().any(axis=-1) # Determine mask for NaN time steps
        x[mask] = 0.0 # Replace NaNs with zeros for processing
        x = self.input_fc(x)  # (B, T, Ch)
        x = x.transpose(1, 2)  # (B, Ch, T)
        x = self.repr_dropout(self.encoder(x))  # (B, Co, T)
        x = x.transpose(1, 2)  # (B, T, Co)
        x[mask] = 0.0 # Reapply mask to avoid false max pooling
        emb = F.max_pool1d(x.transpose(1, 2), kernel_size=x.size(1)).squeeze(-1)  # (B, Co)

        # Decode
        x = emb.unsqueeze(1).repeat(1, mask.size(1), 1)  # (B, T, Co)
        x[mask] = 0.0 # Apply mask before decoding
        x = x.transpose(1, 2)  # (B, Co, T)
        x = self.decoder_dropout(self.decoder(x))  # (B, Ch, T)
        x = x.transpose(1, 2)  # (B, T, Ch)
        recon = self.output_fc(x)  # (B, T, input_dims)
        recon[mask] = 0.0

        return recon

    def encode(self, x):
        # Encode
        mask = x.isnan().any(axis=-1)
        x[mask] = 0
        x = self.input_fc(x)  # (B, T, Ch)
        x = x.transpose(1, 2)  # (B, Ch, T)
        x = self.repr_dropout(self.encoder(x))  # (B, Co, T)
        x = x.transpose(1, 2)  # (B, T, Co)
        x[mask] = 0
        emb = F.max_pool1d(x.transpose(1, 2), kernel_size=x.size(1)).squeeze(-1)  # (B, Co)

        return emb

class LSTM_AutoEncoder(nn.Module):
    def __init__(self, input_dims=9, emb_dims=320):
        super(LSTM_AutoEncoder, self).__init__()

        # Encoder
        self.lstm_encoder = nn.LSTM(input_size=input_dims, hidden_size=512, num_layers=2, batch_first=True, bidirectional=False)
        self.fc_encoder = nn.Linear(512, emb_dims)

        # Decoder
        self.fc_decoder = nn.Linear(emb_dims, 512)
        self.lstm_decoder = nn.LSTM(input_size=512, hidden_size=512, num_layers=2, batch_first=True, bidirectional=False)
        self.output_layer = nn.Linear(512, input_dims)

    def init_hidden(self, batch_size):
        h0 = torch.zeros(2, batch_size, 512).to(next(self.parameters()).device)
        c0 = torch.zeros(2, batch_size, 512).to(next(self.parameters()).device)
        return (h0, c0)

    def forward(self, x):
        mask = ~torch.isnan(x).any(dim=2) # Create mask for non-NaN time steps
        x = torch.nan_to_num(x, nan=0.0) # Replace NaNs with zeros for processing (N, T, D)

        h0 , c0 = self.init_hidden(x.size(0))
        enc_out, _ = self.lstm_encoder(x, (h0, c0)) # (N, T, 512)
        last_indices = mask.sum(dim=1) - 1 # Last valid indices is the embedding (N,)
        last_outputs = enc_out[torch.arange(x.size(0)), last_indices] # (N, 512)
        emb = self.fc_encoder(last_outputs) # (N, emb_dims)

        dec_input = self.fc_decoder(emb).unsqueeze(1).repeat(1, x.size(1), 1) # (N, T, 512)
        dec_out, _ = self.lstm_decoder(dec_input, (h0, c0)) # (N, T, 512)
        recon = self.output_layer(dec_out) # (N, T, D)
        recon[~mask] = 0.0

        return recon

    def encode(self, x):
        mask = ~torch.isnan(x).any(dim=2) # Create mask for non-NaN time steps
        x = torch.nan_to_num(x, nan=0.0) # Replace NaNs with zeros for processing (N, T, D)

        h0 , c0 = self.init_hidden(x.size(0))
        enc_out, _ = self.lstm_encoder(x, (h0, c0)) # (N, T, 512)
        last_indices = mask.sum(dim=1) - 1 # Last valid indices is the embedding (N,)
        last_outputs = enc_out[torch.arange(x.size(0)), last_indices] # (N, 512)
        emb = self.fc_encoder(last_outputs) # (N, emb_dims)

        return emb

def train(model, dataloader, num_epochs=20, learning_rate=1e-3):
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model_train_flag = model.training

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, torch.nan_to_num(inputs, nan=0.0))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss /= len(dataloader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    model.train(model_train_flag)

def visualize_tsne(embeddings, perplexity=30, n_iter=1000):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=0)
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=5)
    plt.title('t-SNE Visualization of Embeddings')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()

def load_trained_tcn_autoencoder(model_path, **tcn_kwargs):
    """Instantiate TCN_AutoEncoder with defaults used here and load pretrained weights."""
    defaults = {
        'input_dims': 9,
        'emb_dims': 320,
    }
    defaults.update(tcn_kwargs)
    model = TCN_AutoEncoder(**defaults)
    model.load_state_dict(torch.load(model_path))
    return model

def load_standardization_params(mean_path, std_path):
    """Load mean and std numpy arrays for data standardization."""
    mean = np.load(mean_path)
    std = np.load(std_path)
    return mean, std

def pretraining():
    # Pretraining Phase
    # Make the code see only gpu 1
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    print(f'Running for RKSI')
    seed = 0
    torch.cuda.empty_cache()
    print(f'Running for seed {seed}')
    reproducibility(seed)

    # Load Air Traffic Data
    train_data_a, test_data_a, train_labels_a, test_labels_a = load_ATFM_data(data_to_path('RKSIa_v'), downsample=5, size_lim=None)
    train_data_d, test_data_d, train_labels_d, test_labels_d = load_ATFM_data(data_to_path('RKSId_v'), downsample=5, size_lim=None)

    train_data_a, train_data_d, test_data_a, test_data_d = _pad_to_common_length(train_data_a, train_data_d, test_data_a, test_data_d)
    train_data = np.concatenate((train_data_a, train_data_d, test_data_a, test_data_d), axis=0)

    # Create Train Dataloaders
    train_loader = create_dataloader(train_data, batch_size=32, shuffle=True)
    print('Train data shape:', train_data.shape)

    # Standardize data
    mean = np.nanmean(train_data, axis=(0,1), keepdims=True)
    std = np.nanstd(train_data, axis=(0,1), keepdims=True)
    train_data = (train_data - mean) / (std + 1e-8)

    # Train a LSTM AutoEncoder model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TCN_AutoEncoder(input_dims=9, emb_dims=320).to(device)
    train(model, train_loader, num_epochs=20, learning_rate=1e-3)
    torch.save(model.state_dict(), 'models/RKSI_tcn_autoencoder.pth')
    print('Saved pretrained TCN AutoEncoder model to models/RKSI_tcn_autoencoder.pth')

    # Save standardization parameters
    np.save('models/RKSI_tcn_autoencoder_mean.npy', mean)
    np.save('models/RKSI_tcn_autoencoder_std.npy', std)
    print("Saved standardization parameters to models/RKSI_tcn_autoencoder_mean.npy and models/RKSI_tcn_autoencoder_std.npy")

    # Compute instance-level representations for the combined dataset
    model.eval()
    with torch.no_grad():
        all_data_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
        embeddings = model.encode(all_data_tensor).cpu().numpy()
    print('Train representation shape:', embeddings.shape)

    # Visualize embeddings using t-SNE
    visualize_tsne(embeddings, perplexity=30, n_iter=1000)
