import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dilated_conv import DilatedConvEncoder

class TCN_Encoder(nn.Module):
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

    def forward(self, x):  # x: (B, T, input_dims)
        mask = x.isnan().any(axis=-1) # Determine mask for NaN time steps
        x[mask] = 0.0 # Replace NaNs with zeros for processing
        x = self.input_fc(x)  # (B, T, Ch)
        x = x.transpose(1, 2)  # (B, Ch, T)
        x = self.repr_dropout(self.encoder(x))  # (B, Co, T)
        x = x.transpose(1, 2)  # (B, T, Co)
        x[mask] = 0.0 # Reapply mask to avoid false max pooling
        emb = F.max_pool1d(x.transpose(1, 2), kernel_size=x.size(1)).squeeze(-1)  # (B, Co)
        return emb

class LSTM_Encoder(nn.Module):
    def __init__(self, input_dims=9, emb_dims=320):
        super().__init__()

        # Encoder
        self.lstm_encoder = nn.LSTM(input_size=input_dims, hidden_size=500, num_layers=1, batch_first=True, bidirectional=False)
        self.fc_encoder = nn.Linear(500, emb_dims)

    def init_hidden(self, batch_size):
        h0 = torch.zeros(1, batch_size, 500).to(next(self.parameters()).device)
        c0 = torch.zeros(1, batch_size, 500).to(next(self.parameters()).device)
        return (h0, c0)

    def forward(self, x):
        mask = ~torch.isnan(x).any(dim=2) # Create mask for non-NaN time steps
        x = torch.nan_to_num(x, nan=0.0) # Replace NaNs with zeros for processing (N, T, D)
        h0 , c0 = self.init_hidden(x.size(0))
        enc_out, _ = self.lstm_encoder(x, (h0, c0)) # (N, T, 512)
        last_indices = mask.sum(dim=1) - 1 # Last valid indices is the embedding (N,)
        last_outputs = enc_out[torch.arange(x.size(0)), last_indices] # (N, 512)
        emb = self.fc_encoder(last_outputs) # (N, emb_dims)
        return emb

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h, mask=None):
        """
        h: (B, T, H) LSTM hidden states
        mask: (B, T) optional, True for valid timesteps
        """
        u = torch.tanh(self.attn(h))  # (B, T, H)
        scores = self.score(u).squeeze(-1) # (B, T, 1) -> (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        alpha = F.softmax(scores, dim=1) # (B, T)
        context = torch.sum(alpha.unsqueeze(-1) * h, dim=1) # (B, H)
        return context, alpha

class LATTICE_Encoder(nn.Module):
    def __init__(self, input_dims, emb_dims, hidden_size=500, depth=1):
        super().__init__()
        self.input_dims = input_dims
        self.emb_dims = emb_dims
        self.hidden_size = hidden_size

        # LATTICE is LSTM followed by a multi-head attention mechanism
        self.lstm = nn.LSTM(input_size=input_dims, hidden_size=hidden_size, num_layers=depth, batch_first=True, bidirectional=False)
        self.attention = BahdanauAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, emb_dims)

    def init_hidden(self, batch_size):
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(next(self.parameters()).device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(next(self.parameters()).device)
        return (h0, c0)

    def src_padding_mask(self, x):
        return torch.isnan(x).any(dim=-1)  # Shape: (B, T); False for valid time steps

    def forward(self, x):
        h0, c0 = self.init_hidden(x.size(0))
        mask = ~self.src_padding_mask(x).to(torch.bool) # (B, T); True for valid time steps
        x = torch.nan_to_num(x, nan=0.0)
        x = self.lstm(x, (h0, c0))[0]  # (B, T, hidden_size)
        context, _ = self.attention(x, mask=mask) # (B, hidden_size)
        out = self.fc(context)  # (B, T, output_size)
        return out

class sinusoidal_position_encoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class CausalTransformerEncoder(nn.Module):
    def __init__(self, input_size, output_size, d_model=128, num_heads=4, ffn_dim=512, num_layers=3, dropout=0.2):
        super().__init__()

        # Hyperparameters
        self.input_size = input_size
        self.d_model = d_model
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout

        # NN Layers
        self.embedding = nn.Linear(input_size, d_model)
        self.pe = sinusoidal_position_encoding(d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, ffn_dim, dropout, activation='gelu', batch_first=True),
            num_layers
        )
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_size)
        )
        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_src_padding_mask(self, x):
        return torch.isnan(x).any(dim=-1)  # Shape: (B, T)

    def generate_causal_mask(self, x):
        T = x.shape[1]
        mask = torch.triu(torch.ones((T, T), device=x.device), diagonal=1).bool()
        mask = torch.where(mask, torch.tensor(float('-inf'), device=x.device), torch.tensor(0.0, device=x.device))
        return mask

    def forward(self, x):
        # Create masks
        src_padding_mask = self.generate_src_padding_mask(x).to(torch.bool) # (B, T)
        causal_mask = self.generate_causal_mask(x).to(torch.bool) # (T, T)

        # Encoding
        x[src_padding_mask] = 0.0
        x = self.embedding(x) # (B, T, F) -> (B, T, E)
        x = self.pe(x)
        x = self.encoder(x, src_key_padding_mask=src_padding_mask, mask=causal_mask, is_causal=True) # (B, T, E)

        # Get last valid time step
        inv_pad = ~src_padding_mask
        last_indices = inv_pad.sum(dim=-1) - 1 # (B,)
        x = x[torch.arange(x.size(0)), last_indices] # (B, E)
        x = self.fc(x) # (B, output_size)
        return x

class InvertedTransformerEncoder(nn.Module):
    def __init__(self, input_size, output_size, seq_len=256, num_feature=9, d_model=128, num_heads=4, ffn_dim=512, num_layers=3, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.embedding = nn.Linear(seq_len, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            ), num_layers=num_layers
        )
        self.fc = nn.Sequential(
            nn.Linear(d_model * num_feature, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_size)
        )
        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0)
        if x.size(1) < self.seq_len:
            pad_size = self.seq_len - x.size(1)
            x = F.pad(x, (0, 0, 0, pad_size), value=0.0)  # Pad time dimension
        elif x.size(1) > self.seq_len:
            x = x[:, :self.seq_len, :]
        x = x.transpose(1, 2)                   # (B, F, T)
        x = self.embedding(x)                   # (B, F, d_model)
        x = self.encoder(x)                     # (B, F, d_model)
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1)  # (B, F * d_model)
        x = self.fc(x)
        return x

class Regressor(nn.Module):
    def __init__(self, tabular_input_size, traj_input_size, traj_emb_size, hidden_size, output_size, encoder_type='TCN'):
        super().__init__()

        # Two components:
        # 1. Trajectory Encoder for trajectory data
        # 2. MLP Regressor for table data + trajectory embedding

        if encoder_type == 'TCN':
            self.traj_encoder = TCN_Encoder(input_dims=traj_input_size, emb_dims=traj_emb_size)
        elif encoder_type == 'LSTM':
            self.traj_encoder = LSTM_Encoder(input_dims=traj_input_size, emb_dims=traj_emb_size)
        elif encoder_type == 'cTransformer':
            self.traj_encoder = CausalTransformerEncoder(input_size=traj_input_size, output_size=traj_emb_size)
        elif encoder_type == 'iTransformer':
            self.traj_encoder = InvertedTransformerEncoder(input_size=traj_input_size, output_size=traj_emb_size)
        elif encoder_type == 'LATTICE':
            self.traj_encoder = LATTICE_Encoder(input_dims=traj_input_size, emb_dims=traj_emb_size)
        elif encoder_type == 'MLP':
            self.traj_encoder = None
        else:
            raise NotImplementedError

        if self.traj_encoder is not None:
            tab_output_size = hidden_size // 2
            tab_hidden_size = hidden_size // 2
        else:
            tab_output_size = hidden_size
            tab_hidden_size = hidden_size

        self.tab_encoder = nn.Sequential(
            nn.Linear(tabular_input_size, tab_hidden_size),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(tab_hidden_size, tab_output_size),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

        if self.traj_encoder is not None:
            regressor_input_size = hidden_size // 2 + traj_emb_size
        else:
            regressor_input_size = hidden_size

        self.regressor = nn.Sequential(
            nn.Linear(regressor_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x, traj): # x: (B, tabular_input_size), traj: (B, T, traj_input_size)
        x = self.tab_encoder(x)  # (B, hidden_size // 2)
        if self.traj_encoder is not None:
            traj_emb = self.traj_encoder(traj)  # (B, traj_emb_size)
            x = torch.cat([x, traj_emb], dim=-1)  # (B, tabular_input_size + traj_emb_size)
        out = self.regressor(x)  # (B, output_size)
        return out