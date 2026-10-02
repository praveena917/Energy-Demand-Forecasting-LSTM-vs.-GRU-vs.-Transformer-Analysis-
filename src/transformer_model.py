"""
Transformer encoder forecaster: same job as the LSTM/GRU (encode input_window
of past hours, predict the next `horizon` hours), but replaces recurrence
with self-attention. Every past timestep can attend directly to every other
past timestep within a single layer, instead of information having to flow
step-by-step through a chain of hidden states like it does in the LSTM/GRU.

Uses PyTorch's built-in nn.TransformerEncoder (the standard multi-head
self-attention + feedforward block) - the same way the LSTM used nn.LSTM
instead of a hand-rolled cell, keeping this consistent with the rest of the
project rather than switching styles halfway through.
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Injects each timestep's position into its representation. Self-attention
    is permutation-equivariant by default - without this, the model has no way
    to tell hour 1 of the window apart from hour 100; it would see the input
    window as an unordered bag of hours instead of a sequence."""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class TransformerForecaster(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, horizon=24, dropout=0.2, max_len=200):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, horizon),
        )

    def forward(self, x):
        # x: (batch, input_window, n_features)
        x = self.input_proj(x)           # (batch, input_window, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch, input_window, d_model) - every position now
                                          # carries information attended from every other position
        last_hidden = x[:, -1, :]        # representation of the most recent timestep
        return self.head(last_hidden)    # (batch, horizon)


if __name__ == "__main__":
    model = TransformerForecaster(n_features=10, horizon=24)
    dummy = torch.randn(8, 168, 10)
    out = model(dummy)
    print("Output shape:", out.shape)  # expect (8, 24)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")