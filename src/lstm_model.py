"""
Recurrent forecaster: encodes the input_window of past hours through a
stacked LSTM (or GRU), takes the final hidden state, and projects it through
a linear head to predict all `horizon` future hours at once (direct
multi-step forecasting, rather than an autoregressive decoder - simpler and
avoids compounding one-step errors, which is a fair baseline for comparing
against the Transformer later).

cell_type is a switch, not a separate class, specifically so that building
the GRU variant next is a one-line change: RNNForecaster(cell_type="gru", ...)
"""
import torch
import torch.nn as nn


class RNNForecaster(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, horizon=24,
                 cell_type="lstm", dropout=0.2):
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell_type]
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, horizon),
        )

    def forward(self, x):
        # x: (batch, input_window, n_features)
        out, _ = self.rnn(x)          # out: (batch, input_window, hidden_size)
        last_hidden = out[:, -1, :]   # final timestep's hidden state: (batch, hidden_size)
        return self.head(last_hidden)  # (batch, horizon)


if __name__ == "__main__":
    # quick shape sanity check
    model = RNNForecaster(n_features=10, horizon=24, cell_type="lstm")
    dummy = torch.randn(8, 168, 10)  # batch=8, input_window=168, n_features=10
    out = model(dummy)
    print("Output shape:", out.shape)  # expect (8, 24)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")