"""
Trains the GRU variant of the same RNNForecaster used for the LSTM. This file
is intentionally tiny - the payoff of building `cell_type` as a switch back in
lstm_model.py instead of a separate class is that GRU support is now a
one-line change, not a rewrite.
"""
from train_lstm import train

if __name__ == "__main__":
    train(cell_type="gru")