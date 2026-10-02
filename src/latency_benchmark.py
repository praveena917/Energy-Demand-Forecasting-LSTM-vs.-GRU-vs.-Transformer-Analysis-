"""
Benchmarks inference latency for each trained model - a genuine systems
concern for a deployed forecasting service, not just an accuracy exercise.
Two numbers matter in practice:
  - single-request latency: how long ONE 24h forecast takes (the "someone
    is waiting on this" case - e.g. an on-demand dashboard refresh)
  - batch throughput: how many forecasts/sec when scoring many substations
    at once (the "run this for the whole grid overnight" case)
A model that's marginally more accurate but several times slower may not be
worth deploying. Reporting this tradeoff is what a production forecasting
team actually has to weigh, on top of raw accuracy.
"""
import os
import time
import torch

from data_pipeline import load_data, engineer_features, chronological_split, fit_scaler, make_sequences
from evaluate import load_rnn, load_transformer, RESULTS_DIR

N_WARMUP = 5
N_REPEATS = 50


def benchmark_torch_model(model, X_sample, batch_size):
    x = torch.from_numpy(X_sample[:batch_size])
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(x)
        start = time.perf_counter()
        for _ in range(N_REPEATS):
            model(x)
        elapsed = time.perf_counter() - start
    per_call_ms = (elapsed / N_REPEATS) * 1000
    throughput = batch_size * N_REPEATS / elapsed
    return per_call_ms, throughput


def main():
    df = load_data()
    df = engineer_features(df)
    train_df, val_df, test_df = chronological_split(df)
    scaler = fit_scaler(train_df)
    X_test, y_test, ts_test = make_sequences(test_df, scaler, horizon=24)

    lstm_model, *_ = load_rnn(os.path.join(RESULTS_DIR, "lstm_best.pt"))
    gru_model, *_ = load_rnn(os.path.join(RESULTS_DIR, "gru_best.pt"))
    tf_model, *_ = load_transformer(os.path.join(RESULTS_DIR, "transformer_best.pt"))

    models = {"LSTM": lstm_model, "GRU": gru_model, "Transformer": tf_model}

    print(f"{'Model':<15}{'Params':>10}{'Single (ms)':>15}{'Batch64 (ms)':>15}{'Throughput/s':>15}")
    print("-" * 70)
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        single_ms, _ = benchmark_torch_model(model, X_test, batch_size=1)
        batch_ms, throughput = benchmark_torch_model(model, X_test, batch_size=64)
        print(f"{name:<15}{n_params:>10,}{single_ms:>15.2f}{batch_ms:>15.2f}{throughput:>15.0f}")

    print(
        "\nCAVEAT: the Transformer here has fewer parameters than the LSTM/GRU "
        "(sized down in Week 2 to make CPU training tractable), so part of its "
        "latency edge reflects a smaller model, not purely the parallelizable- "
        "attention architecture. A parameter-matched rerun would be needed to "
        "isolate the architectural effect alone."
    )


if __name__ == "__main__":
    main()