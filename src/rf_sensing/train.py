"""Train the synthetic occupancy CNN and save a reproducible checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rf_sensing.model import OccupancyCNN
from rf_sensing.synthetic import make_dataset


def train(epochs: int, samples: int, output: Path, seed: int) -> None:
    if epochs < 1 or samples < 20:
        raise ValueError("epochs must be positive and samples must be at least 20")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    iq, labels = make_dataset(samples, seed=seed)
    order = np.random.default_rng(seed).permutation(samples)
    split = max(1, int(samples * 0.8))
    train_ids, valid_ids = order[:split], order[split:]
    if not len(valid_ids):
        raise ValueError("samples must leave at least one validation example")
    x = torch.from_numpy(iq)
    y = torch.from_numpy(labels)
    train_loader = DataLoader(TensorDataset(x[train_ids], y[train_ids]), batch_size=128, shuffle=True)
    valid_x, valid_y = x[valid_ids], y[valid_ids]

    model = OccupancyCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(batch_x), batch_y).backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits = model(valid_x)
            loss = loss_fn(logits, valid_y).item()
            accuracy = (logits.argmax(dim=1) == valid_y).float().mean().item()
        print(f"epoch={epoch + 1}/{epochs} validation_loss={loss:.4f} validation_accuracy={accuracy:.3f}")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model": "OccupancyCNN",
            "seed": seed,
            "samples": samples,
            "samples_per_window": int(iq.shape[-1]),
            "snr_db_range": [-12.0, 12.0],
            "label_semantics": {"0": "noise_only", "1": "signal_present"},
        },
        output,
    )
    print(f"checkpoint={output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("artifacts/rf_sensing.pt"))
    args = parser.parse_args()
    train(args.epochs, args.samples, args.output, args.seed)


if __name__ == "__main__":
    main()
