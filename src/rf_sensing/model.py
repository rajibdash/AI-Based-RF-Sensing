"""Small CNN baseline for two-channel complex-baseband windows."""

from torch import nn


class OccupancyCNN(nn.Module):
    """Predict noise-only vs signal-present logits from [I, Q] windows."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=9, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 2),
        )

    def forward(self, iq):
        return self.network(iq)
