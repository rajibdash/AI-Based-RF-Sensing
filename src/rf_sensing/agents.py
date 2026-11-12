"""Advisory sensing agents; no radio-control interfaces are provided."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class SensingResult:
    occupied_probability: float
    predicted_occupied: bool


class SpectrumSensingAgent:
    """Run model inference on one [I, Q] window shaped (2, samples)."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model.eval()

    @torch.inference_mode()
    def observe(self, iq: np.ndarray) -> SensingResult:
        if iq.ndim != 2 or iq.shape[0] != 2 or iq.shape[1] < 2:
            raise ValueError("iq must have shape (2, samples) with at least 2 samples")
        tensor = torch.as_tensor(iq, dtype=torch.float32).unsqueeze(0)
        probability = torch.softmax(self.model(tensor), dim=1)[0, 1].item()
        return SensingResult(probability, probability >= 0.5)


class SpectrumPolicyAgent:
    """Convert occupancy confidence into a conservative advisory outcome."""

    def __init__(self, occupied_threshold: float = 0.8, candidate_threshold: float = 0.2) -> None:
        if not 0.5 < occupied_threshold <= 1.0:
            raise ValueError("occupied_threshold must be in (0.5, 1]")
        if not 0.0 <= candidate_threshold < 0.5:
            raise ValueError("candidate_threshold must be in [0, 0.5)")
        self.occupied_threshold = occupied_threshold
        self.candidate_threshold = candidate_threshold

    def recommend(self, occupied_probability: float) -> str:
        if not 0.0 <= occupied_probability <= 1.0:
            raise ValueError("occupied_probability must be in [0, 1]")
        if occupied_probability >= self.occupied_threshold:
            return "protect"
        if occupied_probability <= self.candidate_threshold:
            return "access_candidate"
        return "resense"
