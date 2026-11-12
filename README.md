# AI-Based Spectrum and Radio Sensing

An educational starter for AI-assisted spectrum occupancy sensing and the wider radio-sensing landscape. It pairs an original technical guide with a reproducible synthetic-data baseline, advisory agents, and a small CI setup.

> **Scope:** the executable code is a synthetic binary occupancy-classification demo with advisory recommendations. It is not an implementation of the survey's broader sensing framework, a certified spectrum monitor, or an autonomous transmitter controller. The included data is synthetic; it does not reproduce results from the cited paper.

## What's here

- [`AI-based spectrum sensing or RF Sensing.md`](AI-based%20spectrum%20sensing%20or%20RF%20Sensing.md): This md file is expanded, page-referenced survey synthesis covering control and data channel models and metrics, radar detection/estimation/tracking, spectrum and signal analysis, localization/channel charting, structured datasets, and deployment architecture. Also, includes a source-coverage index and clearly separates survey findings from the original starter blueprint document.
- `src/rf_sensing/synthetic.py`: This file is seeded IQ data generator with occupied/unoccupied labels and randomized SINR, carrier offset, phase, and fading.
- `src/rf_sensing/model.py`: compact 1-D CNN over real/imaginary IQ channels.
- `src/rf_sensing/agents.py`: model-backed sensing agent and a conservative policy agent that emits recommendations, not radio-control commands.
- `src/rf_sensing/train.py`: deterministic train/validation loop with a versioned checkpoint payload.

## Quick start

```bash

python -m venv .venv
. .venv/bin/activate              # Windows: .venv\\Scripts\\activate
python -m pip install -e '.[dev]'
python -m rf_sensing.train --epochs 8 --samples 6000 --output artifacts/rf_sensing.pt
pytest

```

The training command prints validation loss/accuracy and saves model weights plus the generator/training metadata. Keep `artifacts/` out of version control; checkpoints should be tracked with a model/data registry when moving beyond this demo.

## Minimal inference

```python
import numpy as np
import torch
from rf_sensing.agents import SpectrumSensingAgent, SpectrumPolicyAgent
from rf_sensing.model import OccupancyCNN

model = OccupancyCNN()
checkpoint = torch.load('artifacts/rf_sensing.pt', map_location='cpu', weights_only=True)
model.load_state_dict(checkpoint['state_dict'])
sensor = SpectrumSensingAgent(model)
policy = SpectrumPolicyAgent()
# iq must have shape (2, window_samples): [I, Q]
result = sensor.observe(iq)
recommendation = policy.recommend(result.occupied_probability)
print(result, recommendation)
```

For production, add calibrated confidence, a measured noise-floor baseline, site-specific validation, a regulatory policy gate, and hardware-in-the-loop tests before acting on any recommendation.

## Development

```bash
python -m rf_sensing.train --help
pytest -q
```

## Disclaimer

See the guide for the scope, assumptions, metrics, lifecycle, caveats, and further-work plan. The source survey citation is included there. 
This is AI assisted tool generated document, response might have some errors Or hallucinations.
