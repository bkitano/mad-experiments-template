# MAD Experiments

Evaluation harness for novel architecture discovery. Tests whether sequence models can learn algorithmic composition (NC^1) vs memorization (TC^0) by measuring length generalization on group composition tasks.

## Setup

```bash
uv sync
```

## Quick Start

Train the baseline DeltaNet on S5 permutation composition:

```bash
uv run accelerate launch -m train.run_config --config configs/s5_example.yaml
```

This trains on short sequences (k<=5), then evaluates length generalization at k=10, 25, 50, 100, 250, 500. Results are written to `results.json`.

## Repository Structure

```
models/          Model architectures (DeltaNet, Transformer, custom)
models/__init__.py   MODEL_REGISTRY — maps config names to classes
configs/         YAML training configs
tasks/           Task definitions (token systems, datasets, data generation)
train/           Training loop and config-driven runner
harness/         Evaluation schema and scripts
AGENTS.md        Instructions for AI agents working in this repo
```

## Tasks

### S5 — Permutation Composition

The primary benchmark. 120 group elements (5! permutations of S_5). The word problem over S_5 is NC^1-complete, meaning it requires O(log n) depth to solve — constant-depth architectures (transformers) should fail to generalize to longer sequences.

- Token system: `tasks/s5/tokens.py` — 123 tokens (120 elements + BOS/EOS/PAD)
- Datasets: `tasks/s5/dataset.py` — curriculum training, fixed-k evaluation, harness functions

### Zn — Modular Addition

Control task. Addition over cyclic group Z_n is in TC^0 (solvable by constant-depth threshold circuits). Every architecture should solve this — if it doesn't, something is broken.

- Token system: `tasks/addition/tokens.py` — n+3 tokens
- Datasets: `tasks/addition/dataset.py` — same interface as S5

### Other Tasks

Available but not yet integrated into the harness runner:
- `tasks/in_context_recall/` — key-value associative recall
- `tasks/selective_copy/` — copy marked tokens
- `tasks/compression/` — sequence compression
- `tasks/state_tracking/` — generic state machine tracking

## Models

### DeltaNet (`models/deltanet.py`)

Linear RNN with delta rule updates. Uses flash-linear-attention when available (CUDA), falls back to pure PyTorch on CPU/MPS. The `allow_neg_eigval=True` flag multiplies beta by 2, giving eigenvalues in (0, 2) which is necessary for NC^1 expressivity.

### Transformer (`models/transformer.py`)

Standard transformer with learned positional embeddings. TC^0 expressivity — expected to fail on length generalization for S5.

### Adding a New Model

1. Create `models/your_model.py` implementing the standard interface:

```python
class YourModel(nn.Module):
    def __init__(self, num_tokens, num_classes, eos_idx, max_seq_len,
                 d_model, nhead, num_layers, dropout, **kwargs):
        ...
    def forward(self, tokens, mask):
        # (batch, seq_len) -> (batch, seq_len, num_classes)
        ...
```

2. Register in `models/__init__.py`:

```python
from models.your_model import YourModel
MODEL_REGISTRY["YourModel"] = YourModel
```

3. Create a config in `configs/` referencing `type: "YourModel"`.

## Configs

YAML files in `configs/` control everything. Key sections:

```yaml
dataset:
  task: "s5"              # which task
  max_k: 5                # curriculum depth
  samples_per_k: 200_000  # samples per curriculum stage
  max_seq_len: 512        # padded length (must fit OOD test lengths)

model:
  type: "GroupDeltaNet"    # key into MODEL_REGISTRY
  num_layers: 1
  nhead: 4
  d_model: 128
  # ... any kwargs your model accepts

training:
  batch_size: 2048
  lr: 1e-4
  max_epochs_per_stage: 200
  max_val_acc: 0.99       # early stop per curriculum stage

ood_test:
  lengths: [10, 25, 50, 100, 250, 500]
  samples_per_length: 10_000
```

## Data Generation API

For custom training loops or analysis, use the harness functions directly:

```python
from tasks.s5.dataset import generate_train_data, generate_eval_data, score_predictions
from tasks.s5.tokens import S5TokenSystem

ts = S5TokenSystem()

# Curriculum training data (dict of tensors)
train = generate_train_data(ts, max_k=5, samples_per_k=200_000, max_seq_len=512, seed=42)
# train["tokens"]:  (N, 512) long
# train["targets"]: (N, 512) long — scan (prefix compositions), -100 at BOS/EOS/PAD
# train["masks"]:   (N, 512) float — 1 for real tokens, 0 for padding
# train["ks"]:      (N,) long — composition depth per sample

# OOD eval inputs (NO labels)
inputs, held_out = generate_eval_data(ts, lengths=[10, 50, 100], samples_per_length=10_000, max_seq_len=512, seed=42)

# Score predictions against held-out ground truth
results = score_predictions(held_out, predictions)
# [{"length": 10, "accuracy": 0.82, "num_samples": 10000, "num_correct": 8200}, ...]
```

## Length Generalization Output

After training, `results.json` contains in-distribution and OOD metrics:

```json
{
  "task": "s5",
  "val/accuracy": 0.95,
  "val/loss": 0.18,
  "param_count": 312440,
  "wall_time_seconds": 847,
  "train_max_k": 5,
  "ood/accuracy_k10": 0.82,
  "ood/accuracy_k25": 0.51,
  "ood/accuracy_k50": 0.23,
  "ood/accuracy_k100": 0.09,
  "ood/accuracy_k250": 0.008,
  "ood/accuracy_k500": 0.008
}
```

The training script also prints a degradation table:

```
==================================================
OOD Length Generalization (trained at k<=5)
==================================================
  In-distribution (k<=5): 0.9512
    Length    Accuracy   Degradation
  --------  ----------  ------------
        10      0.8234       +0.1278
        25      0.5123       +0.4389
        50      0.2341       +0.7171
       100      0.0891       +0.8621
       250      0.0084       +0.9428
       500      0.0083       +0.9429
```

## Existing Configs

| Config | Model | Task | Notes |
|--------|-------|------|-------|
| `s5_example.yaml` | DeltaNet | S5 | Baseline with OOD eval |
| `s5_deltanet_deep.yaml` | DeltaNet | S5 | 5 layers |
| `s5_deltanet_scan.yaml` | DeltaNet | S5 | Scan targets |
| `s5_deltanet_swaps.yaml` | DeltaNet | S5 | Swaps-only subset |
| `s5_transformer_deep.yaml` | Transformer | S5 | 5 layers |
| `s5_transformer_scan.yaml` | Transformer | S5 | Scan targets |
