# Agent Instructions

You are an ML researcher agent working on novel architecture discovery. Your job is to implement a model architecture, train it, and evaluate its length generalization on group composition tasks.

## Repository Structure

```
models/          ← YOU EDIT THIS — your model architecture goes here
configs/         ← YOU EDIT THIS — your training config goes here
tasks/           ← DO NOT EDIT — fixed task definitions and data generation
train/           ← DO NOT EDIT — fixed training loop
harness/         ← DO NOT EDIT — evaluation scripts
```

## Model Interface

Your model must implement this interface:

```python
class YourModel(nn.Module):
    def __init__(
        self,
        num_tokens: int,    # total vocab (group elements + BOS/EOS/PAD)
        num_classes: int,    # output classes (group elements)
        eos_idx: int,
        max_seq_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        **kwargs,
    ):
        ...

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        # tokens: (batch, seq_len) — token indices
        # mask:   (batch, seq_len) — 1 for real tokens, 0 for padding
        # returns: (batch, seq_len, num_classes) — logits at each position
        ...
```

See `models/deltanet.py` and `models/transformer.py` for reference implementations.

## Available Tasks

### S5 — Permutation Composition (primary benchmark)
- 120 group elements (all permutations of S_5)
- Non-abelian, NC^1-complete — tests whether architecture can do iterative composition
- Token system: `tasks/s5/tokens.py` (123 tokens: 120 elements + BOS/EOS/PAD)

### Zn — Modular Addition (control task)
- n group elements (cyclic group Z_n, default n=10)
- Abelian, in TC^0 — every architecture should solve this
- Token system: `tasks/addition/tokens.py` (n+3 tokens)

## Workflow

### 1. Implement your model

Create `models/your_model.py` following the interface above. Register it in `models/__init__.py`:

```python
# models/__init__.py
from models.deltanet import GroupDeltaNet
from models.transformer import GroupTransformer
from models.your_model import YourModel  # add this

MODEL_REGISTRY: dict[str, type] = {
    "GroupDeltaNet": GroupDeltaNet,
    "GroupTransformer": GroupTransformer,
    "YourModel": YourModel,  # add this
}
```

### 2. Create a config

Create `configs/your_config.yaml`:

```yaml
dataset:
  task: "s5"
  max_k: 5
  samples_per_k: 200_000
  max_seq_len: 512        # must be >= max OOD test length + 2
  test_size: 0.2

model:
  type: "YourModel"       # must match create_model() in run_config.py
  num_layers: 4
  nhead: 4
  d_model: 128
  dropout: 0.1
  use_compile: true
  # ... any additional kwargs your model accepts

training:
  batch_size: 2048
  lr: 1e-4
  weight_decay: 0.01
  gradient_clip: 1.0
  max_val_acc: 0.99
  max_epochs_per_stage: 200

ood_test:
  lengths: [10, 25, 50, 100, 250, 500]
  samples_per_length: 10_000

logging:
  wandb_project: "mad-experiments"
```

### 3. Train

```bash
uv run accelerate launch -m train.run_config --config configs/your_config.yaml
```

This uses all available GPUs automatically (the sandbox generates an accelerate config at boot). It will:
- Train with curriculum (k=1, then k=1..2, then k=1..3, etc.)
- Evaluate in-distribution after training
- Run OOD length generalization at each length in `ood_test.lengths`
- Print a degradation table showing accuracy vs sequence length
- Write `results.json` with all metrics

### 4. Check results

`results.json` contains:

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

## Data Generation Functions

If you need direct access to data (e.g., for custom training loops), use the harness functions:

```python
from tasks.s5.dataset import generate_train_data, generate_eval_data, score_predictions
from tasks.s5.tokens import S5TokenSystem

ts = S5TokenSystem()

# Training data — returns dict of tensors (tokens, targets, masks, ks)
train = generate_train_data(ts, max_k=5, samples_per_k=200_000, max_seq_len=512)

# OOD eval — returns (inputs, held_out)
# inputs has NO targets — you can't see the answers
inputs, held_out = generate_eval_data(ts, lengths=[10, 50, 100], samples_per_length=10_000, max_seq_len=512)

# Score your predictions
predictions = {}
for k in [10, 50, 100]:
    mask = inputs['ks'] == k
    tokens = inputs['tokens'][mask]
    preds = model(tokens, inputs['masks'][mask]).argmax(-1).tolist()
    predictions[str(k)] = preds

results = score_predictions(held_out, predictions)
```

## What Success Looks Like

The key metric is **length generalization** — can your model solve S5 composition at sequence lengths much longer than it was trained on?

| Metric | Baseline (DeltaNet) | Target |
|--------|-------------------|--------|
| In-distribution (k<=5) | ~95% | >= 90% |
| OOD k=10 | varies | >= 80% |
| OOD k=25 | varies | >= 60% |
| OOD k=50 | varies | any improvement |
| OOD k=500 | ~chance (0.8%) | any improvement |

A model that maintains accuracy at k=100+ is doing genuine iterative composition, not memorization.

## Rules

1. **Only edit** `models/` and `configs/`
2. **Do not edit** `tasks/`, `train/`, or `harness/`
3. **Time budget**: prioritize getting a working run over perfecting the architecture
4. **Register your model**: add it to `MODEL_REGISTRY` in `models/__init__.py` so the config can reference it by name
