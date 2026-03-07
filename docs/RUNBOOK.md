# Memorization vs. Algorithmic Generalization: Runbook

This runbook implements the experimental spec for evaluating memorization vs. NC¹-style
algorithmic composition on S_5 group word evaluation.

## Quick Reference

```bash
# Navigate to experiments directory
cd /Users/bkitano/Desktop/projects/vault/projects/circuit-complexity/code/nc1-tc0-transformer-toy-experiments

# Quick sanity check (5 epochs, 10k samples)
python experiments/s5_permutation_composition.py quick

# Full Grazzi et al. replication
python experiments/s5_permutation_composition.py both

# Run the full evaluation suite (after implementing extensions)
python experiments/s5_evaluation_suite.py --config configs/full_sweep.yaml
```

---

## 1. Current State: What Already Exists

### ✅ Implemented

| Component | File | Status |
|-----------|------|--------|
| S_5 token system | `s5_permutation_composition.py` | Complete (120 tokens, Cayley table) |
| S5Transformer | `s5_permutation_composition.py:558` | Complete (1-8 layers) |
| S5DeltaNet | `s5_permutation_composition.py:760` | Complete (recurrent baseline) |
| DFA baseline | `s5_permutation_composition.py:261` | Complete (100% accuracy reference) |
| Length OOD test | `s5_permutation_composition.py` | train k=32, test k=500 |
| Training loop | `s5_permutation_composition.py:841` | Complete with wandb |
| Z_m addition | `iterated_addition.py` | TC⁰ control task |
| Z_m multiplication | `iterated_multiplication.py` | NC¹ task (alternative) |

### ❌ Not Yet Implemented (Required by Spec)

| Component | Priority | Description |
|-----------|----------|-------------|
| Conjugation invariance | High | Test y → h·y·h⁻¹ equivariance |
| Element relabeling | High | Test permutation of token IDs |
| Matched-stat counterfactuals | High | Same multiset, different product |
| Partial-product probes | High | Layerwise span growth analysis |
| Cancellation-heavy dataset | Medium | Words with deliberate inverses |
| Subgroup-restricted dataset | Medium | Words from H ≤ G |
| Nearest-neighbor analysis | Medium | Exemplar retrieval detection |
| Calibration (ECE) | Medium | Confidence vs accuracy |
| Membership inference | Low | Training set detection attack |
| Abelian control (Z_n) | Low | Easy baseline (already partial) |

---

## 2. First Milestone (1 Week)

The minimal experiment that distinguishes memorization from composition:

### 2.1 Train Models

```bash
# Run quick test first to verify setup
python experiments/s5_permutation_composition.py quick

# Train transformer depth 2 vs depth 6 (reduced samples for speed)
python -c "
from experiments.s5_permutation_composition import run_experiment, ExperimentConfig

# Depth 2
config2 = ExperimentConfig(
    model_type='transformer',
    num_layers=2,
    train_seq_len=6,
    test_seq_len=7,  # k+1 for length OOD
    train_samples=100_000,
    test_samples=10_000,
    max_seq_len=16,
    epochs=50,
    use_wandb=True,
)
run_experiment(config2)

# Depth 6
config6 = ExperimentConfig(
    model_type='transformer',
    num_layers=6,
    train_seq_len=6,
    test_seq_len=7,
    train_samples=100_000,
    test_samples=10_000,
    max_seq_len=16,
    epochs=50,
    use_wandb=True,
)
run_experiment(config6)

# DeltaNet baseline
config_delta = ExperimentConfig(
    model_type='deltanet',
    num_layers=4,
    train_seq_len=6,
    test_seq_len=7,
    train_samples=100_000,
    test_samples=10_000,
    max_seq_len=16,
    epochs=50,
    use_wandb=True,
)
run_experiment(config_delta)
"
```

### 2.2 Evaluate (Manual for Now)

After training, evaluate:

1. **IID accuracy** - should be high for all models
2. **Length OOD (k+1)** - sharp drop = memorization signal
3. **Conjugation OOD** - requires implementing `conjugate_dataset()` (see Section 4)
4. **Single-token flip sensitivity** - requires implementing position perturbation

---

## 3. Implementation Plan: Negative Checks

### 3.1 Conjugation Invariance Test

Add to `s5_permutation_composition.py`:

```python
def conjugate_sequence(
    token_system: S5TokenSystem,
    tokens: list[int],
    h_idx: int,
) -> tuple[list[int], int]:
    """
    Conjugate all tokens by h: g_i -> h·g_i·h⁻¹
    Also transform the label: y -> h·y·h⁻¹

    Args:
        token_system: S5TokenSystem instance
        tokens: list of token indices
        h_idx: index of conjugating element h

    Returns:
        (conjugated_tokens, conjugated_label)
    """
    h_inv_idx = token_system.perm_to_idx[
        inverse_permutation(token_system.idx_to_perm[h_idx])
    ]

    conjugated = []
    for g_idx in tokens:
        # h·g·h⁻¹ = compose(compose(h, g), h_inv)
        hg = token_system.compose_indices(h_idx, g_idx)
        hgh_inv = token_system.compose_indices(hg, h_inv_idx)
        conjugated.append(hgh_inv)

    # Transform label similarly
    original_product = token_system.compose_sequence(tokens)
    hy = token_system.compose_indices(h_idx, original_product)
    hyh_inv = token_system.compose_indices(hy, h_inv_idx)

    return conjugated, hyh_inv


class S5ConjugatedDataset(Dataset):
    """Dataset with conjugation applied to all sequences."""

    def __init__(
        self,
        base_dataset: S5FixedKDataset,
        token_system: S5TokenSystem,
        h_idx: int | None = None,  # None = random per sample
    ):
        self.base = base_dataset
        self.token_system = token_system
        self.h_idx = h_idx

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        tokens, target, mask, k = self.base[idx]

        # Get original token list
        original_tokens = tokens[:k].tolist()

        # Choose conjugating element
        h = self.h_idx if self.h_idx is not None else random.randint(0, 119)

        # Apply conjugation
        conj_tokens, conj_label = conjugate_sequence(
            self.token_system, original_tokens, h
        )

        # Pack back into tensor format
        new_tokens = torch.zeros_like(tokens)
        for i, t in enumerate(conj_tokens):
            new_tokens[i] = t
        new_target = torch.tensor(conj_label, dtype=torch.long)

        return new_tokens, new_target, mask, k
```

**Evaluation code:**

```python
def evaluate_conjugation_invariance(
    model: nn.Module,
    base_dataset: S5FixedKDataset,
    token_system: S5TokenSystem,
    device: str,
    num_conjugators: int = 10,
) -> dict:
    """
    Test if model predictions transform correctly under conjugation.

    For each sample:
    - Get prediction y_pred on original
    - Get prediction y'_pred on conjugated input
    - Check if y'_pred == h·y_pred·h⁻¹ (equivariance)

    Returns dict with accuracy and equivariance rate.
    """
    results = {
        'original_acc': 0,
        'conjugated_acc': 0,
        'equivariance_rate': 0,  # Does output transform correctly?
        'total': 0,
    }

    model.eval()
    criterion = nn.CrossEntropyLoss()

    for h_idx in random.sample(range(120), num_conjugators):
        conj_dataset = S5ConjugatedDataset(base_dataset, token_system, h_idx)
        loader = DataLoader(conj_dataset, batch_size=256)
        base_loader = DataLoader(base_dataset, batch_size=256)

        for (tokens, targets, masks, _), (orig_tokens, orig_targets, orig_masks, _) in zip(loader, base_loader):
            # ... evaluation logic
            pass

    return results
```

### 3.2 Element Relabeling Test

```python
def create_relabeling_permutation() -> dict[int, int]:
    """Create a random bijection π: {0..119} -> {0..119}."""
    perm = list(range(120))
    random.shuffle(perm)
    return {i: perm[i] for i in range(120)}


class S5RelabeledDataset(Dataset):
    """Dataset with element IDs consistently permuted."""

    def __init__(
        self,
        base_dataset: S5FixedKDataset,
        relabeling: dict[int, int],
    ):
        self.base = base_dataset
        self.relabel = relabeling

    def __getitem__(self, idx):
        tokens, target, mask, k = self.base[idx]

        # Relabel input tokens
        new_tokens = torch.zeros_like(tokens)
        for i in range(k):
            new_tokens[i] = self.relabel[tokens[i].item()]

        # Relabel target
        new_target = torch.tensor(self.relabel[target.item()], dtype=torch.long)

        return new_tokens, new_target, mask, k
```

### 3.3 Matched-Statistics Counterfactuals

```python
def create_matched_stat_pair(
    token_system: S5TokenSystem,
    k: int,
    max_attempts: int = 100,
) -> tuple[list[int], list[int], int, int] | None:
    """
    Create two sequences with same element multiset but different products.

    Returns (seq1, seq2, product1, product2) or None if failed.
    """
    for _ in range(max_attempts):
        # Generate random sequence
        seq1 = [token_system.get_random_index() for _ in range(k)]
        product1 = token_system.compose_sequence(seq1)

        # Shuffle to get same multiset
        seq2 = seq1.copy()
        random.shuffle(seq2)
        product2 = token_system.compose_sequence(seq2)

        # Check products differ (S_5 is non-abelian, so usually they will)
        if product1 != product2:
            return seq1, seq2, product1, product2

    return None


class S5MatchedStatDataset(Dataset):
    """Dataset of matched-statistics pairs for counterfactual testing."""

    def __init__(
        self,
        token_system: S5TokenSystem,
        k: int,
        num_pairs: int,
        max_seq_len: int,
    ):
        self.token_system = token_system
        self.k = k
        self.max_seq_len = max_seq_len
        self.pairs = []

        for _ in range(num_pairs):
            pair = create_matched_stat_pair(token_system, k)
            if pair:
                self.pairs.append(pair)

    def __len__(self):
        return len(self.pairs) * 2  # Each pair gives 2 samples

    def __getitem__(self, idx):
        pair_idx = idx // 2
        is_second = idx % 2

        seq1, seq2, prod1, prod2 = self.pairs[pair_idx]
        seq = seq2 if is_second else seq1
        target = prod2 if is_second else prod1

        tokens = torch.zeros(self.max_seq_len, dtype=torch.long)
        mask = torch.zeros(self.max_seq_len, dtype=torch.float)
        for i, t in enumerate(seq):
            tokens[i] = t
            mask[i] = 1.0

        return tokens, torch.tensor(target), mask, self.k
```

### 3.4 Position Sensitivity Analysis

```python
def evaluate_position_sensitivity(
    model: nn.Module,
    dataset: S5FixedKDataset,
    token_system: S5TokenSystem,
    device: str,
    num_samples: int = 1000,
) -> dict[int, float]:
    """
    Measure how sensitive predictions are to single-token flips at each position.

    For algorithmic models: sensitivity should be uniform across positions.
    For memorizing models: early positions often have lower influence.

    Returns dict mapping position -> average prediction change rate.
    """
    model.eval()
    k = dataset.k
    sensitivities = {pos: [] for pos in range(k)}

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    with torch.no_grad():
        for idx in indices:
            tokens, target, mask, _ = dataset[idx]
            tokens = tokens.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)

            # Get original prediction
            orig_logits = model(tokens, mask)
            orig_pred = orig_logits.argmax(dim=-1).item()

            # Flip each position and check if prediction changes
            for pos in range(k):
                perturbed = tokens.clone()
                # Replace with random different token
                old_tok = tokens[0, pos].item()
                new_tok = (old_tok + random.randint(1, 119)) % 120
                perturbed[0, pos] = new_tok

                new_logits = model(perturbed, mask)
                new_pred = new_logits.argmax(dim=-1).item()

                sensitivities[pos].append(1 if new_pred != orig_pred else 0)

    return {pos: sum(v) / len(v) for pos, v in sensitivities.items()}
```

---

## 4. Implementation Plan: Positive Checks

### 4.1 Partial-Product Probes (Critical)

This is the key diagnostic for NC¹-style composition. The signature of algorithmic
computation is that deeper layers can decode products of longer spans.

```python
class PartialProductProbe(nn.Module):
    """Linear probe to predict partial products from intermediate representations."""

    def __init__(self, d_model: int, num_classes: int = 120):
        super().__init__()
        self.probe = nn.Linear(d_model, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.probe(hidden_states)


def extract_layer_representations(
    model: S5Transformer,
    tokens: torch.Tensor,
    mask: torch.Tensor,
) -> list[torch.Tensor]:
    """
    Extract hidden states after each transformer layer.

    Returns list of (batch, seq_len, d_model) tensors, one per layer.
    """
    batch_size, seq_len = tokens.shape
    device = tokens.device

    # Embed
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    x = model.token_embed(tokens) + model.pos_embed(positions)

    # Create attention mask
    attn_mask = (mask == 0)

    representations = [x.clone()]  # Layer 0 = embeddings

    # Pass through each layer
    for layer in model.transformer.layers:
        x = layer(x, src_key_padding_mask=attn_mask)
        representations.append(x.clone())

    return representations


def train_partial_product_probes(
    model: S5Transformer,
    train_dataset: S5FixedKDataset,
    token_system: S5TokenSystem,
    device: str,
    span_lengths: list[int] = [2, 4, 6, 8],  # Spans to probe
    num_epochs: int = 20,
) -> dict:
    """
    Train linear probes to predict partial products at each layer.

    For each layer ℓ and span length m, train a probe to predict
    the product of the first m tokens from the layer-ℓ representation.

    Returns:
        Dict with structure: {layer: {span: accuracy}}
    """
    model.eval()
    k = train_dataset.k
    num_layers = len(list(model.transformer.layers)) + 1  # +1 for embeddings

    # Limit spans to what's possible
    span_lengths = [s for s in span_lengths if s <= k]

    # Create probes
    probes = {
        layer: {span: PartialProductProbe(model.d_model, 120).to(device)
                for span in span_lengths}
        for layer in range(num_layers)
    }

    # Create targets for each span
    def get_partial_products(tokens: torch.Tensor, spans: list[int]) -> dict[int, torch.Tensor]:
        """Compute partial products for each span length."""
        batch_size = tokens.size(0)
        results = {}
        for span in spans:
            products = []
            for b in range(batch_size):
                seq = tokens[b, :span].tolist()
                prod = token_system.compose_sequence(seq)
                products.append(prod)
            results[span] = torch.tensor(products, device=tokens.device)
        return results

    # Training loop
    loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    for layer in range(num_layers):
        for span in span_lengths:
            probe = probes[layer][span]
            optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(num_epochs):
                total_loss = 0
                correct = 0
                total = 0

                for tokens, _, masks, _ in loader:
                    tokens = tokens.to(device)
                    masks = masks.to(device)

                    # Get representations
                    with torch.no_grad():
                        reps = extract_layer_representations(model, tokens, masks)

                    # Get partial product targets
                    targets = get_partial_products(tokens, [span])[span]

                    # Mean pool over first `span` positions
                    rep = reps[layer][:, :span, :].mean(dim=1)

                    # Train probe
                    optimizer.zero_grad()
                    logits = probe(rep)
                    loss = criterion(logits, targets)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * tokens.size(0)
                    correct += (logits.argmax(dim=-1) == targets).sum().item()
                    total += tokens.size(0)

    # Evaluate probes
    results = {layer: {} for layer in range(num_layers)}
    test_loader = DataLoader(train_dataset, batch_size=256)  # Use same data for eval

    for layer in range(num_layers):
        for span in span_lengths:
            probe = probes[layer][span]
            probe.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for tokens, _, masks, _ in test_loader:
                    tokens = tokens.to(device)
                    masks = masks.to(device)

                    reps = extract_layer_representations(model, tokens, masks)
                    targets = get_partial_products(tokens, [span])[span]
                    rep = reps[layer][:, :span, :].mean(dim=1)
                    logits = probe(rep)

                    correct += (logits.argmax(dim=-1) == targets).sum().item()
                    total += tokens.size(0)

            results[layer][span] = correct / total

    return results


def plot_probe_heatmap(results: dict, save_path: str = None):
    """
    Plot probe accuracy heatmap: layers vs span lengths.

    Key signature of NC¹-style computation:
    - Accuracy increases with layer depth
    - Deeper layers can decode longer spans
    - Roughly "doubling" span per layer
    """
    import matplotlib.pyplot as plt
    import numpy as np

    layers = sorted(results.keys())
    spans = sorted(results[layers[0]].keys())

    data = np.array([[results[l][s] for s in spans] for l in layers])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap='viridis', aspect='auto')

    ax.set_xticks(range(len(spans)))
    ax.set_xticklabels([f'm={s}' for s in spans])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f'Layer {l}' for l in layers])

    ax.set_xlabel('Partial Product Span Length')
    ax.set_ylabel('Layer')
    ax.set_title('Partial Product Probe Accuracy\n(NC¹ signature: diagonal increase)')

    plt.colorbar(im, label='Accuracy')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

---

## 5. Dataset Extensions

### 5.1 Cancellation-Heavy Distribution

```python
def generate_cancellation_heavy_sequence(
    token_system: S5TokenSystem,
    k: int,
    cancel_prob: float = 0.3,
) -> list[int]:
    """
    Generate sequence with deliberate cancellations (g, g⁻¹ pairs).

    This creates sequences that "look complex" but have simpler effective products.
    A memorizing model may struggle; an algebraic model should handle it.
    """
    tokens = []

    while len(tokens) < k:
        if len(tokens) > 0 and random.random() < cancel_prob:
            # Insert inverse of previous element
            last = tokens[-1]
            last_perm = token_system.idx_to_perm[last]
            inv_perm = inverse_permutation(last_perm)
            inv_idx = token_system.perm_to_idx[inv_perm]
            tokens.append(inv_idx)
        else:
            # Random element
            tokens.append(token_system.get_random_index())

    return tokens[:k]


class S5CancellationDataset(Dataset):
    """Dataset with cancellation-heavy sequences."""

    def __init__(
        self,
        token_system: S5TokenSystem,
        k: int,
        num_samples: int,
        max_seq_len: int,
        cancel_prob: float = 0.3,
    ):
        self.token_system = token_system
        self.k = k
        self.max_seq_len = max_seq_len
        self.data = []

        for _ in range(num_samples):
            tokens = generate_cancellation_heavy_sequence(token_system, k, cancel_prob)
            product = token_system.compose_sequence(tokens)
            self.data.append((tokens, product))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens_list, target = self.data[idx]

        tokens = torch.zeros(self.max_seq_len, dtype=torch.long)
        mask = torch.zeros(self.max_seq_len, dtype=torch.float)

        for i, t in enumerate(tokens_list):
            tokens[i] = t
            mask[i] = 1.0

        return tokens, torch.tensor(target), mask, self.k
```

### 5.2 Subgroup-Restricted Distribution

```python
def get_dihedral_subgroup_D5(token_system: S5TokenSystem) -> list[int]:
    """
    Get indices of D_5 (dihedral group of order 10) as subgroup of S_5.

    D_5 = <r, s> where:
    - r = (0 1 2 3 4) is a 5-cycle
    - s = (1 4)(2 3) is a reflection
    """
    # 5-cycle
    r = (1, 2, 3, 4, 0)
    r_idx = token_system.perm_to_idx[r]

    # Reflection (fixes 0, swaps 1↔4 and 2↔3)
    s = (0, 4, 3, 2, 1)
    s_idx = token_system.perm_to_idx[s]

    # Generate all elements by closure
    subgroup = set()
    to_process = [token_system.identity_idx, r_idx, s_idx]

    while to_process:
        g = to_process.pop()
        if g in subgroup:
            continue
        subgroup.add(g)

        # Generate products with generators
        for gen in [r_idx, s_idx]:
            prod = token_system.compose_indices(g, gen)
            if prod not in subgroup:
                to_process.append(prod)
            prod = token_system.compose_indices(gen, g)
            if prod not in subgroup:
                to_process.append(prod)

    return sorted(list(subgroup))


def get_alternating_subgroup_A5(token_system: S5TokenSystem) -> list[int]:
    """
    Get indices of A_5 (alternating group, even permutations) in S_5.
    |A_5| = 60
    """
    def is_even_permutation(perm: tuple[int, ...]) -> bool:
        """Check if permutation is even (even number of transpositions)."""
        n = len(perm)
        visited = [False] * n
        num_cycles = 0

        for i in range(n):
            if visited[i]:
                continue
            j = i
            while not visited[j]:
                visited[j] = True
                j = perm[j]
            num_cycles += 1

        # Parity: even if (n - num_cycles) is even
        return (n - num_cycles) % 2 == 0

    return [idx for idx, perm in token_system.idx_to_perm.items() if is_even_permutation(perm)]
```

---

## 6. Evaluation Suite Entry Point

Create `experiments/s5_evaluation_suite.py`:

```python
#!/usr/bin/env python3
"""
S_5 Composition: Full Evaluation Suite

Runs all negative and positive checks from the memorization vs. generalization spec.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from s5_permutation_composition import (
    S5TokenSystem, S5Transformer, S5DeltaNet, S5DFA,
    S5FixedKDataset, ExperimentConfig, run_experiment,
    evaluate, train_epoch,
)

# Import the extensions (after implementing them)
# from s5_extensions import (
#     S5ConjugatedDataset, S5RelabeledDataset, S5MatchedStatDataset,
#     S5CancellationDataset, evaluate_conjugation_invariance,
#     evaluate_position_sensitivity, train_partial_product_probes,
#     plot_probe_heatmap, get_dihedral_subgroup_D5,
# )


def run_full_evaluation(
    model: torch.nn.Module,
    token_system: S5TokenSystem,
    config: dict,
    device: str,
    output_dir: Path,
):
    """
    Run complete evaluation suite on trained model.

    Args:
        model: Trained S5Transformer or S5DeltaNet
        token_system: S5TokenSystem instance
        config: Dict with k_train, k_test_lengths, etc.
        device: 'cuda', 'mps', or 'cpu'
        output_dir: Where to save results
    """
    results = {
        'config': config,
        'negative_checks': {},
        'positive_checks': {},
    }

    k_train = config['k_train']

    # === NEGATIVE CHECKS ===

    # 1. Length extrapolation
    print("\n[1/8] Length extrapolation...")
    for k_test in config['k_test_lengths']:
        test_ds = S5FixedKDataset(
            token_system, k=k_test, num_samples=10000,
            max_seq_len=config['max_seq_len'],
        )
        loader = DataLoader(test_ds, batch_size=256)
        _, acc, _ = evaluate(model, loader, torch.nn.CrossEntropyLoss(), device)
        results['negative_checks'][f'length_ood_k{k_test}'] = acc
        print(f"  k={k_test}: acc={acc:.4f}")

    # 2. Conjugation invariance (TODO: implement)
    print("\n[2/8] Conjugation invariance... (TODO)")

    # 3. Relabeling robustness (TODO: implement)
    print("\n[3/8] Relabeling robustness... (TODO)")

    # 4. Matched-stat counterfactuals (TODO: implement)
    print("\n[4/8] Matched-stat counterfactuals... (TODO)")

    # 5. Position sensitivity (TODO: implement)
    print("\n[5/8] Position sensitivity... (TODO)")

    # 6. Nearest-neighbor dependence (TODO: implement)
    print("\n[6/8] Nearest-neighbor analysis... (TODO)")

    # 7. Calibration / ECE (TODO: implement)
    print("\n[7/8] Calibration (ECE)... (TODO)")

    # 8. Membership inference (TODO: implement)
    print("\n[8/8] Membership inference... (TODO)")

    # === POSITIVE CHECKS ===

    # 1. Partial-product probes (TODO: implement)
    print("\n[+1] Partial-product probes... (TODO)")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'evaluation_results.json'}")
    return results


def main():
    parser = argparse.ArgumentParser(description='S_5 Composition Evaluation Suite')
    parser.add_argument('--model_path', type=str, help='Path to saved model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    parser.add_argument('--k_train', type=int, default=6)
    parser.add_argument('--k_test', type=str, default='7,12', help='Comma-separated test lengths')
    parser.add_argument('--device', type=str, default='auto')

    args = parser.parse_args()

    # Device selection
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # Setup
    token_system = S5TokenSystem()
    k_test_lengths = [int(k) for k in args.k_test.split(',')]

    config = {
        'k_train': args.k_train,
        'k_test_lengths': k_test_lengths,
        'max_seq_len': max(k_test_lengths) + 8,
    }

    # Load or train model
    if args.model_path:
        print(f"Loading model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model = S5Transformer(
            num_tokens=120,
            max_seq_len=config['max_seq_len'],
            d_model=checkpoint.get('d_model', 128),
            nhead=checkpoint.get('nhead', 4),
            num_layers=checkpoint.get('num_layers', 4),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No model path provided. Training a new model...")
        exp_config = ExperimentConfig(
            model_type='transformer',
            num_layers=4,
            train_seq_len=args.k_train,
            test_seq_len=k_test_lengths[0],
            train_samples=100_000,
            test_samples=10_000,
            max_seq_len=config['max_seq_len'],
            epochs=30,
            use_wandb=False,
        )
        model, _, _ = run_experiment(exp_config)

    model = model.to(device)
    model.eval()

    # Run evaluation
    run_full_evaluation(
        model=model,
        token_system=token_system,
        config=config,
        device=device,
        output_dir=Path(args.output_dir),
    )


if __name__ == '__main__':
    main()
```

---

## 7. Expected Outcomes

### TC⁰-like Transformer (Negative Profile)

| Check | Expected Result |
|-------|-----------------|
| IID accuracy (k=6) | High (>90%) |
| Length OOD (k=7) | Sharp drop (<30%) |
| Length OOD (k=12) | Near random (~0.8%) |
| Conjugation invariance | Fails (<50% equivariance) |
| Relabeling robustness | Fails (accuracy drops) |
| Matched-stat counterfactuals | Fails (same prediction) |
| Position sensitivity | Non-uniform (early < late) |
| Partial-product probes | No span growth with depth |

### NC¹-Capable Model (Positive Profile)

| Check | Expected Result |
|-------|-----------------|
| IID accuracy (k=6) | High (>90%) |
| Length OOD (k=7) | Gradual decline (~70%) |
| Length OOD (k=12) | Still reasonable (~40%) |
| Conjugation invariance | Holds (>80% equivariance) |
| Relabeling robustness | Robust (minimal drop) |
| Matched-stat counterfactuals | Distinguishes correctly |
| Position sensitivity | Uniform across positions |
| Partial-product probes | Clear span growth with depth |

---

## 8. Concrete Commands: Full Sweep

After implementing all extensions, run the full sweep:

```bash
# 1. Control task: Z_113 addition (should work for everyone)
python experiments/iterated_addition.py

# 2. S_5 basic sweep: vary depth
for depth in 1 2 4 6 8; do
  python -c "
from experiments.s5_permutation_composition import run_experiment, ExperimentConfig
config = ExperimentConfig(
    model_type='transformer',
    num_layers=$depth,
    train_seq_len=6,
    test_seq_len=7,
    train_samples=500_000,
    test_samples=20_000,
    epochs=50,
    use_wandb=True,
)
run_experiment(config)
"
done

# 3. DeltaNet comparison
for depth in 1 2 4 6; do
  python -c "
from experiments.s5_permutation_composition import run_experiment, ExperimentConfig
config = ExperimentConfig(
    model_type='deltanet',
    num_layers=$depth,
    train_seq_len=6,
    test_seq_len=7,
    train_samples=500_000,
    test_samples=20_000,
    epochs=50,
    use_wandb=True,
)
run_experiment(config)
"
done

# 4. Full evaluation suite (after training)
python experiments/s5_evaluation_suite.py \
  --model_path checkpoints/s5_transformer_L4.pt \
  --output_dir results/transformer_L4 \
  --k_train 6 \
  --k_test 7,12

# 5. Generate probe heatmaps
python -c "
from experiments.s5_evaluation_suite import load_model, train_partial_product_probes, plot_probe_heatmap
# ... (implementation)
"
```

---

## 9. Checklist for First Milestone

- [ ] Verify setup works: `python experiments/s5_permutation_composition.py quick`
- [ ] Train transformer depth 2 on k=6
- [ ] Train transformer depth 6 on k=6
- [ ] Train DeltaNet on k=6
- [ ] Evaluate all on k=6 (IID), k=7 (OOD), k=12 (far OOD)
- [ ] Implement and run conjugation invariance test
- [ ] Implement and run position sensitivity test
- [ ] Implement partial-product probes for m ∈ {2, 4, 6}
- [ ] Generate probe heatmap
- [ ] Document findings

---

## 10. Notes

1. **Sequence disjointness**: Current implementation doesn't enforce train/test split by sequence identity. For rigorous experiments, add deduplication.

2. **Grazzi et al. numbers**: The paper uses 1.6M training samples, k=32 train, k=500 test. Our milestone uses smaller values for faster iteration.

3. **DFA is always 100%**: This is by design—it's the ground truth for what perfect algebraic computation looks like.

4. **Abelian control**: Z_n addition is in TC⁰ and should generalize OOD. Use it to verify the pipeline before drawing conclusions about S_5.
