"""
Iterated Addition over Z_n Dataset

Dataset classes for training models on iterated modular addition:
- Input: [BOS, a_1, a_2, ..., a_k, EOS, PAD, ...]
- Output: (a_1 + a_2 + ... + a_k) mod n

Includes curriculum learning support with increasingly long sequences.
"""

import random

import torch
from torch.utils.data import Dataset

from tasks.addition.tokens import ZnTokenSystem


class ZnAdditionDataset(Dataset):
    """
    Dataset for iterated addition over Z_n.

    Input: [BOS, a_1, ..., a_k, EOS, PAD, ...]
    Target: single token representing the sum mod n (0 to n-1)
    """

    def __init__(
        self,
        token_system: ZnTokenSystem,
        k_range: tuple[int, int],
        num_samples: int,
        max_seq_len: int,
    ):
        self.token_system = token_system
        self.k_range = k_range
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.data = self._generate()

    def _generate(self) -> list[tuple[list[int], int, int]]:
        """Generate all samples: (list of element indices, result index, k)."""
        data = []

        for _ in range(self.num_samples):
            k = random.randint(*self.k_range)
            tokens = [self.token_system.get_random_index() for _ in range(k)]
            result = self.token_system.add_sequence(tokens)
            data.append((tokens, result, k))
        return data

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            tokens: (max_seq_len,) tensor: [BOS, a_1, ..., a_k, EOS, PAD, ...]
            target: scalar tensor with sum mod n (0 to n-1)
            mask: (max_seq_len,) tensor (1 for real tokens incl BOS/EOS, 0 for PAD)
            k: the number of elements (not counting BOS/EOS)
        """
        elements, result, k = self.data[idx]

        tokens = torch.full((self.max_seq_len,), self.token_system.PAD_IDX, dtype=torch.long)
        mask = torch.zeros(self.max_seq_len, dtype=torch.float)

        # Format: [BOS, a_1, a_2, ..., a_k, EOS, PAD, ...]
        tokens[0] = self.token_system.BOS_IDX
        mask[0] = 1.0

        for i, tok in enumerate(elements):
            tokens[i + 1] = tok
            mask[i + 1] = 1.0

        eos_pos = len(elements) + 1
        tokens[eos_pos] = self.token_system.EOS_IDX
        mask[eos_pos] = 1.0

        target = torch.tensor(result, dtype=torch.long)
        return tokens, target, mask, k


class ZnFixedKDataset(Dataset):
    """
    Dataset with fixed sequence length k.

    Produces sequences in format: [BOS, a_1, a_2, ..., a_k, EOS, PAD, PAD, ...]
    The actual sequence length is k + 2 (for BOS and EOS).
    """

    def __init__(
        self,
        token_system: ZnTokenSystem,
        k: int,
        num_samples: int,
        max_seq_len: int,
    ):
        self.token_system = token_system
        self.k = k
        self.max_seq_len = max_seq_len
        self.data = self._generate(num_samples)

        assert max_seq_len >= k + 2, f"max_seq_len ({max_seq_len}) must be >= k + 2 ({k + 2})"

    def _generate(self, num_samples: int) -> list[tuple[list[int], int]]:
        data = []

        for _ in range(num_samples):
            tokens = [self.token_system.get_random_index() for _ in range(self.k)]
            result = self.token_system.add_sequence(tokens)
            data.append((tokens, result))
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            tokens: (max_seq_len,) tensor: [BOS, a_1, ..., a_k, EOS, PAD, ...]
            target: scalar tensor with sum mod n (0 to n-1)
            mask: (max_seq_len,) tensor (1 for real tokens incl BOS/EOS, 0 for PAD)
            k: the number of elements (not counting BOS/EOS)
        """
        elements, result = self.data[idx]

        tokens = torch.full((self.max_seq_len,), self.token_system.PAD_IDX, dtype=torch.long)
        mask = torch.zeros(self.max_seq_len, dtype=torch.float)

        tokens[0] = self.token_system.BOS_IDX
        mask[0] = 1.0

        for i, tok in enumerate(elements):
            tokens[i + 1] = tok
            mask[i + 1] = 1.0

        eos_pos = len(elements) + 1
        tokens[eos_pos] = self.token_system.EOS_IDX
        mask[eos_pos] = 1.0

        target = torch.tensor(result, dtype=torch.long)
        return tokens, target, mask, self.k


class ZnMixedKDataset(Dataset):
    """
    Dataset with mixed sequence lengths for curriculum learning.

    Produces sequences in format: [BOS, a_1, a_2, ..., a_k, EOS, PAD, PAD, ...]
    """

    def __init__(
        self,
        token_system: ZnTokenSystem,
        k_values: list[int],
        samples_per_k: int,
        max_seq_len: int,
    ):
        self.token_system = token_system
        self.max_seq_len = max_seq_len
        self.data = self._generate(k_values, samples_per_k)

    def _generate(self, k_values: list[int], samples_per_k: int) -> list[tuple[list[int], int, int]]:
        data = []

        for k in k_values:
            for _ in range(samples_per_k):
                tokens = [self.token_system.get_random_index() for _ in range(k)]
                result = self.token_system.add_sequence(tokens)
                data.append((tokens, result, k))
        random.shuffle(data)
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            tokens: (max_seq_len,) tensor: [BOS, a_1, ..., a_k, EOS, PAD, ...]
            target: scalar tensor with sum mod n (0 to n-1)
            mask: (max_seq_len,) tensor (1 for real tokens incl BOS/EOS, 0 for PAD)
            k: the number of elements (not counting BOS/EOS)
        """
        elements, result, k = self.data[idx]

        tokens = torch.full((self.max_seq_len,), self.token_system.PAD_IDX, dtype=torch.long)
        mask = torch.zeros(self.max_seq_len, dtype=torch.float)

        tokens[0] = self.token_system.BOS_IDX
        mask[0] = 1.0

        for i, tok in enumerate(elements):
            tokens[i + 1] = tok
            mask[i + 1] = 1.0

        eos_pos = len(elements) + 1
        tokens[eos_pos] = self.token_system.EOS_IDX
        mask[eos_pos] = 1.0

        target = torch.tensor(result, dtype=torch.long)
        return tokens, target, mask, k


class ZnCurriculumDataset(Dataset):
    """
    Curriculum learning dataset with expanding stages.

    Each curriculum stage includes all k values from 1 to current_k:
    - Stage 1: k=1 only
    - Stage 2: k=1-2
    - Stage 3: k=1-3
    - ...
    - Stage max_k: k=1 to max_k

    Use get_stage(k) to get train/test splits for curriculum stage k.
    """

    def __init__(
        self,
        token_system: ZnTokenSystem,
        max_k: int,
        samples_per_k: int,
        max_seq_len: int,
        test_size: float = 0.2,
    ):
        self.token_system = token_system
        self.max_k = max_k
        self.samples_per_k = samples_per_k
        self.max_seq_len = max_seq_len
        self.test_size = test_size
        # Store data grouped by k: {k: [(tokens, result), ...]}
        self._data_by_k = self._generate_all_data()

    def _generate_all_data(self) -> dict[int, list[tuple[list[int], list[int]]]]:
        """Generate and store data for each k value."""
        data_by_k = {}
        for k in range(1, self.max_k + 1):
            k_data = []
            for _ in range(self.samples_per_k):
                tokens = [self.token_system.get_random_index() for _ in range(k)]
                scan = self.token_system.scan_sequence(tokens)  # All prefix sums
                k_data.append((tokens, scan))
            data_by_k[k] = k_data
        return data_by_k

    def get_stage(self, stage_k: int) -> tuple["_ZnStageDataset", "_ZnStageDataset"]:
        """
        Get train/test datasets for curriculum stage k.

        Stage k includes all samples with k values from 1 to stage_k.
        Both train and test sets contain samples from all k=1 to k=stage_k.

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        assert 1 <= stage_k <= self.max_k, f"stage_k must be in [1, {self.max_k}]"

        train_data = []
        test_data = []

        for k in range(1, stage_k + 1):
            k_data = self._data_by_k[k]
            n_test = int(len(k_data) * self.test_size)

            # Shuffle within each k before splitting
            indices = list(range(len(k_data)))
            random.shuffle(indices)

            test_indices = indices[:n_test]
            train_indices = indices[n_test:]

            for idx in train_indices:
                tokens, scan = k_data[idx]
                train_data.append((tokens, scan, k))
            for idx in test_indices:
                tokens, scan = k_data[idx]
                test_data.append((tokens, scan, k))

        # Shuffle both sets so k values are mixed
        random.shuffle(train_data)
        random.shuffle(test_data)

        train_ds = _ZnStageDataset(self.token_system, self.max_seq_len, train_data)
        test_ds = _ZnStageDataset(self.token_system, self.max_seq_len, test_data)

        return train_ds, test_ds

    def num_stages(self) -> int:
        """Return the number of curriculum stages (equal to max_k)."""
        return self.max_k


class _ZnStageDataset(Dataset):
    """Internal dataset for a single curriculum stage."""

    IGNORE_INDEX = -100  # Standard ignore index for CrossEntropyLoss

    def __init__(
        self,
        token_system: ZnTokenSystem,
        max_seq_len: int,
        data: list[tuple[list[int], list[int], int]],  # (elements, scan, k)
    ):
        self.token_system = token_system
        self.max_seq_len = max_seq_len
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            tokens: (max_seq_len,) tensor: [BOS, a_1, ..., a_k, EOS, PAD, ...]
            target: (max_seq_len,) tensor with prefix sums at element positions, IGNORE elsewhere
            mask: (max_seq_len,) tensor (1 for real tokens incl BOS/EOS, 0 for PAD)
            k: the number of elements (not counting BOS/EOS)
        """
        elements, scan, k = self.data[idx]

        tokens = torch.full((self.max_seq_len,), self.token_system.PAD_IDX, dtype=torch.long)
        # Target: IGNORE at BOS/EOS/PAD, scan values at group element positions
        target = torch.full((self.max_seq_len,), self.IGNORE_INDEX, dtype=torch.long)
        mask = torch.zeros(self.max_seq_len, dtype=torch.float)

        tokens[0] = self.token_system.BOS_IDX
        mask[0] = 1.0
        # target[0] stays IGNORE (no prediction at BOS)

        for i, (tok, scan_val) in enumerate(zip(elements, scan)):
            tokens[i + 1] = tok
            target[i + 1] = scan_val  # Predict prefix sum at each position
            mask[i + 1] = 1.0

        eos_pos = len(elements) + 1
        tokens[eos_pos] = self.token_system.EOS_IDX
        mask[eos_pos] = 1.0
        # target[eos_pos] stays IGNORE (no prediction at EOS)

        return tokens, target, mask, k


# =============================================================================
# Harness functions — uniform data generation and scoring
# =============================================================================

IGNORE_INDEX = _ZnStageDataset.IGNORE_INDEX


def generate_train_data(
    token_system: ZnTokenSystem,
    max_k: int,
    samples_per_k: int,
    max_seq_len: int,
    stage: int | None = None,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Generate curriculum training data with scan targets.

    Returns dict with keys: tokens, targets, masks, ks.
    All tensors have shape (N, max_seq_len) except ks which is (N,).
    """
    if seed is not None:
        random.seed(seed)

    k_range = range(1, (stage or max_k) + 1)

    all_tokens = []
    all_targets = []
    all_masks = []
    all_ks = []

    for k in k_range:
        for _ in range(samples_per_k):
            elements = [token_system.get_random_index() for _ in range(k)]
            scan = token_system.scan_sequence(elements)

            tokens = torch.full((max_seq_len,), token_system.PAD_IDX, dtype=torch.long)
            target = torch.full((max_seq_len,), IGNORE_INDEX, dtype=torch.long)
            mask = torch.zeros(max_seq_len, dtype=torch.float)

            tokens[0] = token_system.BOS_IDX
            mask[0] = 1.0
            for i, (tok, scan_val) in enumerate(zip(elements, scan)):
                tokens[i + 1] = tok
                target[i + 1] = scan_val
                mask[i + 1] = 1.0
            eos_pos = len(elements) + 1
            tokens[eos_pos] = token_system.EOS_IDX
            mask[eos_pos] = 1.0

            all_tokens.append(tokens)
            all_targets.append(target)
            all_masks.append(mask)
            all_ks.append(k)

    indices = list(range(len(all_tokens)))
    random.shuffle(indices)

    return {
        "tokens": torch.stack([all_tokens[i] for i in indices]),
        "targets": torch.stack([all_targets[i] for i in indices]),
        "masks": torch.stack([all_masks[i] for i in indices]),
        "ks": torch.tensor([all_ks[i] for i in indices], dtype=torch.long),
    }


def generate_eval_data(
    token_system: ZnTokenSystem,
    lengths: list[int],
    samples_per_length: int,
    max_seq_len: int,
    seed: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Generate OOD eval data — inputs WITHOUT labels.

    Returns:
        inputs: dict with tokens, masks, ks (no targets)
        held_out: dict with targets, ks (for scoring)
    """
    if seed is not None:
        random.seed(seed)

    all_tokens = []
    all_targets = []
    all_masks = []
    all_ks = []

    for k in lengths:
        assert k + 2 <= max_seq_len, f"length {k} + 2 > max_seq_len {max_seq_len}"
        for _ in range(samples_per_length):
            elements = [token_system.get_random_index() for _ in range(k)]
            scan = token_system.scan_sequence(elements)

            tokens = torch.full((max_seq_len,), token_system.PAD_IDX, dtype=torch.long)
            target = torch.full((max_seq_len,), IGNORE_INDEX, dtype=torch.long)
            mask = torch.zeros(max_seq_len, dtype=torch.float)

            tokens[0] = token_system.BOS_IDX
            mask[0] = 1.0
            for i, (tok, scan_val) in enumerate(zip(elements, scan)):
                tokens[i + 1] = tok
                target[i + 1] = scan_val
                mask[i + 1] = 1.0
            eos_pos = len(elements) + 1
            tokens[eos_pos] = token_system.EOS_IDX
            mask[eos_pos] = 1.0

            all_tokens.append(tokens)
            all_targets.append(target)
            all_masks.append(mask)
            all_ks.append(k)

    inputs = {
        "tokens": torch.stack(all_tokens),
        "masks": torch.stack(all_masks),
        "ks": torch.tensor(all_ks, dtype=torch.long),
    }
    held_out = {
        "targets": torch.stack(all_targets),
        "ks": torch.tensor(all_ks, dtype=torch.long),
    }
    return inputs, held_out


def score_predictions(
    held_out: dict[str, torch.Tensor],
    predictions: dict[str, list[list[int]]],
) -> list[dict]:
    """Score predictions against held-out targets.

    Args:
        held_out: targets and ks tensors from generate_eval_data
        predictions: {str(length): [[pred at each position], ...]}

    Returns:
        list of {length, accuracy, num_samples, num_correct}
    """
    targets = held_out["targets"]
    ks = held_out["ks"]

    results = []
    for length in sorted(set(ks.tolist())):
        length_str = str(length)
        length_mask = ks == length
        length_targets = targets[length_mask]
        n_samples = length_targets.shape[0]

        if length_str not in predictions:
            results.append({"length": length, "accuracy": 0.0, "num_samples": 0, "num_correct": 0})
            continue

        preds_list = predictions[length_str]
        if len(preds_list) != n_samples:
            results.append({"length": length, "accuracy": 0.0, "num_samples": n_samples, "num_correct": 0})
            continue

        preds_tensor = torch.tensor(preds_list, dtype=torch.long)
        valid = length_targets != IGNORE_INDEX
        correct = ((preds_tensor == length_targets) & valid).sum().item()
        total = valid.sum().item()

        results.append({
            "length": length,
            "accuracy": correct / total if total > 0 else 0.0,
            "num_samples": n_samples,
            "num_correct": int(correct),
        })

    return results


if __name__ == "__main__":
    token_system = ZnTokenSystem(n=10)

    # Test generate_train_data
    train = generate_train_data(token_system, max_k=3, samples_per_k=100, max_seq_len=16, seed=42)
    print(f"Train: {train['tokens'].shape}")

    # Test generate_eval_data
    inputs, held_out = generate_eval_data(token_system, lengths=[5, 10], samples_per_length=50, max_seq_len=32, seed=42)
    print(f"Eval: {inputs['tokens'].shape}, no targets: {'targets' not in inputs}")

    # Test scoring with perfect predictions
    preds = {str(k): held_out["targets"][held_out["ks"] == k].tolist() for k in [5, 10]}
    results = score_predictions(held_out, preds)
    for r in results:
        print(f"  k={r['length']}: accuracy={r['accuracy']:.4f}")
