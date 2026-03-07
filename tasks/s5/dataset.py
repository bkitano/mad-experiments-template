"""
S_5 Permutation Composition Experiment (Grazzi et al. Setup)

Implements the experimental setup from:
"Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues"
(Grazzi et al., ICLR 2025)

Tokenization (Section 5.2, Appendix E.2):
- Each permutation σ ∈ S_5 is mapped to a single token (integer 0-119)
- Input: sequence of k tokens representing k permutations
- Output: single token representing the composed permutation

Key experimental parameters from the paper:
- Training samples: 1.6M
- Test samples: 40K at sequence length 500
- Batch size: 512
- Learning rate: 1e-4
- Weight decay: 0.01
- Gradient clipping: 1.0
- Embedding dimension: 128
- Number of heads: 4
- Number of layers: 1 or 5
- Training sequence length: 32
- Epochs: 100
- No 1-D convolutions

Variations:
- Full S_5: all 120 permutations
- Only swaps: permutations that permute up to 2 elements (transpositions + identity)
- Swaps + 3-perm: permutations that permute up to 3 elements
- 4 tokens per transition: using special token 120 as padding

Key insight: A DFA can simulate this perfectly because:
- States = 120 (one per group element)
- Transitions = composition (δ(g, h) = g ∘ h)
- This is exactly the Cayley machine of S_5

The DFA approach is NC^1 complete (requires O(log n) depth parallel prefix),
while transformers have TC^0 expressivity (constant depth). This means transformers
should fail to generalize to longer compositions.
"""

import random
from typing import Optional

import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, concatenate_datasets

from tasks.s5.tokens import S5TokenSystem


class S5CompositionDataset(Dataset):
    """
    Dataset for S_5 composition task using single-token representation.

    Following Grazzi et al. (ICLR 2025):
    - Each permutation is a single token (0-119)
    - Input: [BOS, g_1, ..., g_k, EOS, PAD, ...]
    - Target: single token representing the composed permutation (0-119)

    Example for k=3:
      Input: [BOS, 42, 17, 89, EOS, PAD, ...]
      Target: 63  (composed permutation token)
    """

    def __init__(
        self,
        token_system: S5TokenSystem,
        k_range: tuple[int, int],
        num_samples: int,
        max_seq_len: int,
        generator_subset: Optional[list[int]] = None,  # For restricted subsets (swaps, etc.)
    ):
        self.token_system = token_system
        self.k_range = k_range
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.generator_subset = generator_subset  # If None, use all 120 permutations
        self.data = self._generate()

    def _generate(self) -> list[tuple[list[int], int, int]]:
        """Generate all samples: (list of group element indices, result index, k)."""
        data = []
        available_tokens = self.generator_subset if self.generator_subset else list(range(self.token_system.num_group_elements))

        for _ in range(self.num_samples):
            k = random.randint(*self.k_range)
            # Generate k random group element tokens
            tokens = [random.choice(available_tokens) for _ in range(k)]
            # Compose them using the token system
            result = self.token_system.compose_sequence(tokens)
            data.append((tokens, result, k))
        return data

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            tokens: (max_seq_len,) tensor: [BOS, g_1, ..., g_k, EOS, PAD, ...]
            target: scalar tensor with composed permutation index (0-119)
            mask: (max_seq_len,) tensor (1 for real tokens incl BOS/EOS, 0 for PAD)
            k: the number of group elements (not counting BOS/EOS)
        """
        group_elements, result, k = self.data[idx]

        # Initialize with PAD tokens
        tokens = torch.full((self.max_seq_len,), self.token_system.PAD_IDX, dtype=torch.long)
        mask = torch.zeros(self.max_seq_len, dtype=torch.float)

        # Format: [BOS, g_1, g_2, ..., g_k, EOS, PAD, ...]
        tokens[0] = self.token_system.BOS_IDX
        mask[0] = 1.0

        for i, tok in enumerate(group_elements):
            tokens[i + 1] = tok  # Offset by 1 for BOS
            mask[i + 1] = 1.0

        eos_pos = len(group_elements) + 1  # Position after last group element
        tokens[eos_pos] = self.token_system.EOS_IDX
        mask[eos_pos] = 1.0

        target = torch.tensor(result, dtype=torch.long)
        return tokens, target, mask, k


class S5FixedKDataset(Dataset):
    """
    Dataset with fixed sequence length k (Grazzi setup).

    Produces sequences in format: [BOS, g_1, g_2, ..., g_k, EOS, PAD, PAD, ...]
    The actual sequence length is k + 2 (for BOS and EOS).
    """

    def __init__(
        self,
        token_system: S5TokenSystem,
        k: int,
        num_samples: int,
        max_seq_len: int,
        generator_subset: Optional[list[int]] = None,
    ):
        self.token_system = token_system
        self.k = k
        self.max_seq_len = max_seq_len
        self.generator_subset = generator_subset
        self.data = self._generate(num_samples)

        # Verify max_seq_len is sufficient for k + BOS + EOS
        assert max_seq_len >= k + 2, f"max_seq_len ({max_seq_len}) must be >= k + 2 ({k + 2})"

    def _generate(self, num_samples: int) -> list[tuple[list[int], int]]:
        data = []
        available_tokens = self.generator_subset if self.generator_subset else list(range(self.token_system.num_group_elements))

        for _ in range(num_samples):
            tokens = [random.choice(available_tokens) for _ in range(self.k)]
            result = self.token_system.compose_sequence(tokens)
            data.append((tokens, result))
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            tokens: (max_seq_len,) tensor: [BOS, g_1, ..., g_k, EOS, PAD, ...]
            target: scalar tensor with composed permutation index (0-119)
            mask: (max_seq_len,) tensor (1 for real tokens incl BOS/EOS, 0 for PAD)
            k: the number of group elements (not counting BOS/EOS)
        """
        group_elements, result = self.data[idx]

        # Initialize with PAD tokens
        tokens = torch.full((self.max_seq_len,), self.token_system.PAD_IDX, dtype=torch.long)
        mask = torch.zeros(self.max_seq_len, dtype=torch.float)

        # Format: [BOS, g_1, g_2, ..., g_k, EOS, PAD, ...]
        tokens[0] = self.token_system.BOS_IDX
        mask[0] = 1.0

        for i, tok in enumerate(group_elements):
            tokens[i + 1] = tok  # Offset by 1 for BOS
            mask[i + 1] = 1.0

        eos_pos = len(group_elements) + 1  # Position after last group element
        tokens[eos_pos] = self.token_system.EOS_IDX
        mask[eos_pos] = 1.0

        target = torch.tensor(result, dtype=torch.long)
        return tokens, target, mask, self.k


class S5MixedKDataset(Dataset):
    """
    Dataset with mixed sequence lengths for curriculum learning (Grazzi setup).

    Produces sequences in format: [BOS, g_1, g_2, ..., g_k, EOS, PAD, PAD, ...]
    """

    def __init__(
        self,
        token_system: S5TokenSystem,
        k_values: list[int],
        samples_per_k: int,
        max_seq_len: int,
        generator_subset: Optional[list[int]] = None,
    ):
        self.token_system = token_system
        self.max_seq_len = max_seq_len
        self.generator_subset = generator_subset
        self.data = self._generate(k_values, samples_per_k)

    def _generate(self, k_values: list[int], samples_per_k: int) -> list[tuple[list[int], int, int]]:
        data = []
        available_tokens = self.generator_subset if self.generator_subset else list(range(self.token_system.num_group_elements))

        for k in k_values:
            for _ in range(samples_per_k):
                tokens = [random.choice(available_tokens) for _ in range(k)]
                result = self.token_system.compose_sequence(tokens)
                data.append((tokens, result, k))
        random.shuffle(data)
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            tokens: (max_seq_len,) tensor: [BOS, g_1, ..., g_k, EOS, PAD, ...]
            target: scalar tensor with composed permutation index (0-119)
            mask: (max_seq_len,) tensor (1 for real tokens incl BOS/EOS, 0 for PAD)
            k: the number of group elements (not counting BOS/EOS)
        """
        group_elements, result, k = self.data[idx]

        # Initialize with PAD tokens
        tokens = torch.full((self.max_seq_len,), self.token_system.PAD_IDX, dtype=torch.long)
        mask = torch.zeros(self.max_seq_len, dtype=torch.float)

        # Format: [BOS, g_1, g_2, ..., g_k, EOS, PAD, ...]
        tokens[0] = self.token_system.BOS_IDX
        mask[0] = 1.0

        for i, tok in enumerate(group_elements):
            tokens[i + 1] = tok  # Offset by 1 for BOS
            mask[i + 1] = 1.0

        eos_pos = len(group_elements) + 1  # Position after last group element
        tokens[eos_pos] = self.token_system.EOS_IDX
        mask[eos_pos] = 1.0

        target = torch.tensor(result, dtype=torch.long)
        return tokens, target, mask, k


class S5CurriculumDataset(Dataset):
    """
    Curriculum learning dataset using HuggingFace datasets concatenation.

    Creates samples ordered from k=1 to k=max_k (easy to hard).
    Uses HuggingFace `concatenate_datasets` to join datasets for each k value.
    """

    def __init__(
        self,
        token_system: S5TokenSystem,
        max_k: int,
        samples_per_k: int,
        max_seq_len: int,
        generator_subset: Optional[list[int]] = None,
    ):
        self.token_system = token_system
        self.max_k = max_k
        self.samples_per_k = samples_per_k
        self.max_seq_len = max_seq_len
        self.generator_subset = generator_subset
        self.hf_dataset = self._create_curriculum_dataset()

    def _create_curriculum_dataset(self) -> HFDataset:
        """Create curriculum dataset using HuggingFace concatenation."""
        hf_datasets = []

        for k in range(1, self.max_k + 1):
            # Create PyTorch dataset for this k
            pytorch_ds = S5FixedKDataset(
                self.token_system,
                k=k,
                num_samples=self.samples_per_k,
                max_seq_len=self.max_seq_len,
                generator_subset=self.generator_subset,
            )

            # Convert to HuggingFace dataset
            data_dict = {"tokens": [], "targets": [], "masks": [], "k": []}
            for i in range(len(pytorch_ds)):
                tokens, target, mask, k_val = pytorch_ds[i]
                data_dict["tokens"].append(tokens.tolist())
                data_dict["targets"].append(target.item())
                data_dict["masks"].append(mask.tolist())
                data_dict["k"].append(k_val)

            hf_ds = HFDataset.from_dict(data_dict)
            hf_datasets.append(hf_ds)

        # Concatenate in order (curriculum: k=1, k=2, ..., k=max_k)
        return concatenate_datasets(hf_datasets)

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            tokens: (max_seq_len,) tensor: [BOS, g_1, ..., g_k, EOS, PAD, ...]
            target: scalar tensor with composed permutation index (0-119)
            mask: (max_seq_len,) tensor (1 for real tokens incl BOS/EOS, 0 for PAD)
            k: the number of group elements (not counting BOS/EOS)
        """
        item = self.hf_dataset[idx]
        tokens = torch.tensor(item["tokens"], dtype=torch.long)
        target = torch.tensor(item["targets"], dtype=torch.long)
        mask = torch.tensor(item["masks"], dtype=torch.float)
        k = item["k"]
        return tokens, target, mask, k

    def train_test_split(self, test_size: float = 0.2) -> tuple["S5CurriculumDataset", "S5CurriculumDataset"]:
        """
        Split dataset into train and test sets, preserving curriculum order.

        Returns:
            Tuple of (train_dataset, test_dataset) as S5CurriculumDataset instances
        """
        split = self.hf_dataset.train_test_split(test_size=test_size, shuffle=False)

        train_ds = S5CurriculumDataset.__new__(S5CurriculumDataset)
        train_ds.token_system = self.token_system
        train_ds.max_k = self.max_k
        train_ds.samples_per_k = self.samples_per_k
        train_ds.max_seq_len = self.max_seq_len
        train_ds.generator_subset = self.generator_subset
        train_ds.hf_dataset = split["train"]

        test_ds = S5CurriculumDataset.__new__(S5CurriculumDataset)
        test_ds.token_system = self.token_system
        test_ds.max_k = self.max_k
        test_ds.samples_per_k = self.samples_per_k
        test_ds.max_seq_len = self.max_seq_len
        test_ds.generator_subset = self.generator_subset
        test_ds.hf_dataset = split["test"]

        return train_ds, test_ds


# =============================================================================
# Staged Curriculum Wrapper (provides get_stage interface for run_config.py)
# =============================================================================


class _S5StageDataset(Dataset):
    """Internal dataset for a single S5 curriculum stage."""

    IGNORE_INDEX = -100  # Standard ignore index for CrossEntropyLoss

    def __init__(
        self,
        token_system: S5TokenSystem,
        max_seq_len: int,
        data: list[tuple[list[int], list[int], int]],  # (elements, scan, k)
    ):
        self.token_system = token_system
        self.max_seq_len = max_seq_len
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
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
            target[i + 1] = scan_val  # Predict prefix composition at each position
            mask[i + 1] = 1.0

        eos_pos = len(elements) + 1
        tokens[eos_pos] = self.token_system.EOS_IDX
        mask[eos_pos] = 1.0
        # target[eos_pos] stays IGNORE (no prediction at EOS)

        return tokens, target, mask, k


class S5CurriculumWrapper:
    """
    Wrapper to provide get_stage() interface for S5, matching ZnCurriculumDataset.

    This generates data similarly to ZnCurriculumDataset with staged curriculum.
    """

    def __init__(
        self,
        token_system: S5TokenSystem,
        max_k: int,
        samples_per_k: int,
        max_seq_len: int,
        test_size: float = 0.2,
        generator_subset: Optional[str] = None,  # "swaps", "3perm", or None for all
        fixed_k: Optional[int] = None,  # For non-curriculum mode, only generate this k
    ):
        self.token_system = token_system
        self.max_k = max_k
        self.samples_per_k = samples_per_k
        self.max_seq_len = max_seq_len
        self.test_size = test_size
        self.generator_subset = generator_subset
        self.fixed_k = fixed_k
        self._data_by_k = self._generate_all_data()

    def _generate_all_data(self) -> dict[int, list[tuple[list[int], list[int]]]]:
        """Generate and store data for each k value."""
        data_by_k = {}
        # Determine which tokens to sample from
        if self.generator_subset == "swaps":
            available_tokens = self.token_system.get_swap_indices()
        elif self.generator_subset == "3perm":
            available_tokens = self.token_system.get_3perm_indices()
        else:
            available_tokens = list(range(self.token_system.num_group_elements))

        # If fixed_k is set, only generate data for that k (non-curriculum mode)
        if self.fixed_k is not None:
            k_values = [self.fixed_k]
        else:
            k_values = range(1, self.max_k + 1)

        for k in k_values:
            k_data = []
            for _ in range(self.samples_per_k):
                tokens = [random.choice(available_tokens) for _ in range(k)]
                scan = self.token_system.scan_sequence(tokens)  # All prefix compositions
                k_data.append((tokens, scan))
            data_by_k[k] = k_data
        return data_by_k

    def get_stage(self, stage_k: int) -> tuple[_S5StageDataset, _S5StageDataset]:
        """
        Get train/test datasets for curriculum stage k.

        Stage k includes all samples with k values from 1 to stage_k.
        """
        assert 1 <= stage_k <= self.max_k, f"stage_k must be in [1, {self.max_k}]"

        train_data = []
        test_data = []

        for k in range(1, stage_k + 1):
            k_data = self._data_by_k[k]
            n_test = int(len(k_data) * self.test_size)

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

        random.shuffle(train_data)
        random.shuffle(test_data)

        train_ds = _S5StageDataset(self.token_system, self.max_seq_len, train_data)
        test_ds = _S5StageDataset(self.token_system, self.max_seq_len, test_data)

        return train_ds, test_ds

    def num_stages(self) -> int:
        """Return the number of curriculum stages (equal to max_k)."""
        return self.max_k

    def get_fixed_k(self, k: int) -> tuple[_S5StageDataset, _S5StageDataset]:
        """
        Get train/test datasets for a fixed k value only (non-curriculum mode).

        Unlike get_stage(), this returns data ONLY for the specified k,
        not k=1 to k. This is useful for testing at long sequence lengths
        like k=500 (Grazzi et al. setup).
        """
        assert 1 <= k <= self.max_k, f"k must be in [1, {self.max_k}]"

        k_data = self._data_by_k[k]
        n_test = int(len(k_data) * self.test_size)

        indices = list(range(len(k_data)))
        random.shuffle(indices)

        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        train_data = []
        test_data = []

        for idx in train_indices:
            tokens, scan = k_data[idx]
            train_data.append((tokens, scan, k))
        for idx in test_indices:
            tokens, scan = k_data[idx]
            test_data.append((tokens, scan, k))

        random.shuffle(train_data)
        random.shuffle(test_data)

        train_ds = _S5StageDataset(self.token_system, self.max_seq_len, train_data)
        test_ds = _S5StageDataset(self.token_system, self.max_seq_len, test_data)

        return train_ds, test_ds


if __name__ == "__main__":
    from tasks.s5.tokens import S5TokenSystem

    token_system = S5TokenSystem()
    dataset = S5CurriculumDataset(token_system, max_k=5, samples_per_k=10_000, max_seq_len=512)

    train_dataset, test_dataset = dataset.train_test_split(test_size=0.2)
    print(len(train_dataset))
    print(len(test_dataset))
    tokens, target, mask, k = train_dataset[0]
    print(tokens)
    print(target)
    print(mask)

    tokens, target, mask, k = test_dataset[0]
    print(tokens)
    print(target)
    print(mask)