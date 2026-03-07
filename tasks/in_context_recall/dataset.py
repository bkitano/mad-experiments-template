"""
In-Context Recall Dataset

Parametrizable by:
- vocab_size: number of distinct key/value tokens
- num_pairs: number of key-value pairs per sequence
- max_seq_len: maximum sequence length (with padding)
- noise_keys: whether to add noise to query keys
- multi_query: number of queries per sequence
"""

import random
from typing import Optional

import torch
from torch.utils.data import Dataset

from tasks.in_context_recall.tokens import InContextRecallTokenSystem


IGNORE_INDEX = -100


class InContextRecallDataset(Dataset):
    """
    Dataset for in-context recall task.

    Input: [BOS, k1, v1, SEP, k2, v2, SEP, ..., QUERY, k_q, EOS, PAD, ...]
    Target: IGNORE everywhere except at the query key position, where target = correct_value
    """

    def __init__(
        self,
        token_system: InContextRecallTokenSystem,
        num_pairs_range: tuple[int, int],
        num_samples: int,
        max_seq_len: int,
        noise_keys: bool = False,
        noise_prob: float = 0.1,
    ):
        self.token_system = token_system
        self.num_pairs_range = num_pairs_range
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.noise_keys = noise_keys
        self.noise_prob = noise_prob
        self.data = self._generate()

    def _generate(self) -> list[tuple[list[int], int, int]]:
        """Generate all samples: (full_seq_tokens, correct_value, num_pairs)."""
        data = []
        for _ in range(self.num_samples):
            num_pairs = random.randint(*self.num_pairs_range)
            seq, _query_key, correct_value = self.token_system.generate_sample(
                num_pairs=num_pairs,
                noise_keys=self.noise_keys,
                noise_prob=self.noise_prob,
            )
            data.append((seq, correct_value, num_pairs))
        return data

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            tokens: (max_seq_len,) tensor
            target: (max_seq_len,) tensor with IGNORE everywhere except answer position
            mask: (max_seq_len,) tensor (1 for real tokens, 0 for PAD)
            k: num_pairs (for per-k accuracy tracking)
        """
        seq, correct_value, num_pairs = self.data[idx]

        tokens = torch.full((self.max_seq_len,), self.token_system.PAD_IDX, dtype=torch.long)
        target = torch.full((self.max_seq_len,), IGNORE_INDEX, dtype=torch.long)
        mask = torch.zeros(self.max_seq_len, dtype=torch.float)

        # BOS
        tokens[0] = self.token_system.BOS_IDX
        mask[0] = 1.0

        # Fill sequence
        for i, tok in enumerate(seq):
            if i + 1 < self.max_seq_len - 1:  # leave room for EOS
                tokens[i + 1] = tok
                mask[i + 1] = 1.0

        # The answer position is the last token in seq (the query key position)
        answer_pos = len(seq)  # position after the query key
        if answer_pos < self.max_seq_len - 1:
            target[answer_pos] = correct_value

        # EOS
        eos_pos = len(seq) + 1
        if eos_pos < self.max_seq_len:
            tokens[eos_pos] = self.token_system.EOS_IDX
            mask[eos_pos] = 1.0

        return tokens, target, mask, num_pairs


class InContextRecallCurriculumDataset:
    """Curriculum wrapper for in-context recall with increasing num_pairs."""

    def __init__(
        self,
        token_system: InContextRecallTokenSystem,
        max_pairs: int,
        samples_per_stage: int,
        max_seq_len: int,
        test_size: float = 0.2,
        noise_keys: bool = False,
    ):
        self.token_system = token_system
        self.max_pairs = max_pairs
        self.samples_per_stage = samples_per_stage
        self.max_seq_len = max_seq_len
        self.test_size = test_size
        self.noise_keys = noise_keys
        self._data_by_k = self._generate_all_data()

    def _generate_all_data(self) -> dict[int, list]:
        data_by_k = {}
        for k in range(1, self.max_pairs + 1):
            samples = []
            for _ in range(self.samples_per_stage):
                seq, _qk, correct_value = self.token_system.generate_sample(
                    num_pairs=k, noise_keys=self.noise_keys,
                )
                samples.append((seq, correct_value, k))
            data_by_k[k] = samples
        return data_by_k

    def get_stage(self, stage_k: int) -> tuple[InContextRecallDataset, InContextRecallDataset]:
        """Get train/test datasets for curriculum stage k (includes k=1..stage_k)."""
        assert 1 <= stage_k <= self.max_pairs

        train_data, test_data = [], []
        for k in range(1, stage_k + 1):
            k_data = self._data_by_k[k]
            n_test = int(len(k_data) * self.test_size)
            indices = list(range(len(k_data)))
            random.shuffle(indices)
            for idx in indices[n_test:]:
                train_data.append(k_data[idx])
            for idx in indices[:n_test]:
                test_data.append(k_data[idx])

        random.shuffle(train_data)
        random.shuffle(test_data)

        train_ds = _RecallStageDataset(self.token_system, self.max_seq_len, train_data)
        test_ds = _RecallStageDataset(self.token_system, self.max_seq_len, test_data)
        return train_ds, test_ds

    def num_stages(self) -> int:
        return self.max_pairs


class _RecallStageDataset(Dataset):
    """Internal dataset for a single curriculum stage."""

    def __init__(self, token_system, max_seq_len, data):
        self.token_system = token_system
        self.max_seq_len = max_seq_len
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, correct_value, num_pairs = self.data[idx]

        tokens = torch.full((self.max_seq_len,), self.token_system.PAD_IDX, dtype=torch.long)
        target = torch.full((self.max_seq_len,), IGNORE_INDEX, dtype=torch.long)
        mask = torch.zeros(self.max_seq_len, dtype=torch.float)

        tokens[0] = self.token_system.BOS_IDX
        mask[0] = 1.0

        for i, tok in enumerate(seq):
            if i + 1 < self.max_seq_len - 1:
                tokens[i + 1] = tok
                mask[i + 1] = 1.0

        answer_pos = len(seq)
        if answer_pos < self.max_seq_len - 1:
            target[answer_pos] = correct_value

        eos_pos = len(seq) + 1
        if eos_pos < self.max_seq_len:
            tokens[eos_pos] = self.token_system.EOS_IDX
            mask[eos_pos] = 1.0

        return tokens, target, mask, num_pairs
