"""
Compression Dataset

Parametrizable by:
- vocab_size: number of distinct content tokens
- seq_len: length of sequence to memorize/compress
- structured: whether sequences have repeating patterns
- max_seq_len: maximum padded length
"""

import random

import torch
from torch.utils.data import Dataset

from tasks.compression.tokens import CompressionTokenSystem


IGNORE_INDEX = -100


class CompressionDataset(Dataset):
    """
    Dataset for compression/memorization task.

    Input: [BOS, t1, ..., tn, COMPRESS, slot1, ..., slotn, EOS, PAD, ...]
    Target: IGNORE at input positions, original tokens at output slot positions
    """

    def __init__(
        self,
        token_system: CompressionTokenSystem,
        seq_len_range: tuple[int, int],
        num_samples: int,
        max_seq_len: int,
        structured: bool = False,
    ):
        self.token_system = token_system
        self.seq_len_range = seq_len_range
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.structured = structured
        self.data = self._generate()

    def _generate(self):
        data = []
        for _ in range(self.num_samples):
            seq_len = random.randint(*self.seq_len_range)
            input_seq, target_seq = self.token_system.generate_sample(
                seq_len, structured=self.structured,
            )
            data.append((input_seq, target_seq, seq_len))
        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_seq, target_seq, seq_len = self.data[idx]

        tokens = torch.full((self.max_seq_len,), self.token_system.PAD_IDX, dtype=torch.long)
        target = torch.full((self.max_seq_len,), IGNORE_INDEX, dtype=torch.long)
        mask = torch.zeros(self.max_seq_len, dtype=torch.float)

        # BOS
        tokens[0] = self.token_system.BOS_IDX
        mask[0] = 1.0

        # Input sequence (original + COMPRESS)
        pos = 1
        for tok in input_seq:
            if pos < self.max_seq_len - seq_len - 1:
                tokens[pos] = tok
                mask[pos] = 1.0
                pos += 1

        # Output slots: predict original sequence
        for val in target_seq:
            if pos < self.max_seq_len - 1:
                tokens[pos] = self.token_system.PAD_IDX  # placeholder
                target[pos] = val
                mask[pos] = 1.0
                pos += 1

        # EOS
        if pos < self.max_seq_len:
            tokens[pos] = self.token_system.EOS_IDX
            mask[pos] = 1.0

        return tokens, target, mask, seq_len


class CompressionCurriculumDataset:
    """Curriculum wrapper: increase sequence length over stages."""

    def __init__(
        self,
        token_system: CompressionTokenSystem,
        max_seq_len_task: int,
        max_seq_len: int,
        samples_per_stage: int,
        test_size: float = 0.2,
        structured: bool = False,
    ):
        self.token_system = token_system
        self.max_seq_len_task = max_seq_len_task
        self.max_seq_len = max_seq_len
        self.samples_per_stage = samples_per_stage
        self.test_size = test_size
        self.structured = structured
        self._data_by_k = self._generate_all_data()

    def _generate_all_data(self):
        data_by_k = {}
        for k in range(1, self.max_seq_len_task + 1):
            samples = []
            for _ in range(self.samples_per_stage):
                input_seq, target_seq = self.token_system.generate_sample(
                    k, structured=self.structured,
                )
                samples.append((input_seq, target_seq, k))
            data_by_k[k] = samples
        return data_by_k

    def get_stage(self, stage_k: int):
        assert 1 <= stage_k <= self.max_seq_len_task
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

        train_ds = _CompressionStageDataset(self.token_system, self.max_seq_len, train_data)
        test_ds = _CompressionStageDataset(self.token_system, self.max_seq_len, test_data)
        return train_ds, test_ds

    def num_stages(self) -> int:
        return self.max_seq_len_task


class _CompressionStageDataset(Dataset):
    def __init__(self, token_system, max_seq_len, data):
        self.token_system = token_system
        self.max_seq_len = max_seq_len
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, target_seq, seq_len = self.data[idx]

        tokens = torch.full((self.max_seq_len,), self.token_system.PAD_IDX, dtype=torch.long)
        target = torch.full((self.max_seq_len,), IGNORE_INDEX, dtype=torch.long)
        mask = torch.zeros(self.max_seq_len, dtype=torch.float)

        tokens[0] = self.token_system.BOS_IDX
        mask[0] = 1.0

        pos = 1
        for tok in input_seq:
            if pos < self.max_seq_len - seq_len - 1:
                tokens[pos] = tok
                mask[pos] = 1.0
                pos += 1

        for val in target_seq:
            if pos < self.max_seq_len - 1:
                tokens[pos] = self.token_system.PAD_IDX
                target[pos] = val
                mask[pos] = 1.0
                pos += 1

        if pos < self.max_seq_len:
            tokens[pos] = self.token_system.EOS_IDX
            mask[pos] = 1.0

        return tokens, target, mask, seq_len
