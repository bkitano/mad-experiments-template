"""
State Tracking Dataset

Parametrizable by:
- token_system: any StateTrackingTokenSystem
- sequence length (number of operations)
- max_seq_len: padded sequence length
"""

import random

import torch
from torch.utils.data import Dataset

from tasks.state_tracking.tokens import StateTrackingTokenSystem


IGNORE_INDEX = -100


class StateTrackingDataset(Dataset):
    """
    Dataset for state tracking task.

    Input: [BOS, op1, op2, ..., opk, EOS, PAD, ...]
    Target: [IGNORE, state1, state2, ..., statek, IGNORE, IGNORE, ...]
    where state_i = transition(state_{i-1}, op_i)
    """

    def __init__(
        self,
        token_system: StateTrackingTokenSystem,
        k_range: tuple[int, int],
        num_samples: int,
        max_seq_len: int,
    ):
        self.token_system = token_system
        self.k_range = k_range
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.data = self._generate()

    def _generate(self) -> list[tuple[list[int], list[int], int]]:
        data = []
        for _ in range(self.num_samples):
            k = random.randint(*self.k_range)
            ops = [self.token_system.get_random_index() for _ in range(k)]
            states = self.token_system.apply_ops(ops)
            data.append((ops, states, k))
        return data

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        ops, states, k = self.data[idx]

        tokens = torch.full((self.max_seq_len,), self.token_system.PAD_IDX, dtype=torch.long)
        target = torch.full((self.max_seq_len,), IGNORE_INDEX, dtype=torch.long)
        mask = torch.zeros(self.max_seq_len, dtype=torch.float)

        tokens[0] = self.token_system.BOS_IDX
        mask[0] = 1.0

        for i, (op, state) in enumerate(zip(ops, states)):
            tokens[i + 1] = op
            target[i + 1] = state
            mask[i + 1] = 1.0

        eos_pos = len(ops) + 1
        if eos_pos < self.max_seq_len:
            tokens[eos_pos] = self.token_system.EOS_IDX
            mask[eos_pos] = 1.0

        return tokens, target, mask, k


class StateTrackingCurriculumDataset:
    """Curriculum wrapper: increase sequence length over stages."""

    def __init__(
        self,
        token_system: StateTrackingTokenSystem,
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
        self._data_by_k = self._generate_all_data()

    def _generate_all_data(self) -> dict:
        data_by_k = {}
        for k in range(1, self.max_k + 1):
            samples = []
            for _ in range(self.samples_per_k):
                ops = [self.token_system.get_random_index() for _ in range(k)]
                states = self.token_system.apply_ops(ops)
                samples.append((ops, states, k))
            data_by_k[k] = samples
        return data_by_k

    def get_stage(self, stage_k: int):
        assert 1 <= stage_k <= self.max_k
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

        train_ds = _StateTrackingStageDataset(self.token_system, self.max_seq_len, train_data)
        test_ds = _StateTrackingStageDataset(self.token_system, self.max_seq_len, test_data)
        return train_ds, test_ds

    def num_stages(self) -> int:
        return self.max_k


class _StateTrackingStageDataset(Dataset):
    def __init__(self, token_system, max_seq_len, data):
        self.token_system = token_system
        self.max_seq_len = max_seq_len
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ops, states, k = self.data[idx]

        tokens = torch.full((self.max_seq_len,), self.token_system.PAD_IDX, dtype=torch.long)
        target = torch.full((self.max_seq_len,), IGNORE_INDEX, dtype=torch.long)
        mask = torch.zeros(self.max_seq_len, dtype=torch.float)

        tokens[0] = self.token_system.BOS_IDX
        mask[0] = 1.0

        for i, (op, state) in enumerate(zip(ops, states)):
            tokens[i + 1] = op
            target[i + 1] = state
            mask[i + 1] = 1.0

        eos_pos = len(ops) + 1
        if eos_pos < self.max_seq_len:
            tokens[eos_pos] = self.token_system.EOS_IDX
            mask[eos_pos] = 1.0

        return tokens, target, mask, k
