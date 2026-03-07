"""
Selective Copying Dataset

Parametrizable by:
- vocab_size: number of distinct content tokens
- seq_len: length of the content sequence
- num_marked: number of tokens marked for copying
- max_seq_len: maximum padded sequence length
"""

import random

import torch
from torch.utils.data import Dataset

from tasks.selective_copy.tokens import SelectiveCopyTokenSystem


IGNORE_INDEX = -100


class SelectiveCopyDataset(Dataset):
    """
    Dataset for selective copying task.

    Input: [BOS, t1, MARK, t2, t3, MARK, ..., OUTPUT, answer_slots..., EOS, PAD, ...]
    Target: IGNORE everywhere except at answer slot positions
    """

    def __init__(
        self,
        token_system: SelectiveCopyTokenSystem,
        seq_len_range: tuple[int, int],
        num_marked_range: tuple[int, int],
        num_samples: int,
        max_seq_len: int,
    ):
        self.token_system = token_system
        self.seq_len_range = seq_len_range
        self.num_marked_range = num_marked_range
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.data = self._generate()

    def _generate(self) -> list[tuple[list[int], list[int], int]]:
        data = []
        for _ in range(self.num_samples):
            seq_len = random.randint(*self.seq_len_range)
            num_marked = random.randint(
                self.num_marked_range[0],
                min(self.num_marked_range[1], seq_len),
            )
            seq, marked_values = self.token_system.generate_sample(seq_len, num_marked)
            data.append((seq, marked_values, num_marked))
        return data

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        seq, marked_values, num_marked = self.data[idx]

        tokens = torch.full((self.max_seq_len,), self.token_system.PAD_IDX, dtype=torch.long)
        target = torch.full((self.max_seq_len,), IGNORE_INDEX, dtype=torch.long)
        mask = torch.zeros(self.max_seq_len, dtype=torch.float)

        # BOS
        tokens[0] = self.token_system.BOS_IDX
        mask[0] = 1.0

        # Fill input sequence
        pos = 1
        for tok in seq:
            if pos < self.max_seq_len - len(marked_values) - 1:
                tokens[pos] = tok
                mask[pos] = 1.0
                pos += 1

        # Answer slots: model must predict marked values here
        for i, val in enumerate(marked_values):
            if pos < self.max_seq_len - 1:
                tokens[pos] = self.token_system.PAD_IDX  # placeholder
                target[pos] = val
                mask[pos] = 1.0
                pos += 1

        # EOS
        if pos < self.max_seq_len:
            tokens[pos] = self.token_system.EOS_IDX
            mask[pos] = 1.0

        return tokens, target, mask, num_marked


class SelectiveCopyCurriculumDataset:
    """Curriculum wrapper: increase num_marked over stages."""

    def __init__(
        self,
        token_system: SelectiveCopyTokenSystem,
        seq_len: int,
        max_marked: int,
        samples_per_stage: int,
        max_seq_len: int,
        test_size: float = 0.2,
    ):
        self.token_system = token_system
        self.seq_len = seq_len
        self.max_marked = max_marked
        self.samples_per_stage = samples_per_stage
        self.max_seq_len = max_seq_len
        self.test_size = test_size
        self._data_by_k = self._generate_all_data()

    def _generate_all_data(self) -> dict:
        data_by_k = {}
        for k in range(1, self.max_marked + 1):
            samples = []
            for _ in range(self.samples_per_stage):
                seq, marked = self.token_system.generate_sample(self.seq_len, k)
                samples.append((seq, marked, k))
            data_by_k[k] = samples
        return data_by_k

    def get_stage(self, stage_k: int):
        assert 1 <= stage_k <= self.max_marked
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
        train_ds = _SelectiveCopyStageDataset(self.token_system, self.max_seq_len, train_data)
        test_ds = _SelectiveCopyStageDataset(self.token_system, self.max_seq_len, test_data)
        return train_ds, test_ds

    def num_stages(self) -> int:
        return self.max_marked


class _SelectiveCopyStageDataset(Dataset):
    def __init__(self, token_system, max_seq_len, data):
        self.token_system = token_system
        self.max_seq_len = max_seq_len
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, marked_values, num_marked = self.data[idx]

        tokens = torch.full((self.max_seq_len,), self.token_system.PAD_IDX, dtype=torch.long)
        target = torch.full((self.max_seq_len,), IGNORE_INDEX, dtype=torch.long)
        mask = torch.zeros(self.max_seq_len, dtype=torch.float)

        tokens[0] = self.token_system.BOS_IDX
        mask[0] = 1.0

        pos = 1
        for tok in seq:
            if pos < self.max_seq_len - len(marked_values) - 1:
                tokens[pos] = tok
                mask[pos] = 1.0
                pos += 1

        for val in marked_values:
            if pos < self.max_seq_len - 1:
                tokens[pos] = self.token_system.PAD_IDX
                target[pos] = val
                mask[pos] = 1.0
                pos += 1

        if pos < self.max_seq_len:
            tokens[pos] = self.token_system.EOS_IDX
            mask[pos] = 1.0

        return tokens, target, mask, num_marked
