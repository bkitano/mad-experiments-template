"""
In-Context Recall Task

Tests associative recall: the model sees key-value pairs, then must recall
the value associated with a queried key.

Variants:
- Direct recall: exact key match
- Fuzzy recall: approximate key match (noisy keys)
- Multi-query: multiple queries per sequence
- Noisy recall: distractors interspersed

Circuit complexity: In TC^0 for direct recall (constant-depth threshold circuits
can implement associative lookup). Fuzzy variants may require more depth.

This is one of the core MAD (Mechanistic Architecture Design) benchmark tasks
that predicts scaling performance.
"""

import random

from tasks.base import TokenSystem


class InContextRecallTokenSystem(TokenSystem):
    """
    Token system for in-context recall.

    Token layout:
    - 0 to vocab_size-1: content tokens (keys and values come from same vocab)
    - vocab_size: SEP (separator between key-value pairs)
    - vocab_size+1: QUERY (marks the query key)
    - vocab_size+2: BOS
    - vocab_size+3: EOS
    - vocab_size+4: PAD

    Input format: [BOS, k1, v1, SEP, k2, v2, SEP, ..., QUERY, k_q, EOS, PAD, ...]
    Output: the value associated with k_q
    """

    def __init__(self, vocab_size: int = 64):
        self.vocab_size = vocab_size
        self.num_group_elements = vocab_size  # output is a value token

        self.SEP_IDX = vocab_size
        self.QUERY_IDX = vocab_size + 1
        self.BOS_IDX = vocab_size + 2
        self.EOS_IDX = vocab_size + 3
        self.PAD_IDX = vocab_size + 4

        self.num_tokens = vocab_size + 5
        self.identity_idx = 0

    def get_random_index(self) -> int:
        return random.randint(0, self.vocab_size - 1)

    def token_string(self, idx: int) -> str:
        if idx == self.BOS_IDX:
            return "<|BOS|>"
        elif idx == self.EOS_IDX:
            return "<|EOS|>"
        elif idx == self.PAD_IDX:
            return "<|PAD|>"
        elif idx == self.SEP_IDX:
            return "<|SEP|>"
        elif idx == self.QUERY_IDX:
            return "<|QUERY|>"
        else:
            return f"<|{idx}|>"

    def generate_sample(
        self,
        num_pairs: int,
        noise_keys: bool = False,
        noise_prob: float = 0.1,
    ) -> tuple[list[int], int, int]:
        """
        Generate a single recall sample.

        Returns:
            (sequence_tokens, query_key, correct_value)
            where sequence_tokens does NOT include BOS/EOS/PAD
        """
        # Generate unique keys
        keys = random.sample(range(self.vocab_size), min(num_pairs, self.vocab_size))
        values = [random.randint(0, self.vocab_size - 1) for _ in range(len(keys))]
        kv_map = dict(zip(keys, values))

        # Build sequence: k1, v1, SEP, k2, v2, SEP, ...
        seq = []
        for k, v in zip(keys, values):
            seq.extend([k, v, self.SEP_IDX])

        # Pick a random key to query
        query_key = random.choice(keys)
        correct_value = kv_map[query_key]

        # Optionally add noise to the query key
        if noise_keys and random.random() < noise_prob:
            query_key = (query_key + random.randint(1, 3)) % self.vocab_size

        seq.extend([self.QUERY_IDX, query_key])
        return seq, query_key, correct_value
