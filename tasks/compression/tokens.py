"""
Compression Task

Tests ability to compress a sequence into a shorter representation
and then decompress it. The model sees a full sequence, then a COMPRESS
token, then must reproduce the sequence from the compressed state.

This tests the model's information-theoretic capacity: how much
information can the hidden state retain?

Circuit complexity: Depends on the structure of the sequence.
Random sequences require high capacity. Structured sequences (with
patterns/repetition) can be compressed more efficiently.
"""

import random

from tasks.base import TokenSystem


class CompressionTokenSystem(TokenSystem):
    """
    Token system for compression/memorization task.

    Token layout:
    - 0 to vocab_size-1: content tokens
    - vocab_size: COMPRESS (separator between input and output)
    - vocab_size+1: BOS
    - vocab_size+2: EOS
    - vocab_size+3: PAD

    Input format: [BOS, t1, t2, ..., tn, COMPRESS, ?, ?, ..., ?, EOS, PAD, ...]
    Target: at the ? positions, reproduce the original sequence
    """

    def __init__(self, vocab_size: int = 64):
        self.vocab_size = vocab_size
        self.num_group_elements = vocab_size

        self.COMPRESS_IDX = vocab_size
        self.BOS_IDX = vocab_size + 1
        self.EOS_IDX = vocab_size + 2
        self.PAD_IDX = vocab_size + 3

        self.num_tokens = vocab_size + 4
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
        elif idx == self.COMPRESS_IDX:
            return "<|COMPRESS|>"
        else:
            return f"<|{idx}|>"

    def generate_sample(
        self,
        seq_len: int,
        structured: bool = False,
        repeat_factor: int = 2,
    ) -> tuple[list[int], list[int]]:
        """
        Generate a compression sample.

        Args:
            seq_len: length of the original sequence
            structured: if True, sequence has repeating patterns
            repeat_factor: for structured sequences, pattern repeats this many times

        Returns:
            (input_seq, target_seq) without BOS/EOS/PAD
        """
        if structured:
            pattern_len = seq_len // repeat_factor
            pattern = [random.randint(0, self.vocab_size - 1) for _ in range(pattern_len)]
            original = (pattern * repeat_factor)[:seq_len]
        else:
            original = [random.randint(0, self.vocab_size - 1) for _ in range(seq_len)]

        # Input: original + COMPRESS + placeholders
        input_seq = original + [self.COMPRESS_IDX]
        target_seq = original  # what the model should reproduce

        return input_seq, target_seq
