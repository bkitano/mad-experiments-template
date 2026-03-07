"""
Selective Copying Task

Tests ability to selectively copy tokens from input to output based on
marker tokens. The model sees a sequence with some tokens marked, and must
output only the marked tokens in order.

Circuit complexity: In TC^0 (constant-depth threshold circuits can implement
selection + ordering with sufficient width).

This tests the model's ability to route information selectively,
which is a core capability for attention-like mechanisms.
"""

import random

from tasks.base import TokenSystem


class SelectiveCopyTokenSystem(TokenSystem):
    """
    Token system for selective copying.

    Token layout:
    - 0 to vocab_size-1: content tokens
    - vocab_size: MARK (marks a token for copying)
    - vocab_size+1: OUTPUT (separator before output section)
    - vocab_size+2: BOS
    - vocab_size+3: EOS
    - vocab_size+4: PAD

    Input format: [BOS, t1, MARK, t2, t3, MARK, t4, ..., OUTPUT, EOS, PAD, ...]
    Target: at OUTPUT section positions, predict the marked tokens in order
    """

    def __init__(self, vocab_size: int = 64):
        self.vocab_size = vocab_size
        self.num_group_elements = vocab_size

        self.MARK_IDX = vocab_size
        self.OUTPUT_IDX = vocab_size + 1
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
        elif idx == self.MARK_IDX:
            return "<|MARK|>"
        elif idx == self.OUTPUT_IDX:
            return "<|OUTPUT|>"
        else:
            return f"<|{idx}|>"

    def generate_sample(
        self,
        seq_len: int,
        num_marked: int,
    ) -> tuple[list[int], list[int]]:
        """
        Generate a selective copy sample.

        Args:
            seq_len: number of content tokens
            num_marked: number of tokens to mark for copying

        Returns:
            (input_tokens, marked_values)
            input_tokens does NOT include BOS/EOS/PAD
        """
        content = [random.randint(0, self.vocab_size - 1) for _ in range(seq_len)]

        # Pick positions to mark
        mark_positions = sorted(random.sample(range(seq_len), min(num_marked, seq_len)))
        marked_values = [content[p] for p in mark_positions]

        # Build input: interleave content with MARK tokens after marked positions
        seq = []
        for i, tok in enumerate(content):
            seq.append(tok)
            if i in mark_positions:
                seq.append(self.MARK_IDX)

        seq.append(self.OUTPUT_IDX)
        return seq, marked_values
