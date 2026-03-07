"""
Iterated Addition over Z_n (Cyclic Group)

Implements a token system for modular addition:
- Input: sequence of k integers from Z_n
- Output: single integer representing (a_1 + a_2 + ... + a_k) mod n

Key insight: A DFA can simulate this perfectly because:
- States = n (one per group element 0, 1, ..., n-1)
- Transitions = addition (δ(g, h) = (g + h) mod n)
- This is the Cayley machine of Z_n

Unlike S_5 composition (NC^1 complete), addition is in TC^0 since it can be
computed with constant-depth threshold circuits. However, the iterated version
still provides a useful testbed for understanding transformer capabilities.
"""

import random

from tasks.base import TokenSystem


class ZnTokenSystem(TokenSystem):
    """
    Maps between:
    - Integers in Z_n (0 to n-1)
    - Token indices
    - Token strings (e.g., "<|0|>" to "<|n-1|>")

    Token layout:
    - 0 to n-1: Group elements
    - n: BOS (begin-of-sequence)
    - n+1: EOS (end-of-sequence)
    - n+2: PAD (padding)

    Input format: [BOS, a_1, a_2, ..., a_k, EOS, PAD, PAD, ...]
    Output: single class from 0 to n-1 (the sum mod n)
    """

    def __init__(self, n: int = 10):
        """
        Initialize the token system for Z_n.

        Args:
            n: The modulus (group order). Default is 10.
        """
        self.n = n
        self.num_group_elements = n

        # Special token indices (after the n group elements)
        self.BOS_IDX = n
        self.EOS_IDX = n + 1
        self.PAD_IDX = n + 2

        self.num_tokens = n + 3  # n group elements + BOS + EOS + PAD

        # Identity element for addition
        self.identity_idx = 0

        # Build addition table
        self._build_addition_table()

    def _build_addition_table(self):
        """Build the full n x n addition table."""
        self.addition_table = {}
        for i in range(self.n):
            for j in range(self.n):
                self.addition_table[(i, j)] = (i + j) % self.n

    def add_indices(self, i: int, j: int) -> int:
        """Add two elements given their token indices."""
        return self.addition_table[(i, j)]

    def add_sequence(self, indices: list[int]) -> int:
        """Add a sequence of elements (left to right)."""
        result = self.identity_idx
        for idx in indices:
            result = self.add_indices(result, idx)
        return result

    def scan_sequence(self, indices: list[int]) -> list[int]:
        """Return all prefix sums: [a1, a1+a2, a1+a2+a3, ...]."""
        result = self.identity_idx
        scan = []
        for idx in indices:
            result = self.add_indices(result, idx)
            scan.append(result)
        return scan

    def token_string(self, idx: int) -> str:
        """Convert token index to string representation."""
        if idx == self.BOS_IDX:
            return "<|BOS|>"
        elif idx == self.EOS_IDX:
            return "<|EOS|>"
        elif idx == self.PAD_IDX:
            return "<|PAD|>"
        else:
            return f"<|{idx}|>"

    def format_sequence(self, indices: list[int]) -> str:
        """Format a sequence of tokens for logging."""
        tokens = [self.token_string(i) for i in indices]
        return " + ".join(tokens)

    def log_sample(self, indices: list[int], result_idx: int):
        """Log a sample with the addition."""
        seq_str = self.format_sequence(indices)
        result_token = self.token_string(result_idx)
        print(f"  {seq_str} = {result_token} (mod {self.n})")

    def get_random_index(self) -> int:
        """Get a random element index (group element only, 0 to n-1)."""
        return random.randint(0, self.n - 1)


if __name__ == "__main__":
    # Demo
    token_system = ZnTokenSystem(n=10)
    print(f"Z_{token_system.n} Token System")
    print(f"Number of group elements: {token_system.num_group_elements}")
    print(f"Total tokens (with BOS/EOS/PAD): {token_system.num_tokens}")
    print(f"BOS index: {token_system.BOS_IDX}")
    print(f"EOS index: {token_system.EOS_IDX}")
    print(f"PAD index: {token_system.PAD_IDX}")
    print()

    # Test addition
    seq = [3, 7, 5, 2]
    result = token_system.add_sequence(seq)
    print("Example:")
    token_system.log_sample(seq, result)
    print(f"  Expected: (3 + 7 + 5 + 2) mod 10 = {(3 + 7 + 5 + 2) % 10}")
