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
import itertools

from tasks.base import TokenSystem


# =============================================================================
# S_5 GROUP UTILITIES
# =============================================================================

def generate_all_permutations(n: int = 5) -> list[tuple[int, ...]]:
    """Generate all n! permutations of {0, 1, ..., n-1}."""
    return list(itertools.permutations(range(n)))


def compose_permutations(p1: tuple[int, ...], p2: tuple[int, ...]) -> tuple[int, ...]:
    """
    Compose two permutations: (p1 ∘ p2)(x) = p1(p2(x))
    This is right-to-left composition (standard mathematical convention).
    """
    return tuple(p1[p2[i]] for i in range(len(p1)))


def inverse_permutation(p: tuple[int, ...]) -> tuple[int, ...]:
    """Compute the inverse of a permutation."""
    inv = [0] * len(p)
    for i, val in enumerate(p):
        inv[val] = i
    return tuple(inv)


def to_cycle_notation(perm: tuple[int, ...]) -> str:
    """
    Convert a permutation to cycle notation.

    Examples:
        (1, 0, 2, 3, 4) -> "(0 1)"
        (0, 1, 2, 3, 4) -> "()" [identity]
        (1, 2, 0, 4, 3) -> "(0 1 2)(3 4)"
    """
    n = len(perm)
    visited = [False] * n
    cycles = []

    for start in range(n):
        if visited[start]:
            continue

        cycle = []
        i = start
        while not visited[i]:
            visited[i] = True
            cycle.append(i)
            i = perm[i]

        if len(cycle) > 1:
            cycles.append(cycle)

    if not cycles:
        return "()"  # Identity

    return "".join(f"({' '.join(map(str, c))})" for c in cycles)


def permutation_order(perm: tuple[int, ...]) -> int:
    """Compute the order of a permutation (smallest k s.t. p^k = identity)."""
    identity = tuple(range(len(perm)))
    current = perm
    order = 1
    while current != identity:
        current = compose_permutations(perm, current)
        order += 1
    return order


# =============================================================================
# S_5 TOKEN SYSTEM
# =============================================================================

class S5TokenSystem(TokenSystem):
    """
    Maps between:
    - Permutation tuples (e.g., (1, 0, 2, 3, 4))
    - Token indices (0 to 119)
    - Token strings (e.g., "<|t_1|>" to "<|t_120|>")
    - Cycle notation (e.g., "(0 1)")

    Token layout:
    - 0-119: Group elements (permutations)
    - 120: BOS (begin-of-sequence)
    - 121: EOS (end-of-sequence)
    - 122: PAD (padding)

    Input format: [BOS, g_1, g_2, ..., g_k, EOS, PAD, PAD, ...]
    Output: single class from 0-119 (the product)
    """

    # Special token indices (after the 120 group elements)
    BOS_IDX = 120
    EOS_IDX = 121
    PAD_IDX = 122

    def __init__(self):
        self.n = 5
        self.all_perms = generate_all_permutations(self.n)
        self.num_group_elements = len(self.all_perms)  # 120
        self.num_tokens = self.num_group_elements + 3  # 123 (120 + BOS + EOS + PAD)

        # Bijections (only for group elements 0-119)
        self.perm_to_idx = {p: i for i, p in enumerate(self.all_perms)}
        self.idx_to_perm = {i: p for i, p in enumerate(self.all_perms)}

        # Find identity
        self.identity_perm = tuple(range(self.n))
        self.identity_idx = self.perm_to_idx[self.identity_perm]

        # Precompute cycle notations
        self.idx_to_cycle = {i: to_cycle_notation(p) for i, p in enumerate(self.all_perms)}

        # Build composition (Cayley) table
        self._build_composition_table()

    def _build_composition_table(self):
        """Build the full 120x120 composition table (group elements only)."""
        self.composition_table = {}
        for i in range(self.num_group_elements):
            for j in range(self.num_group_elements):
                p1 = self.idx_to_perm[i]
                p2 = self.idx_to_perm[j]
                composed = compose_permutations(p1, p2)
                self.composition_table[(i, j)] = self.perm_to_idx[composed]

    def compose_indices(self, i: int, j: int) -> int:
        """Compose two permutations given their token indices."""
        return self.composition_table[(i, j)]

    def compose_sequence(self, indices: list[int]) -> int:
        """Compose a sequence of permutations (left to right)."""
        result = self.identity_idx
        for idx in indices:
            result = self.compose_indices(result, idx)
        return result

    def scan_sequence(self, indices: list[int]) -> list[int]:
        """Return all prefix compositions: [g1, g1∘g2, g1∘g2∘g3, ...]."""
        result = self.identity_idx
        scan = []
        for idx in indices:
            result = self.compose_indices(result, idx)
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
            return f"<|t_{idx + 1}|>"

    def format_sequence(self, indices: list[int], show_cycles: bool = True) -> str:
        """Format a sequence of tokens for logging."""
        tokens = [self.token_string(i) for i in indices]
        if show_cycles:
            cycles = [self.idx_to_cycle[i] for i in indices]
            return " ∘ ".join(f"{t}={c}" for t, c in zip(tokens, cycles))
        return " ∘ ".join(tokens)

    def log_sample(self, indices: list[int], result_idx: int):
        """Log a sample with full cycle notation."""
        seq_str = self.format_sequence(indices)
        result_cycle = self.idx_to_cycle[result_idx]
        result_token = self.token_string(result_idx)
        print(f"  {seq_str} = {result_token}={result_cycle}")

    def get_random_index(self) -> int:
        """Get a random permutation index (group element only, 0-119)."""
        return random.randint(0, self.num_group_elements - 1)

    def get_generators(self) -> list[int]:
        """
        Return indices of standard generators for S_5.
        S_5 is generated by:
        - σ = (0 1 2 3 4) - 5-cycle
        - τ = (0 1) - transposition
        """
        # Find the 5-cycle (0 1 2 3 4)
        five_cycle = (1, 2, 3, 4, 0)
        five_cycle_idx = self.perm_to_idx[five_cycle]

        # Find the transposition (0 1)
        transposition = (1, 0, 2, 3, 4)
        transposition_idx = self.perm_to_idx[transposition]

        return [five_cycle_idx, transposition_idx]

    def get_swap_indices(self) -> list[int]:
        """
        Return indices of permutations that permute at most 2 elements.
        This includes: identity + all 10 transpositions = 11 elements.

        Following Grazzi et al. "S5 only swaps" setup.
        """
        indices = []
        for idx, perm in self.idx_to_perm.items():
            # Count how many elements are moved
            moved = sum(1 for i, p in enumerate(perm) if i != p)
            if moved <= 2:  # Identity (0 moved) or transposition (2 moved)
                indices.append(idx)
        return sorted(indices)

    def get_3perm_indices(self) -> list[int]:
        """
        Return indices of permutations that permute at most 3 elements.
        This includes: identity + transpositions + 3-cycles = 1 + 10 + 20 = 31 elements.

        Following Grazzi et al. "S5 swaps, 3-permutations" setup.
        """
        indices = []
        for idx, perm in self.idx_to_perm.items():
            # Count how many elements are moved
            moved = sum(1 for i, p in enumerate(perm) if i != p)
            if moved <= 3:  # Identity, transposition, or 3-cycle
                indices.append(idx)
        return sorted(indices)

