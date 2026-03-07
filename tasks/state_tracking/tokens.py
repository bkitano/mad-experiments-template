"""
State Tracking Task

Tests ability to maintain and update internal state through a sequence
of operations. This is the task that most cleanly separates NC^1 from TC^0:
transformers (TC^0) cannot solve general state tracking that requires
unbounded depth.

Variants:
- Permutation state tracking (S5 composition): NC^1-complete
- Counter state tracking (Zn addition): TC^0
- Bit-flip state tracking: NC^1

The key distinction: tasks where the state transition function is a
non-abelian group operation require NC^1 depth, while abelian operations
(like addition) are in TC^0.

This module provides a generalized state tracking framework where you
specify the state space and transition function.
"""

import random
from typing import Callable, Optional

from tasks.base import TokenSystem


class StateTrackingTokenSystem(TokenSystem):
    """
    Generic state tracking token system.

    Parameterized by:
    - num_states: size of state space
    - num_ops: number of distinct operations
    - transition_fn: (state, op) -> new_state

    Token layout:
    - 0 to num_ops-1: operation tokens
    - num_ops to num_ops+num_states-1: state tokens (for output)
    - num_ops+num_states: BOS
    - num_ops+num_states+1: EOS
    - num_ops+num_states+2: PAD
    """

    def __init__(
        self,
        num_states: int,
        num_ops: int,
        transition_fn: Callable[[int, int], int],
        initial_state: int = 0,
    ):
        self.num_states = num_states
        self.num_ops = num_ops
        self.transition_fn = transition_fn
        self.initial_state = initial_state

        self.num_group_elements = num_states  # output predicts state
        self.op_offset = 0
        self.state_offset = num_ops

        self.BOS_IDX = num_ops + num_states
        self.EOS_IDX = num_ops + num_states + 1
        self.PAD_IDX = num_ops + num_states + 2

        self.num_tokens = num_ops + num_states + 3
        self.identity_idx = initial_state

    def get_random_index(self) -> int:
        """Get a random operation index."""
        return random.randint(0, self.num_ops - 1)

    def token_string(self, idx: int) -> str:
        if idx == self.BOS_IDX:
            return "<|BOS|>"
        elif idx == self.EOS_IDX:
            return "<|EOS|>"
        elif idx == self.PAD_IDX:
            return "<|PAD|>"
        elif idx < self.num_ops:
            return f"<|op_{idx}|>"
        else:
            return f"<|state_{idx - self.state_offset}|>"

    def apply_ops(self, ops: list[int]) -> list[int]:
        """Apply a sequence of operations, returning state after each op."""
        state = self.initial_state
        states = []
        for op in ops:
            state = self.transition_fn(state, op)
            states.append(state)
        return states


def make_bitflip_system(num_bits: int = 3) -> StateTrackingTokenSystem:
    """
    Create a bit-flip state tracking system.

    Operations flip individual bits. State is the integer encoding of the bit vector.
    num_states = 2^num_bits, num_ops = num_bits (one flip per bit).
    """
    num_states = 2 ** num_bits
    num_ops = num_bits

    def transition_fn(state: int, op: int) -> int:
        return state ^ (1 << op)

    return StateTrackingTokenSystem(num_states, num_ops, transition_fn)


def make_counter_system(modulus: int = 8) -> StateTrackingTokenSystem:
    """
    Create a modular counter system.

    Operations are increment amounts. State is current count mod n.
    This is in TC^0 (abelian group).
    """
    def transition_fn(state: int, op: int) -> int:
        return (state + op) % modulus

    return StateTrackingTokenSystem(modulus, modulus, transition_fn)
