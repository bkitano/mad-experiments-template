"""
DFA for Iterated Addition over Z_n

This is the Cayley machine of Z_n:
- States: Q = Z_n (n states: 0, 1, ..., n-1)
- Alphabet: Σ = Z_n (n symbols)
- Transition: δ(g, h) = (g + h) mod n
- Initial state: q_0 = 0 (identity)
- All states are accepting (output = current state)

This DFA computes the sum of any sequence of integers mod n
with 100% accuracy.
"""

import random
from tasks.addition.tokens import ZnTokenSystem


class ZnDFA:
    """
    Deterministic Finite Automaton that computes iterated addition over Z_n.
    """

    def __init__(self, token_system: ZnTokenSystem):
        self.token_system = token_system
        self.n = token_system.n
        self.num_states = self.n
        self.initial_state = token_system.identity_idx  # 0

        # Transition table: δ[current_state][input_symbol] = next_state
        # Only for group elements (0 to n-1), not special tokens
        self.transition = [[0] * self.n for _ in range(self.n)]
        for state in range(self.n):
            for symbol in range(self.n):
                self.transition[state][symbol] = token_system.add_indices(state, symbol)

    def process(self, sequence: list[int]) -> int:
        """Process a sequence and return the final state (= sum mod n)."""
        state = self.initial_state
        for symbol in sequence:
            state = self.transition[state][symbol]
        return state

    def process_batch(self, sequences: list[list[int]]) -> list[int]:
        """Process a batch of sequences."""
        return [self.process(seq) for seq in sequences]

    def verify_correctness(self, num_tests: int = 10000, max_length: int = 100) -> dict:
        """
        Verify DFA correctness on random sequences of various lengths.
        Returns statistics about the verification.
        """
        results = {
            "total_tests": num_tests,
            "correct": 0,
            "by_length": {},
        }

        for _ in range(num_tests):
            length = random.randint(1, max_length)
            sequence = [self.token_system.get_random_index() for _ in range(length)]

            dfa_result = self.process(sequence)
            direct_result = self.token_system.add_sequence(sequence)

            if dfa_result == direct_result:
                results["correct"] += 1
                if length not in results["by_length"]:
                    results["by_length"][length] = {"correct": 0, "total": 0}
                results["by_length"][length]["correct"] += 1
                results["by_length"][length]["total"] += 1
            else:
                if length not in results["by_length"]:
                    results["by_length"][length] = {"correct": 0, "total": 0}
                results["by_length"][length]["total"] += 1

        results["accuracy"] = results["correct"] / results["total_tests"]
        return results


def demonstrate_dfa_perfection(token_system: ZnTokenSystem, max_length: int = 50):
    """Demonstrate that DFA achieves perfect accuracy for any sequence length."""
    print("=" * 70)
    print(f"DFA CORRECTNESS VERIFICATION FOR Z_{token_system.n} ADDITION")
    print("=" * 70)

    dfa = ZnDFA(token_system)

    lengths_to_test = [1, 2, 5, 10, 20, 50, 100, 500, 1000]
    lengths_to_test = [l for l in lengths_to_test if l <= max_length or l <= 100]

    print(f"\nTesting DFA on sequences of various lengths...")
    print(f"{'Length':<10} {'Tests':<10} {'Accuracy':<15}")
    print("-" * 35)

    all_correct = True
    for length in lengths_to_test:
        num_tests = min(1000, 10000 // length)
        correct = 0

        for _ in range(num_tests):
            sequence = [token_system.get_random_index() for _ in range(length)]
            dfa_result = dfa.process(sequence)
            direct_result = token_system.add_sequence(sequence)
            if dfa_result == direct_result:
                correct += 1
            else:
                all_correct = False

        acc = correct / num_tests
        print(f"{length:<10} {num_tests:<10} {acc:.6f}")

    print("-" * 35)
    if all_correct:
        print("DFA achieves PERFECT accuracy on all tested lengths!")
    else:
        print("DFA had some errors (this should never happen)")

    print("\n" + "-" * 70)
    print("EXAMPLE COMPUTATIONS:")
    print("-" * 70)

    for length in [2, 3, 5]:
        sequence = [token_system.get_random_index() for _ in range(length)]
        result = dfa.process(sequence)

        print(f"\nSequence (length {length}):")
        token_system.log_sample(sequence, result)

    return dfa


if __name__ == "__main__":
    token_system = ZnTokenSystem(n=10)
    dfa = ZnDFA(token_system)

    # Test
    seq = [3, 7, 5, 2]
    result = dfa.process(seq)
    print(f"DFA result for {seq}: {result}")
    print(f"Expected: {sum(seq) % 10}")

    print()
    demonstrate_dfa_perfection(token_system, max_length=100)
