import random
from dataset.tokens import S5TokenSystem

class S5DFA:
    """
    Deterministic Finite Automaton that computes S_5 composition.

    This is the Cayley machine of S_5:
    - States: Q = S_5 (120 states)
    - Alphabet: Σ = S_5 (120 symbols)
    - Transition: δ(g, h) = g ∘ h
    - Initial state: q_0 = identity
    - All states are accepting (output = current state)

    This DFA computes the composition of any sequence of permutations
    with 100% accuracy, demonstrating that the problem is solvable
    by a finite automaton (hence in NC^1 via parallel prefix).
    """

    def __init__(self, token_system: S5TokenSystem):
        self.token_system = token_system
        self.num_states = token_system.num_group_elements  # 120 (group elements only)
        self.initial_state = token_system.identity_idx

        # Transition table: δ[current_state][input_symbol] = next_state
        # Only for group elements (0-119), not special tokens
        self.transition = [[0] * self.num_states for _ in range(self.num_states)]
        for state in range(self.num_states):
            for symbol in range(self.num_states):
                self.transition[state][symbol] = token_system.compose_indices(state, symbol)

    def process(self, sequence: list[int]) -> int:
        """Process a sequence and return the final state (= composed permutation)."""
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
            # Random length from 1 to max_length
            length = random.randint(1, max_length)

            # Generate random sequence
            sequence = [self.token_system.get_random_index() for _ in range(length)]

            # Compute via DFA
            dfa_result = self.process(sequence)

            # Compute directly
            direct_result = self.token_system.compose_sequence(sequence)

            # Check
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


def demonstrate_dfa_perfection(token_system: S5TokenSystem, max_length: int = 50):
    """Demonstrate that DFA achieves perfect accuracy for any sequence length."""
    print("=" * 70)
    print("DFA CORRECTNESS VERIFICATION FOR S_5 COMPOSITION")
    print("=" * 70)

    dfa = S5DFA(token_system)

    # Test various lengths
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
            direct_result = token_system.compose_sequence(sequence)
            if dfa_result == direct_result:
                correct += 1
            else:
                all_correct = False

        acc = correct / num_tests
        print(f"{length:<10} {num_tests:<10} {acc:.6f}")

    print("-" * 35)
    if all_correct:
        print("✓ DFA achieves PERFECT accuracy on all tested lengths!")
    else:
        print("✗ DFA had some errors (this should never happen)")

    # Show a few example computations
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
    token_system = S5TokenSystem()
    dfa = S5DFA(token_system)
    print(dfa.process([0, 1, 2, 3, 4]))