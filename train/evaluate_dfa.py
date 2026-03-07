from dataset.dfa import S5DFA
from torch.utils.data import DataLoader
import torch

def evaluate_dfa(
    dfa: S5DFA,
    loader: DataLoader,
) -> tuple[float, dict[int, float]]:
    """Evaluate DFA on a dataset. Returns (accuracy, per-k accuracy dict).

    Input format: [BOS, g_1, ..., g_k, EOS, PAD, ...]
    Extract group elements (positions 1 to k) for DFA processing.
    """
    correct = 0
    total = 0

    k_correct = {}
    k_total = {}

    for tokens, targets, _, ks in loader:
        for i in range(tokens.size(0)):
            k = ks[i].item() if isinstance(ks[i], torch.Tensor) else ks[i]

            # Extract group elements: skip BOS at position 0, take k elements
            # Format: [BOS, g_1, g_2, ..., g_k, EOS, PAD, ...]
            # Group elements are at positions 1 through k (inclusive)
            seq = tokens[i, 1:k+1].tolist()

            # DFA prediction
            pred_idx = dfa.process(seq)

            # Target is the composed permutation index
            target_idx = targets[i].item()

            is_correct = (pred_idx == target_idx)

            correct += int(is_correct)
            total += 1

            if k not in k_correct:
                k_correct[k] = 0
                k_total[k] = 0
            k_correct[k] += int(is_correct)
            k_total[k] += 1

    k_acc = {k: k_correct[k] / k_total[k] for k in sorted(k_correct.keys())}
    return correct / total, k_acc


if __name__ == "__main__":

    from dataset.tokens import S5TokenSystem
    from dataset.dataset import S5CompositionDataset

    token_system = S5TokenSystem()
    dataset = S5CompositionDataset(token_system, k_range=(1, 5), num_samples=1000, max_seq_len=512)
    
    tokens, target, mask, k = dataset[0]
    print(f"Tokens: {tokens}")
    print(f"Target: {target}")
    print(f"Mask: {mask}")
    print(f"K: {k}")
    
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    dfa = S5DFA(token_system)
    accuracy, k_accuracy = evaluate_dfa(dfa, loader)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"K Accuracy: {k_accuracy}")