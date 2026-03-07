import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from accelerate import Accelerator
from tqdm import tqdm


IGNORE_INDEX = -100  # Standard ignore index for CrossEntropyLoss


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    max_grad_norm: float = 1.0,
    accelerator: Optional[Accelerator] = None,
) -> tuple[float, float]:
    """Train for one epoch. Returns (loss, accuracy).

    For scan output:
    - logits: (batch, seq_len, num_classes)
    - targets: (batch, seq_len) with IGNORE_INDEX at BOS/EOS/PAD positions

    If accelerator is provided, uses it for backward pass and gradient clipping.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for tokens, targets, masks, _ in tqdm(loader, desc="Epoch"):
        # With accelerator, data is already on correct device
        if accelerator is None:
            tokens = tokens.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(tokens, masks)  # (batch, seq_len, num_classes)

        # Reshape for cross-entropy: (batch*seq, classes) vs (batch*seq,)
        batch_size, seq_len, num_classes = logits.shape
        logits_flat = logits.view(-1, num_classes)
        targets_flat = targets.view(-1)

        # Cross-entropy with ignore_index handles masked positions
        loss = criterion(logits_flat, targets_flat)

        # Backward pass
        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()

        # Gradient clipping (Grazzi: 1.0)
        if accelerator is not None:
            accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item() * batch_size

        # Accuracy: only count non-ignored positions
        preds_flat = logits_flat.argmax(dim=-1)
        valid_mask = targets_flat != IGNORE_INDEX
        correct += ((preds_flat == targets_flat) & valid_mask).sum().item()
        total += valid_mask.sum().item()

    return total_loss / (total / seq_len), correct / total if total > 0 else 0.0


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    accelerator: Optional[Accelerator] = None,
) -> tuple[float, float, dict[int, float]]:
    """Evaluate model. Returns (loss, accuracy, per-k accuracy dict).

    For scan output:
    - logits: (batch, seq_len, num_classes)
    - targets: (batch, seq_len) with IGNORE_INDEX at BOS/EOS/PAD positions
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    k_correct = {}
    k_total = {}

    with torch.no_grad():
        for tokens, targets, masks, ks in loader:
            # With accelerator, data is already on correct device
            if accelerator is None:
                tokens = tokens.to(device)
                targets = targets.to(device)
                masks = masks.to(device)

            logits = model(tokens, masks)  # (batch, seq_len, num_classes)
            batch_size, seq_len, num_classes = logits.shape

            # Reshape for cross-entropy
            logits_flat = logits.view(-1, num_classes)
            targets_flat = targets.view(-1)

            # Loss
            loss = criterion(logits_flat, targets_flat)
            total_loss += loss.item() * batch_size

            # Accuracy: only count non-ignored positions
            preds_flat = logits_flat.argmax(dim=-1)
            valid_mask = targets_flat != IGNORE_INDEX
            correct += ((preds_flat == targets_flat) & valid_mask).sum().item()
            total += valid_mask.sum().item()

            # Track per-k accuracy (based on final position prediction for each sample)
            # For each sample, check if the last valid prediction (at position k) is correct
            preds = logits.argmax(dim=-1)  # (batch, seq_len)
            for i, k in enumerate(ks):
                k_val = k.item() if isinstance(k, torch.Tensor) else k
                if k_val not in k_correct:
                    k_correct[k_val] = 0
                    k_total[k_val] = 0
                # Position k (1-indexed in seq after BOS) is where the final prefix result is
                final_pos = k_val  # targets[i, k_val] has the final composition
                if targets[i, final_pos] != IGNORE_INDEX:
                    is_correct = (preds[i, final_pos] == targets[i, final_pos]).item()
                    k_correct[k_val] += is_correct
                    k_total[k_val] += 1

    k_acc = {k: k_correct[k] / k_total[k] if k_total[k] > 0 else 0.0 for k in sorted(k_correct.keys())}
    avg_loss = total_loss / (total / seq_len) if total > 0 else 0.0
    avg_acc = correct / total if total > 0 else 0.0
    return avg_loss, avg_acc, k_acc
