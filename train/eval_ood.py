"""
Standalone OOD (out-of-distribution) length generalization evaluation.

Loads a trained model from a checkpoint and runs OOD eval without training.

Usage:
    uv run python -m train.eval_ood --config configs/s5_example.yaml
    uv run python -m train.eval_ood --config configs/s5_example.yaml --lengths 10 25 50 100
    uv run python -m train.eval_ood --config configs/s5_example.yaml --samples 5000
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader

from tasks.s5.dataset import S5FixedKDataset, _S5StageDataset
from tasks.s5.tokens import S5TokenSystem
from train.run_config import create_model, create_token_system, load_config, CHECKPOINT_DIR
from train.train import evaluate


def main():
    parser = argparse.ArgumentParser(description="Run OOD eval on a trained model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--lengths", type=int, nargs="+", default=None, help="OOD lengths to test (overrides config)")
    parser.add_argument("--samples", type=int, default=None, help="Samples per length (overrides config)")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory (default: ./checkpoints)")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_config = config["dataset"]
    model_config = config["model"]
    train_config = config["training"]
    task = dataset_config.get("task", "zn")

    if task != "s5":
        print(f"OOD eval currently only supports s5, got: {task}")
        return

    ood_config = config.get("ood_test", {})
    ood_lengths = args.lengths or ood_config.get("lengths", [10, 25, 50])
    ood_samples = args.samples or ood_config.get("samples_per_length", 10_000)
    max_seq_len = dataset_config["max_seq_len"]

    # Filter lengths that fit in max_seq_len
    valid_lengths = [k for k in ood_lengths if k + 2 <= max_seq_len]
    skipped = [k for k in ood_lengths if k + 2 > max_seq_len]
    if skipped:
        print(f"Skipping lengths {skipped} (exceed max_seq_len {max_seq_len})")
    ood_lengths = valid_lengths

    if not ood_lengths:
        print("No valid OOD lengths to test.")
        return

    # Setup
    token_system = create_token_system(task, dataset_config)
    accelerator = Accelerator(mixed_precision="fp16" if torch.cuda.is_available() else "no")

    # Create and load model
    model_config["max_seq_len"] = dataset_config["max_seq_len"]
    model = create_model(model_config, token_system)
    if model_config.get("use_compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_config["lr"]))
    model, optimizer = accelerator.prepare(model, optimizer)

    ckpt_dir = args.checkpoint_dir or str(CHECKPOINT_DIR)
    if not Path(ckpt_dir).exists():
        print(f"No checkpoint found at {ckpt_dir}")
        return

    print(f"Loading model from {ckpt_dir}")
    accelerator.load_state(ckpt_dir)

    # Determine generator subset
    gen_subset_name = dataset_config.get("generator_subset", None)
    if gen_subset_name == "swaps":
        gen_subset = token_system.get_swap_indices()
    elif gen_subset_name == "3perm":
        gen_subset = token_system.get_3perm_indices()
    else:
        gen_subset = None

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    ood_results = {}

    print(f"\n{'='*50}")
    print(f"OOD Length Generalization (max_seq_len={max_seq_len})")
    print(f"{'='*50}")

    for test_k in sorted(ood_lengths):
        ood_dataset = S5FixedKDataset(
            token_system=token_system,
            k=test_k,
            num_samples=ood_samples,
            max_seq_len=max_seq_len,
            generator_subset=gen_subset,
        )
        ood_data = []
        for tokens_list, _result in ood_dataset.data:
            scan = token_system.scan_sequence(tokens_list)
            ood_data.append((tokens_list, scan, test_k))
        ood_stage = _S5StageDataset(token_system, max_seq_len, ood_data)

        ood_loader = DataLoader(
            ood_stage, batch_size=train_config["batch_size"], shuffle=False
        )
        ood_loader = accelerator.prepare(ood_loader)

        ood_loss, ood_acc, ood_k_acc = evaluate(
            model, ood_loader, criterion,
            device=accelerator.device, accelerator=accelerator,
        )
        ood_results[test_k] = {"accuracy": ood_acc, "loss": ood_loss}
        print(f"  k={test_k:>4d}: accuracy={ood_acc:.4f}  loss={ood_loss:.4f}")

    # Write results
    if accelerator.is_main_process and ood_results:
        results_dict = {"task": task, "eval_type": "ood_only"}
        for k, metrics in sorted(ood_results.items()):
            results_dict[f"ood/accuracy_k{k}"] = metrics["accuracy"]
            results_dict[f"ood/loss_k{k}"] = metrics["loss"]

        out_path = "ood_results.json"
        with open(out_path, "w") as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
