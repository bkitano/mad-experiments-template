"""
Config-based training script supporting multiple tasks (Zn, S5).

Usage:
    uv run accelerate launch -m train.run_config --config configs/example.yaml
"""

import argparse
import json
import os
import tempfile
import time
from typing import Protocol

import torch
import torch.nn as nn
import wandb
import yaml
from accelerate import Accelerator
from huggingface_hub import HfApi, whoami
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.deltanet import GroupDeltaNet
from models.transformer import GroupTransformer
from tasks.addition.dataset import ZnCurriculumDataset
from tasks.addition.tokens import ZnTokenSystem
from tasks.s5.dataset import S5CurriculumWrapper
from tasks.s5.tokens import S5TokenSystem
from train.train import evaluate, train_epoch


# =============================================================================
# Token System Protocol
# =============================================================================


class TokenSystemProtocol(Protocol):
    """Protocol for token systems (Zn, S5, etc.)."""

    num_tokens: int
    num_group_elements: int
    EOS_IDX: int
    PAD_IDX: int
    BOS_IDX: int


# =============================================================================
# Config Loading and Helpers
# =============================================================================


def load_config(config_path: str) -> dict:
    """Load config from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_wandb_name(task: str, dataset_cfg: dict, model_cfg: dict, train_cfg: dict) -> str:
    """Build a descriptive wandb run name from config."""
    task_bits = [task.upper()]
    model_bits = [
        model_cfg["type"],
        f"L{model_cfg['num_layers']}",
        f"H{model_cfg['nhead']}",
        f"D{model_cfg['d_model']}",
        f"seq{dataset_cfg['max_seq_len']}",
    ]
    data_bits = [
        f"k{dataset_cfg['max_k']}",
        f"s{dataset_cfg['samples_per_k']}",
    ]
    train_bits = [
        f"bs{train_cfg['batch_size']}",
        f"lr{train_cfg['lr']}",
        f"wd{train_cfg['weight_decay']}",
    ]
    return "-".join(task_bits + model_bits + data_bits + train_bits)


def create_token_system(task: str, dataset_config: dict) -> TokenSystemProtocol:
    """Create token system based on task type."""
    if task == "zn":
        return ZnTokenSystem(n=dataset_config["modulus"])
    elif task == "s5":
        return S5TokenSystem()
    else:
        raise ValueError(f"Unknown task: {task}. Supported tasks: zn, s5")


def create_curriculum(task: str, token_system: TokenSystemProtocol, dataset_config: dict):
    """Create curriculum dataset based on task type."""
    if task == "zn":
        return ZnCurriculumDataset(
            token_system=token_system,
            max_k=dataset_config["max_k"],
            samples_per_k=dataset_config["samples_per_k"],
            max_seq_len=dataset_config["max_seq_len"],
            test_size=dataset_config.get("test_size", 0.2),
        )
    elif task == "s5":
        return S5CurriculumWrapper(
            token_system=token_system,
            max_k=dataset_config["max_k"],
            samples_per_k=dataset_config["samples_per_k"],
            max_seq_len=dataset_config["max_seq_len"],
            test_size=dataset_config.get("test_size", 0.2),
            generator_subset=dataset_config.get("generator_subset", None),  # "swaps", "3perm", or None
            fixed_k=dataset_config.get("fixed_k", None),  # For non-curriculum mode
        )
    else:
        raise ValueError(f"Unknown task: {task}. Supported tasks: zn, s5")


def create_model(model_config: dict, token_system: TokenSystemProtocol) -> nn.Module:
    """Create model based on config."""
    model_type = model_config["type"]

    if model_type == "GroupDeltaNet":
        return GroupDeltaNet(
            num_tokens=token_system.num_tokens,
            num_classes=token_system.num_group_elements,
            num_layers=model_config["num_layers"],
            nhead=model_config["nhead"],
            max_seq_len=model_config.get("max_seq_len", 12),
            d_model=model_config["d_model"],
            dropout=model_config.get("dropout", 0.1),
            eos_idx=token_system.EOS_IDX,
            allow_neg_eigval=model_config.get("allow_neg_eigval", False),
            use_short_conv=model_config.get("use_short_conv", True),
        )
    elif model_type == "GroupTransformer":
        return GroupTransformer(
            num_tokens=token_system.num_tokens,
            num_classes=token_system.num_group_elements,
            max_seq_len=model_config.get("max_seq_len", 12),
            d_model=model_config["d_model"],
            nhead=model_config["nhead"],
            num_layers=model_config["num_layers"],
            dropout=model_config.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# Main Training Function
# =============================================================================


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)

    dataset_config = config["dataset"]
    task = dataset_config.get("task", "zn")
    model_config = config["model"]
    train_config = config["training"]
    logging_config = config.get("logging", {})

    print(f"Task: {task}")
    print(f"Config: {args.config}")

    # Create token system and curriculum dataset
    token_system = create_token_system(task, dataset_config)
    curriculum = create_curriculum(task, token_system, dataset_config)

    print(f"Token system: {token_system.__class__.__name__}")
    print(f"  num_tokens: {token_system.num_tokens}")
    print(f"  num_group_elements: {token_system.num_group_elements}")

    # Add derived values to model config
    model_config["num_tokens"] = token_system.num_tokens
    model_config["num_classes"] = token_system.num_group_elements
    model_config["max_seq_len"] = dataset_config["max_seq_len"]

    # Create model
    model = create_model(model_config, token_system)

    # Optionally compile the model
    if model_config.get("use_compile", False):
        model = torch.compile(model)

    # Setup accelerator
    accelerator = Accelerator(mixed_precision="fp16" if torch.cuda.is_available() else "no")

    wandb_run_id = None
    wandb_run_name = None
    wandb_run_url = None

    # Initialize wandb on main process
    if accelerator.is_main_process:
        wandb_project = logging_config.get("wandb_project", "nc1-tc0-transformer-toy-experiments")
        wandb_run = wandb.init(
            project=wandb_project,
            name=build_wandb_name(task, dataset_config, model_config, train_config),
            config={
                "task": task,
                "dataset": dataset_config,
                "model": model_config,
                "train": train_config,
            },
        )
        wandb_run_id = wandb_run.id
        wandb_run_name = wandb_run.name
        wandb_run_url = wandb_run.url

    # Setup optimizer (explicitly convert to float in case YAML parses as string)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["lr"]),
        betas=(float(train_config.get("beta1", 0.9)), float(train_config.get("beta2", 0.999))),
        eps=float(train_config.get("op_eps", 1e-8)),
        weight_decay=float(train_config.get("weight_decay", 0.01)),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore BOS/EOS/PAD positions

    # Prepare model and optimizer for accelerator
    model, optimizer = accelerator.prepare(model, optimizer)

    global_step = 0

    # Check if curriculum mode is enabled (default: True)
    use_curriculum = dataset_config.get("curriculum", True)
    fixed_k = dataset_config.get("fixed_k", None)  # For non-curriculum, train only at this k

    if not use_curriculum:
        # Non-curriculum mode: train at fixed k only
        train_k = fixed_k if fixed_k else curriculum.num_stages()
        print(f"\n{'='*50}")
        print(f"Non-Curriculum Mode: Training at fixed k={train_k}")
        print(f"{'='*50}")

        # Get datasets for fixed k only
        train_dataset, test_dataset = curriculum.get_fixed_k(train_k)
        print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=train_config["batch_size"], shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=train_config["batch_size"], shuffle=False
        )

        # Prepare data loaders for accelerator
        train_loader, test_loader = accelerator.prepare(train_loader, test_loader)

        max_epochs = train_config.get("max_epochs_per_stage", 100)
        max_val_acc = train_config.get("max_val_acc", 0.99)
        gradient_clip = train_config.get("gradient_clip", 1.0)

        for epoch in tqdm(range(max_epochs), desc=f"k={train_k}"):
            loss, accuracy = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                accelerator=accelerator,
                device=accelerator.device,
                max_grad_norm=gradient_clip,
            )
            val_loss, val_accuracy, val_k_accuracy = evaluate(
                model, test_loader, criterion, device=accelerator.device, accelerator=accelerator
            )
            global_step += 1

            print(
                f"k={train_k} Epoch {epoch + 1}: "
                f"Train Loss: {loss:.4f}, Train Acc: {accuracy:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
            )

            # Log to wandb
            if accelerator.is_main_process:
                log_dict = {
                    "global_step": global_step,
                    "k": train_k,
                    "epoch": epoch + 1,
                    "train/loss": loss,
                    "train/accuracy": accuracy,
                    "val/loss": val_loss,
                    "val/accuracy": val_accuracy,
                }
                for k, acc in val_k_accuracy.items():
                    log_dict[f"val/accuracy_k{k}"] = acc

                wandb.log(log_dict)

            # Check if we've reached target accuracy
            if max_val_acc is not None and val_accuracy >= max_val_acc:
                print(
                    f"Reached target accuracy {max_val_acc:.2%} at epoch {epoch + 1}"
                )
                break

    else:
        # Curriculum training: iterate through stages
        for stage in range(1, curriculum.num_stages() + 1):
            print(f"\n{'='*50}")
            print(f"Curriculum Stage {stage}: k=1 to k={stage}")
            print(f"{'='*50}")

            # Get datasets for this stage
            train_dataset, test_dataset = curriculum.get_stage(stage)
            print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

            # Create data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=train_config["batch_size"], shuffle=True
            )
            test_loader = DataLoader(
                test_dataset, batch_size=train_config["batch_size"], shuffle=False
            )

            # Prepare data loaders for accelerator
            train_loader, test_loader = accelerator.prepare(train_loader, test_loader)

            max_epochs = train_config.get("max_epochs_per_stage", 100)
            max_val_acc = train_config.get("max_val_acc", 0.99)
            gradient_clip = train_config.get("gradient_clip", 1.0)

            for epoch in tqdm(range(max_epochs), desc=f"Stage {stage}"):
                loss, accuracy = train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    accelerator=accelerator,
                    device=accelerator.device,
                    max_grad_norm=gradient_clip,
                )
                val_loss, val_accuracy, val_k_accuracy = evaluate(
                    model, test_loader, criterion, device=accelerator.device, accelerator=accelerator
                )
                global_step += 1

                print(
                    f"Stage {stage} Epoch {epoch + 1}: "
                    f"Train Loss: {loss:.4f}, Train Acc: {accuracy:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
                )

                # Log to wandb
                if accelerator.is_main_process:
                    log_dict = {
                        "global_step": global_step,
                        "stage": stage,
                        "epoch": epoch + 1,
                        "train/loss": loss,
                        "train/accuracy": accuracy,
                        "val/loss": val_loss,
                        "val/accuracy": val_accuracy,
                    }
                    for k, acc in val_k_accuracy.items():
                        log_dict[f"val/accuracy_k{k}"] = acc

                    wandb.log(log_dict)

                # Check if we've reached target accuracy for this stage
                if max_val_acc is not None and val_accuracy >= max_val_acc:
                    print(
                        f"Reached target accuracy {max_val_acc:.2%} at stage {stage} epoch {epoch + 1}"
                    )
                    break

    # Final evaluation on the last stage's test set (or fixed k for non-curriculum)
    if use_curriculum:
        _, final_test = curriculum.get_stage(curriculum.num_stages())
    else:
        train_k = fixed_k if fixed_k else curriculum.num_stages()
        _, final_test = curriculum.get_fixed_k(train_k)
    final_test_loader = DataLoader(
        final_test, batch_size=train_config["batch_size"], shuffle=False
    )
    final_test_loader = accelerator.prepare(final_test_loader)

    final_loss, final_accuracy, final_k_accuracy = evaluate(
        model, final_test_loader, criterion, device=accelerator.device, accelerator=accelerator
    )
    if use_curriculum:
        test_desc = f"k=1 to {curriculum.num_stages()}"
    else:
        test_k = fixed_k if fixed_k else curriculum.num_stages()
        test_desc = f"k={test_k} (fixed)"
    print(
        f"\nFinal Test ({test_desc}): "
        f"Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}"
    )
    print(f"Per-k accuracy: {final_k_accuracy}")

    # Write results.json for harness evaluation
    if accelerator.is_main_process:
        # Count parameters (unwrap compiled model if needed)
        raw_model = model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
        if hasattr(raw_model, "module"):
            raw_model = raw_model.module
        param_count = sum(p.numel() for p in raw_model.parameters())

        results_dict = {
            "task": task,
            "val/accuracy": final_accuracy,
            "val/loss": final_loss,
            "param_count": param_count,
            "wall_time_seconds": time.time() - start_time,
            "global_step": global_step,
        }
        for k, acc in final_k_accuracy.items():
            results_dict[f"val/accuracy_k{k}"] = acc

        results_path = os.path.join(os.path.dirname(args.config), "..", "results.json")
        results_path = os.path.normpath(results_path)
        # Also write to workspace root for harness evaluate.py
        for path in [results_path, "results.json"]:
            with open(path, "w") as f:
                json.dump(results_dict, f, indent=2)
        print(f"\nResults written to results.json")

    # Log final test metrics to wandb
    if accelerator.is_main_process:
        log_dict = {
            "test/final_loss": final_loss,
            "test/final_accuracy": final_accuracy,
        }
        for k, acc in final_k_accuracy.items():
            log_dict[f"test/accuracy_k{k}"] = acc
        wandb.log(log_dict)
        wandb.finish()

    # Push to Hugging Face Hub
    if accelerator.is_main_process:
        repo_id = logging_config.get(
            "hf_repo_id", os.environ.get("HF_REPO_ID", "bkitano/zn-transformer")
        )
        user = whoami()
        print(user)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_id = wandb_run_id or "no-wandb"
            run_dir = os.path.join(tmpdir, run_id)
            os.makedirs(run_dir, exist_ok=True)

            model_path = os.path.join(tmpdir, "model.pt")
            torch.save(model.state_dict(), model_path)

            metadata = {
                "wandb": {
                    "run_id": wandb_run_id,
                    "name": wandb_run_name,
                    "url": wandb_run_url,
                },
                "task": task,
                "dataset": dataset_config,
                "model": model_config,
                "train": train_config,
            }
            metadata_path = os.path.join(run_dir, "config.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, sort_keys=True)
                f.write("\n")

            api = HfApi()
            api.create_repo(repo_id, exist_ok=True)
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=f"runs/{run_id}/model.pt",
                repo_id=repo_id,
            )
            api.upload_file(
                path_or_fileobj=metadata_path,
                path_in_repo=f"runs/{run_id}/config.json",
                repo_id=repo_id,
            )
            print(f"Model pushed to https://huggingface.co/{repo_id}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
