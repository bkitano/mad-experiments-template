import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from tasks.addition.tokens import ZnTokenSystem
from tasks.addition.dataset import ZnCurriculumDataset
# from models.transformer import GroupTransformer
from models.deltanet import GroupDeltaNet
from accelerate import Accelerator
from huggingface_hub import HfApi, whoami
import tempfile
import json
import os
from train.train import evaluate, train_epoch

"""
uv run accelerate launch -m train.run
"""


def main():
    dataset_config = {
        "type": "ZnCurriculumDataset",
        "max_k": 5,
        "samples_per_k": 100_000,
        "max_seq_len": 12,
        "modulus": 10,
        "test_size": 0.2,
    }
    token_system = ZnTokenSystem(n=dataset_config["modulus"])
    curriculum = ZnCurriculumDataset(
        token_system=token_system,
        max_k=dataset_config["max_k"],
        samples_per_k=dataset_config["samples_per_k"],
        max_seq_len=dataset_config["max_seq_len"],
        test_size=dataset_config["test_size"],
    )

    # Model configuration
    model_config = {
        "type": "GroupDeltaNet",
        "num_tokens": token_system.num_tokens,
        "num_classes": token_system.num_group_elements,
        "max_seq_len": dataset_config["max_seq_len"],
        "num_layers": 1,
        "nhead": 1,
        "d_model": 128,
        "dropout": 0.1,
        "use_compile": True,
    }

    model = GroupDeltaNet(
        num_tokens=model_config["num_tokens"],
        num_classes=model_config["num_classes"],
        num_layers=model_config["num_layers"],
        nhead=model_config["nhead"],
        max_seq_len=model_config["max_seq_len"],
        d_model=model_config["d_model"],
        dropout=model_config["dropout"],
        eos_idx=token_system.EOS_IDX,
    )

    # Optionally compile the model for faster training
    if model_config["use_compile"]:
        model = torch.compile(model)

    # Training configuration
    train_config = {
        "batch_size": 2048,
        "lr": 1e-3,
        "beta1": 0.9,
        "beta2": 0.999,
        "op_eps": 1e-8,
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
        "max_val_acc": 0.99,
        "max_epochs_per_stage": 100,
    }

    accelerator = Accelerator(mixed_precision="fp16" if torch.cuda.is_available() else "no")

    def build_wandb_name(dataset_cfg, model_cfg, train_cfg):
        model_bits = [
            model_cfg["type"],
            f"L{model_cfg['num_layers']}",
            f"H{model_cfg['nhead']}",
            f"D{model_cfg['d_model']}",
            f"seq{model_cfg['max_seq_len']}",
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
        return "-".join(model_bits + data_bits + train_bits)

    wandb_run_id = None
    wandb_run_name = None
    wandb_run_url = None

    # Initialize wandb on main process
    if accelerator.is_main_process:
        wandb_run = wandb.init(
            project="nc1-tc0-transformer-toy-experiments",
            name=build_wandb_name(dataset_config, model_config, train_config),
            config={
                "dataset": dataset_config,
                "model": model_config,
                "train": train_config,
            },
        )
        wandb_run_id = wandb_run.id
        wandb_run_name = wandb_run.name
        wandb_run_url = wandb_run.url

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["lr"],
        betas=(train_config["beta1"], train_config["beta2"]),
        eps=train_config["op_eps"],
        weight_decay=train_config["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    # Prepare model and optimizer for accelerator
    model, optimizer = accelerator.prepare(model, optimizer)

    global_step = 0

    # Curriculum training: iterate through stages
    for stage in range(1, curriculum.num_stages() + 1):
        print(f"\n{'='*50}")
        print(f"Curriculum Stage {stage}: k=1 to k={stage}")
        print(f"{'='*50}")

        # Get datasets for this stage
        train_dataset, test_dataset = curriculum.get_stage(stage)
        print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

        # Create data loaders (shuffle=True since data is already mixed)
        train_loader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=train_config["batch_size"], shuffle=False)

        # Prepare data loaders for accelerator
        train_loader, test_loader = accelerator.prepare(train_loader, test_loader)

        for epoch in tqdm(range(train_config["max_epochs_per_stage"]), desc=f"Stage {stage}"):
            loss, accuracy = train_epoch(
                model, train_loader, optimizer, criterion,
                accelerator=accelerator,
                device=accelerator.device,
                max_grad_norm=train_config["gradient_clip"] if train_config["gradient_clip"] else 1.0,
            )
            val_loss, val_accuracy, val_k_accuracy = evaluate(
                model, test_loader, criterion,
                device=accelerator.device, accelerator=accelerator
            )
            global_step += 1

            print(f"Stage {stage} Epoch {epoch + 1}: Train Loss: {loss:.4f}, Train Acc: {accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

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
            if train_config["max_val_acc"] is not None and val_accuracy >= train_config["max_val_acc"]:
                print(f"Reached target accuracy {train_config['max_val_acc']:.2%} at stage {stage} epoch {epoch + 1}")
                break

    # Final evaluation on the last stage's test set
    final_train, final_test = curriculum.get_stage(curriculum.num_stages())
    final_test_loader = DataLoader(final_test, batch_size=train_config["batch_size"], shuffle=False)
    final_test_loader = accelerator.prepare(final_test_loader)

    final_loss, final_accuracy, final_k_accuracy = evaluate(
        model, final_test_loader, criterion,
        device=accelerator.device, accelerator=accelerator
    )
    print(f"\nFinal Test (k=1 to {curriculum.num_stages()}): Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")
    print(f"Per-k accuracy: {final_k_accuracy}")
    
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
        repo_id = os.environ.get("HF_REPO_ID", "bkitano/zn-transformer")
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
