"""
Modal deployment for config-based training.

Usage:
    uv run modal run --detach -m train.modal_config --config configs/example.yaml
"""

import argparse
from pathlib import Path
from modal import App, Image, Secret, Volume
import yaml


def load_config(config_path: str) -> dict:
    """Load config from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Parse config at module load time to configure Modal resources
# Only parse if running locally (not in Modal container)
import os
import sys

N_GPUS = 1
GPU_TYPE = "A100"
TIMEOUT = 28800

# Check if we're running locally (not in Modal container)
# Modal sets MODAL_ENVIRONMENT when running in the container
_is_modal_container = os.environ.get("MODAL_ENVIRONMENT") is not None
_has_config_arg = "--config" in sys.argv or any(arg.startswith("--config=") for arg in sys.argv)

if not _is_modal_container and _has_config_arg:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, _ = parser.parse_known_args()

    config = load_config(args.config)
    deployment_config = config.get("deployment", {})

    N_GPUS = deployment_config.get("n_gpus", 1)
    GPU_TYPE = deployment_config.get("gpu_type", "A100")
    TIMEOUT = deployment_config.get("timeout_seconds", 28800)

image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate>=1.0.0",
        "datasets>=2.14.0",
        "huggingface-hub>=0.20.0",
        "numpy<2",
        "pyyaml>=6.0",
        "torch>=2.4.0",
        "triton>=3.0.0",
        "tqdm>=4.65.0",
        "wandb>=0.17.6",
    )
    .add_local_dir(Path(__file__).parent.parent, remote_path="/root/app", copy=True)
    .run_commands("pip install -e /root/app")
)

volume = Volume.from_name("nc1-results", create_if_missing=True)

app = App("nc1-tc0-transformer-toy-experiments")


@app.local_entrypoint()
def main(config: str):
    """Local entrypoint that triggers the remote training function."""
    # Read config locally and pass as string to remote function
    with open(config, "r") as f:
        config_content = f.read()

    train_with_config.remote(config_content)


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{N_GPUS}",
    secrets=[Secret.from_name("wandb-secret"), Secret.from_name("huggingface-secret")],
    timeout=TIMEOUT,
)
def train_with_config(config_content: str):
    """Run training with the provided config content."""
    import subprocess
    import os
    import tempfile

    os.chdir("/root/app")

    # Write config to a temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    def _exec_subprocess(cmd: list[str]):
        """Executes subprocess and prints log to terminal while subprocess is running."""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with process.stdout as pipe:
            for line in iter(pipe.readline, b""):
                line_str = line.decode()
                print(f"{line_str}", end="")

        exit_code = process.wait()
        if exit_code != 0:
            raise subprocess.CalledProcessError(exit_code, "\n".join(cmd))

    cmd = (
        [
            "accelerate",
            "launch",
        ]
        + (["--multi_gpu"] if N_GPUS > 1 else [])
        + [
            f"--num_processes={N_GPUS}",
            "-m",
            "train.run_config",
            "--config",
            config_path,
        ]
    )

    print(f"Running command: {' '.join(cmd)}")

    _exec_subprocess(cmd)
