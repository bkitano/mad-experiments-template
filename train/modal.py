"""
uv run modal run --detach -m train.modal
"""

from pathlib import Path
from modal import App, Image, Secret, Volume

N_GPUS = 1
GPU_TYPE = "A100"
image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate>=1.0.0",
        "datasets>=2.14.0",
        "huggingface-hub>=0.20.0",
        "numpy<2",
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


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{N_GPUS}",
    secrets=[Secret.from_name("wandb-secret"), Secret.from_name("huggingface-secret")],
    timeout=28800,  # 8 hours
)
def train_accelerate():
    import subprocess
    import os

    os.chdir("/root/app")

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
            "train.run",
        ]
    )

    print(f"Running command: {' '.join(cmd)}")

    _exec_subprocess(cmd)
