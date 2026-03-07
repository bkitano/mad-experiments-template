"""
Launch training experiments with YAML config files.

Usage:
    uv run python -m train.launch --config configs/example.yaml
    uv run python -m train.launch --config configs/example.yaml --local  # for local testing
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Launch training with config file")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., configs/example.yaml)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally instead of on Modal",
    )
    parser.add_argument(
        "--no-detach",
        action="store_true",
        help="Don't detach Modal job (wait for completion)",
    )
    args = parser.parse_args()

    # Resolve config path
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print(f"Using config: {config_path}")

    if args.local:
        # Run locally with accelerate
        cmd = [
            "uv",
            "run",
            "accelerate",
            "launch",
            "-m",
            "train.run_config",
            "--config",
            str(config_path),
        ]
    else:
        # Run on Modal
        cmd = [
            "uv",
            "run",
            "modal",
            "run",
        ]
        if not args.no_detach:
            cmd.append("--detach")
        cmd.extend([
            "-m",
            "train.modal_config",
            "--config",
            str(config_path),
        ])

    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
