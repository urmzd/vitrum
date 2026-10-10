"""Train a single model version.

Hyperparameters come from environment variables (see
vitrum.core.config.Hyperparameters); --version is sugar for
MODEL_VERSION. With --nproc > 1 the run is delegated to
``torchrun --standalone -m vitrum.training`` for DDP.
"""
from __future__ import annotations

import argparse
import os
import subprocess


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vitrum train", description=__doc__)
    parser.add_argument("--version", default=None,
                        help="model version to train (sets MODEL_VERSION)")
    parser.add_argument("--nproc", type=int, default=1,
                        help="number of GPUs; >1 launches torchrun DDP (default: 1)")
    args = parser.parse_args(argv)

    if args.version:
        os.environ["MODEL_VERSION"] = args.version

    if args.nproc > 1:
        cmd = [
            "torchrun", "--standalone",
            f"--nproc_per_node={args.nproc}",
            "-m", "vitrum.training",
        ]
        return subprocess.run(cmd, env=os.environ.copy()).returncode

    from vitrum.training.train import main as train_main
    train_main()
    return 0
