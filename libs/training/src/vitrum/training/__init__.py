"""Vitrum training loop.

Single-GPU / CPU:      python -m vitrum.training
Multi-GPU (DDP):       torchrun --standalone --nproc_per_node=N -m vitrum.training

All hyperparameters come from environment variables via
:class:`vitrum.core.config.Hyperparameters`. Architectures are resolved
through the registry, so any installed architecture package is trainable.
"""
from vitrum.training.train import main

__all__ = ["main"]
