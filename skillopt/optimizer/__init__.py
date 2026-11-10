"""Optimizer: the user's algorithm (one ``Optimizer`` class) and the plain loop that runs it."""
from .base import OPTIMIZER_REGISTRY, Optimizer, build_optimizer, discover_optimizers, register_optimizer
from .loop import TRAIN_SPLIT, LoopResult, optimize

# import algorithms so they self-register
from . import algorithms
from .algorithms.my_optimizer import MyOptimizer
from .algorithms.opro import OPROOptimizer

__all__ = [
    "Optimizer",
    "OPTIMIZER_REGISTRY",
    "register_optimizer",
    "build_optimizer",
    "discover_optimizers",
    "optimize",
    "LoopResult",
    "TRAIN_SPLIT",
    "MyOptimizer",
    "OPROOptimizer",
]
