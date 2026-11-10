"""Optimizer algorithms. Any ``*.py`` dropped in this folder is imported here and self-registers."""
from skillopt.optimizer.base import discover_optimizers

discover_optimizers(__name__)
