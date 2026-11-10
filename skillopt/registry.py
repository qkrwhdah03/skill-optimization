"""Generic name -> object registry used by every pluggable component."""
from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


class Registry:
    """A simple string-keyed registry with a decorator-based ``register``."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, Any] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            if name in self._items:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            self._items[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> Any:
        try:
            return self._items[name]
        except KeyError:
            available = ", ".join(self.names()) or "<none>"
            raise KeyError(f"unknown {self.kind} '{name}'; available: {available}") from None

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items
