"""Singleton module.

Defines a singleton pattern. Use by ReachySDK to prevent multiple instance within on python kernel.

"""
from __future__ import annotations

import typing as t
from typing import Dict

_T = t.TypeVar("_T")


class Singleton(type, t.Generic[_T]):
    """
    @private.
    Singleton pattern. Only one robot can be instancied by python kernel.
    """

    _instances: Dict[Singleton[_T], _T] = {}

    def __call__(cls, *args: t.Any, **kwargs: t.Any) -> _T:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        else:
            raise ConnectionError("Cannot open 2 robot connections in the same kernel.")
        return cls._instances[cls]

    def clear(cls) -> None:
        del cls._instances[cls]
