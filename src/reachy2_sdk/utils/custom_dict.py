"""Custom dict module.

Defines a custom dict class to modify repr of dict.

"""
from typing import Any, Generic, TypeVar

K = TypeVar("K", bound=Any)
V = TypeVar("V", bound=Any)


class CustomDict(dict[K, V], Generic[K, V]):
    """
    CustomDict for a nice display
    """

    def __repr__(self) -> str:
        items = ",\n".join(f"'{k}': {v}" for k, v in self.items())
        return f"""{{{items}}}"""
