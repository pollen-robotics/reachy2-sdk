"""Custom dict module.

Defines a custom dict class to modify repr of dict.

"""
from typing import Any, Generic, TypeVar

# Définir un type générique pour les clés et les valeurs
K = TypeVar("K", bound=Any)  # Clés de type str
V = TypeVar("V", bound=Any)  # Valeurs de type str


class CustomDict(dict[K, V], Generic[K, V]):
    """
    Singleton pattern. Only one robot can be instancied by python kernel.
    """

    def __repr__(self) -> str:
        # Construire la représentation personnalisée ligne par ligne
        items = ",\n".join(f"'{k}': {v}" for k, v in self.items())
        return f"""{{{items}}}"""
