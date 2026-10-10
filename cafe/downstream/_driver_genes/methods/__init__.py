from .base import BaseDriverMethod, DriverContext
from .de import DEMethod

# Add methods here only after their implementations and tests are ready.
METHODS: dict[str, type[BaseDriverMethod]] = {"de": DEMethod}


def get_method_class(name: str) -> type[BaseDriverMethod]:
    """Resolve a supported method without constructing it."""
    try:
        return METHODS[name]
    except KeyError as error:
        raise KeyError(f"Unknown driver-gene method {name!r}. Available methods: {sorted(METHODS)!r}.") from error


__all__ = ["BaseDriverMethod", "DriverContext", "METHODS", "get_method_class"]
