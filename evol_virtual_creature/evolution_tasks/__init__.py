"""Automatically discovered built-in evolution tasks."""

from importlib import import_module
from pkgutil import iter_modules


def _load_task_modules() -> None:
    for module in sorted(iter_modules(__path__), key=lambda item: item.name):
        if module.name != "shared" and not module.name.startswith("_"):
            import_module(f"{__name__}.{module.name}")


_load_task_modules()

