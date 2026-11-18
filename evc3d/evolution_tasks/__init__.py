"""Built-in task modules, loaded by matching task names to module filenames.

``task_definition("walking_x")`` imports ``evc3d.evolution_tasks.walking_x`` on
first use and returns its ``TASK_DEFINITION`` (a shared, registry-wide object;
copy it before modifying). ``task_names()`` lists every built-in task.
"""

from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules

from . import shared

# Helper modules that are not tasks; every other public module here is one.
NON_TASK_MODULES = frozenset({"shared", "config", "families"})


def _cache_task_definition(
    definition: shared.TaskDefinition,
) -> shared.TaskDefinition:
    existing = shared.TASK_REGISTRY.get(definition.name)
    if existing is not None and existing is not definition:
        raise ValueError(f"Task name {definition.name!r} is already loaded")
    shared.TASK_REGISTRY[definition.name] = definition
    return definition


def _task_module_names() -> tuple[str, ...]:
    return tuple(
        module.name
        for module in sorted(iter_modules(__path__), key=lambda item: item.name)
        if module.name not in NON_TASK_MODULES and not module.name.startswith("_")
    )


def load_task_definition(task: str) -> shared.TaskDefinition:
    """Load a task whose name exactly matches its module filename."""
    if not task.isidentifier() or task.startswith("_"):
        raise KeyError(f"Unknown task: {task!r}")
    existing = shared.TASK_REGISTRY.get(task)
    if existing is not None:
        return existing
    expected_module = f"{__name__}.{task}"
    try:
        module = import_module(f".{task}", package=__name__)
    except ModuleNotFoundError as error:
        if error.name != expected_module:
            raise
        raise KeyError(f"Unknown task: {task!r}") from error
    try:
        definition = module.TASK_DEFINITION
    except AttributeError as error:
        raise ValueError(f"Task module {task!r} does not define TASK_DEFINITION") from error
    if not isinstance(definition, shared.TaskDefinition):
        raise TypeError(f"Task module {task!r} has an invalid TASK_DEFINITION")
    if definition.name != task:
        raise ValueError(
            f"Task module {task!r} declares name {definition.name!r}"
        )
    return _cache_task_definition(definition)


def _load_builtin_tasks() -> None:
    """Load and validate every built-in task module once."""
    if not shared._BUILTIN_TASKS_LOADED:
        for task in _task_module_names():
            load_task_definition(task)
        shared._BUILTIN_TASKS_LOADED = True


def task_definition(task: str) -> shared.TaskDefinition:
    """Return canonical metadata for a task name."""
    return load_task_definition(task)


def task_names() -> tuple[str, ...]:
    """Return registered task names in presentation order."""
    _load_builtin_tasks()
    return tuple(
        definition.name
        for definition in sorted(
            shared.TASK_REGISTRY.values(),
            key=lambda definition: (definition.order, definition.name),
        )
    )
