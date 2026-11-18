"""Keep docs/api-reference.json in sync with the code.

The API reference is hand-written (descriptions, examples), but the facts it
repeats from the code are checked here:

* every documented module, class, function, method, and internal function exists;
* documented parameters match the real signature (names, order, defaults);
* every documented dataclass attribute is a real field or attribute;
* every public class and function in a covered module is documented.

When this test fails after a code change, update the JSON entry it names.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import json
from pathlib import Path
import pkgutil

import pytest

import evc3d

REFERENCE_PATH = Path(__file__).resolve().parents[1] / "docs" / "api-reference.json"
REFERENCE = json.loads(REFERENCE_PATH.read_text())

# Modules whose members follow one documented pattern instead of having
# individual entries: each task module (see "evc3d.evolution_tasks") and each
# subcommand module (see "evc3d.cli"). ``__main__`` only calls ``cli.main``.
PATTERN_DOCUMENTED_PREFIXES = (
    "evc3d.evolution_tasks.flying_",
    "evc3d.evolution_tasks.swimming_",
    "evc3d.evolution_tasks.walking_",
    "evc3d.cli.build",
    "evc3d.cli.evaluate",
    "evc3d.cli.evolve",
    "evc3d.cli.mutate",
    "evc3d.cli.view",
)
EXCLUDED_MODULES = {"evc3d.__main__"}

ENTRY_SECTIONS = ("Classes", "Functions")


def _package_modules() -> list[str]:
    return sorted(
        info.name
        for info in pkgutil.walk_packages(evc3d.__path__, "evc3d.")
        if info.name not in EXCLUDED_MODULES
    )


def _covered_modules() -> list[str]:
    return [
        name
        for name in _package_modules()
        if not name.startswith(PATTERN_DOCUMENTED_PREFIXES)
    ]


def _public_members(module_name: str) -> list[str]:
    module = importlib.import_module(module_name)
    return sorted(
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and (inspect.isclass(value) or inspect.isfunction(value))
        and getattr(value, "__module__", None) == module_name
    )


def _documented_entries():
    for module_name, module_doc in REFERENCE.items():
        for section in ENTRY_SECTIONS:
            for name, entry in (module_doc.get(section) or {}).items():
                yield module_name, section, name, entry


def _callable_specs():
    """Yield (label, callable, documented parameters) for every documented signature."""
    for module_name, section, name, entry in _documented_entries():
        module = importlib.import_module(module_name)
        obj = getattr(module, name, None)
        if obj is None:
            continue
        if section == "Classes":
            constructor = entry.get("Constructor") or {}
            if "Parameters" in constructor:
                yield f"{module_name}.{name}", obj, constructor["Parameters"]
            for method_name, method in (entry.get("Methods") or {}).items():
                if isinstance(method, dict) and "Parameters" in method:
                    target = inspect.getattr_static(obj, method_name, None)
                    if isinstance(target, (staticmethod, classmethod)):
                        target = getattr(obj, method_name)
                    if target is not None:
                        yield f"{module_name}.{name}.{method_name}", target, method["Parameters"]
        elif "Parameters" in entry:
            yield f"{module_name}.{name}", obj, entry["Parameters"]


_CALLABLE_SPECS = list(_callable_specs())


def _real_parameters(obj) -> list[inspect.Parameter]:
    parameters = list(inspect.signature(obj).parameters.values())
    return [
        parameter
        for parameter in parameters
        if parameter.name not in {"self", "cls"}
        and parameter.kind not in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}
    ]


def _documented_default_matches(documented: str, parameter: inspect.Parameter, module) -> bool:
    actual = parameter.default
    if actual is inspect.Parameter.empty:
        return documented == "required"
    if documented == "required":
        return False
    if repr(actual) == "<factory>":
        return "default_factory" in documented
    text = {"true": "True", "false": "False", "null": "None"}.get(documented, documented)
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        # A named constant, e.g. "Genotype.DEFAULT_TOPOLOGY_MUTATION_RATE_MIN".
        value = module
        try:
            for part in text.split("."):
                value = getattr(value, part)
        except AttributeError:
            return False
    return _normalize(value) == _normalize(actual)


def _normalize(value):
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def test_reference_documents_every_covered_module():
    missing = [name for name in _covered_modules() if name not in REFERENCE]
    assert not missing, f"Add docs/api-reference.json entries for: {missing}"


def test_reference_modules_exist():
    stale = [name for name in REFERENCE if name not in _package_modules()]
    assert not stale, f"Documented modules that no longer exist: {stale}"


def test_documented_members_exist():
    stale = []
    for module_name, module_doc in REFERENCE.items():
        module = importlib.import_module(module_name)
        names = [
            name
            for section in (*ENTRY_SECTIONS, "Internal_Functions")
            for name in (module_doc.get(section) or {})
        ]
        stale += [f"{module_name}.{name}" for name in names if not hasattr(module, name)]
        for name, entry in (module_doc.get("Classes") or {}).items():
            cls = getattr(module, name, None)
            if cls is None:
                continue
            for section in ("Methods", "Internal_Methods"):
                stale += [
                    f"{module_name}.{name}.{method}"
                    for method in (entry.get(section) or {})
                    if not hasattr(cls, method)
                ]
    assert not stale, f"Documented names that do not exist in the code: {stale}"


def test_public_members_are_documented():
    undocumented = []
    for module_name in _covered_modules():
        documented = {
            name
            for section in ENTRY_SECTIONS
            for name in (REFERENCE.get(module_name, {}).get(section) or {})
        }
        undocumented += [
            f"{module_name}.{name}"
            for name in _public_members(module_name)
            if name not in documented
        ]
    assert not undocumented, f"Public names missing from the API reference: {undocumented}"


@pytest.mark.parametrize(
    ("label", "obj", "documented"),
    _CALLABLE_SPECS,
    ids=[label for label, _obj, _doc in _CALLABLE_SPECS],
)
def test_documented_parameters_match_signature(label, obj, documented):
    real = _real_parameters(obj)
    assert list(documented) == [parameter.name for parameter in real], label

    module = inspect.getmodule(obj)
    wrong_defaults = [
        f"{parameter.name}: documented {documented[parameter.name].get('Default')!r}, "
        f"code {parameter.default!r}"
        for parameter in real
        if "Default" in documented[parameter.name]
        and not _documented_default_matches(
            documented[parameter.name]["Default"], parameter, module
        )
    ]
    assert not wrong_defaults, f"{label}: {wrong_defaults}"


def test_documented_dataclass_attributes_exist():
    stale = []
    for module_name, section, name, entry in _documented_entries():
        if section != "Classes":
            continue
        cls = getattr(importlib.import_module(module_name), name, None)
        if cls is None or not dataclasses.is_dataclass(cls):
            continue
        fields = {field.name for field in dataclasses.fields(cls)}
        stale += [
            f"{module_name}.{name}.{attribute}"
            for attribute in (entry.get("Attributes") or {})
            if attribute not in fields and not hasattr(cls, attribute)
        ]
    assert not stale, f"Documented attributes that do not exist: {stale}"
