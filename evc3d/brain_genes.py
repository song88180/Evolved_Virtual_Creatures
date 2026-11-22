"""Genes of the graph brain: neurons, effectors, and signal sources.

A creature's brain is a nested graph (Sims 1994). Each body-part gene
(``NodeGene``) holds a ``BrainGene`` of local neurons and effectors that is
copied into every instance of that part, and the ``Genotype`` holds a set of
global neurons that is instantiated once. Sensors are not stored: every part
implicitly has joint-angle sensors (one per joint degree of freedom) and six
contact sensors (one per face), and any input may point at them.

A signal source is a string::

    <scope>:<item>

``scope`` says which part instance to read from, relative to the instance that
owns the input:

* ``self``: the same part; ``parent``: its parent part
* ``child:<connection>``: the first child part made by that connection
* ``global``: the global neuron set
* ``root``: the creature's root part
* ``node:<node>``: the first instance of that node gene

``item`` is a neuron name (``n3``), a joint-angle sensor (``angle0``), or a
contact sensor (``contact+x``). Sources that cannot be resolved when the
phenotype is built (a missing parent, a pruned neuron) read as 0. An input
with no source is a constant equal to its weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, List, Mapping, Optional

from .brain_functions import MAX_INPUTS, NEURON_ARITY, NEURON_FUNCTIONS

SENSOR_ANGLE = "angle"
SENSOR_CONTACT = "contact"
CONTACT_FACES = ("+x", "-x", "+y", "-y", "+z", "-z")
SCOPES = ("self", "parent", "child", "global", "root", "node")
NEURON_NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
DOF_BY_JOINT_TYPE = {"hinge": 1, "slide": 1, "ball": 3}
MAX_ANGLE_DOFS = 3


@dataclass(frozen=True)
class ParsedSource:
    scope: str
    scope_arg: Optional[str]
    kind: str  # "neuron", "angle" or "contact"
    item: str  # neuron name, DOF index as text, or face name


def parse_source(source: str) -> ParsedSource:
    """Parse and validate a source string; raise ``ValueError`` if malformed."""
    if not isinstance(source, str) or ":" not in source:
        raise ValueError(f"Invalid brain source {source!r}")
    scope, *middle, item = source.split(":")
    if scope not in SCOPES:
        raise ValueError(f"Unknown brain source scope in {source!r}")
    scope_arg: Optional[str] = None
    if scope in ("child", "node"):
        if len(middle) != 1 or not middle[0]:
            raise ValueError(f"Brain source {source!r} needs a {scope} name")
        scope_arg = middle[0]
    elif middle:
        raise ValueError(f"Invalid brain source {source!r}")
    if item.startswith(SENSOR_CONTACT):
        face = item[len(SENSOR_CONTACT):]
        if face not in CONTACT_FACES:
            raise ValueError(f"Unknown contact face in brain source {source!r}")
        return ParsedSource(scope, scope_arg, SENSOR_CONTACT, face)
    if item.startswith(SENSOR_ANGLE):
        index_text = item[len(SENSOR_ANGLE):]
        if not index_text.isdigit() or int(index_text) >= MAX_ANGLE_DOFS:
            raise ValueError(f"Unknown angle sensor in brain source {source!r}")
        return ParsedSource(scope, scope_arg, SENSOR_ANGLE, index_text)
    if NEURON_NAME_PATTERN.fullmatch(item) is None:
        raise ValueError(f"Invalid neuron name in brain source {source!r}")
    return ParsedSource(scope, scope_arg, "neuron", item)


def format_source(
    scope: str,
    item: str,
    scope_arg: Optional[str] = None,
) -> str:
    """Build a source string (inverse of ``parse_source``)."""
    middle = f":{scope_arg}" if scope_arg is not None else ""
    return f"{scope}{middle}:{item}"


def _finite(value: Any, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


@dataclass
class BrainInput:
    """One neuron input: ``weight * signal(source)``, or the constant ``weight``."""

    source: Optional[str] = None
    weight: float = 0.0

    def __post_init__(self):
        self.weight = _finite(self.weight, "brain input weight")
        if self.source is not None:
            parse_source(self.source)


@dataclass
class BrainEffector:
    """Drives one joint degree of freedom with ``weight * signal(source)``."""

    source: Optional[str] = None
    weight: float = 0.0

    def __post_init__(self):
        self.weight = _finite(self.weight, "brain effector weight")
        if self.source is not None:
            parse_source(self.source)


@dataclass
class BrainNeuron:
    """A function node with ``MAX_INPUTS`` weighted inputs (extras are ignored)."""

    name: str
    function: str = "sum"
    inputs: List[BrainInput] = field(default_factory=list)

    def __post_init__(self):
        if NEURON_NAME_PATTERN.fullmatch(self.name or "") is None or self.name.startswith(
            (SENSOR_ANGLE, SENSOR_CONTACT)
        ):
            raise ValueError(f"Invalid neuron name {self.name!r}")
        if self.function not in NEURON_ARITY:
            valid = ", ".join(NEURON_FUNCTIONS)
            raise ValueError(
                f"Unknown neuron function {self.function!r}; expected one of {valid}"
            )
        inputs = [_coerce(BrainInput, item) for item in self.inputs]
        if len(inputs) > MAX_INPUTS:
            raise ValueError(f"A neuron has at most {MAX_INPUTS} inputs")
        inputs.extend(BrainInput() for _ in range(MAX_INPUTS - len(inputs)))
        self.inputs = inputs

    @property
    def used_inputs(self) -> List[BrainInput]:
        """The inputs the neuron's function actually reads."""
        return self.inputs[: NEURON_ARITY[self.function]]


@dataclass
class BrainGene:
    """The local neurons and joint effectors owned by one body-part gene."""

    neurons: List[BrainNeuron] = field(default_factory=list)
    effectors: List[BrainEffector] = field(default_factory=list)

    def __post_init__(self):
        self.neurons = [_coerce(BrainNeuron, item) for item in self.neurons]
        self.effectors = [_coerce(BrainEffector, item) for item in self.effectors]
        names = [neuron.name for neuron in self.neurons]
        if len(set(names)) != len(names):
            raise ValueError("Brain neuron names must be unique within a part")

    def neuron_named(self, name: str) -> Optional[BrainNeuron]:
        return next((n for n in self.neurons if n.name == name), None)

    def new_neuron_name(self) -> str:
        return next_neuron_name(self.neurons)


def next_neuron_name(neurons: List[BrainNeuron]) -> str:
    """Return an unused ``n<k>`` name for a list of neurons."""
    used = {neuron.name for neuron in neurons}
    index = 0
    while f"n{index}" in used:
        index += 1
    return f"n{index}"


def coerce_neurons(neurons: List[Any]) -> List[BrainNeuron]:
    """Turn dict entries (from JSON) into neurons and check names are unique."""
    result = [_coerce(BrainNeuron, item) for item in neurons]
    names = [neuron.name for neuron in result]
    if len(set(names)) != len(names):
        raise ValueError("Global neuron names must be unique")
    return result


def _coerce(cls, item):
    if isinstance(item, cls):
        return item
    if isinstance(item, Mapping):
        return cls(**dict(item))
    raise TypeError(f"Cannot build {cls.__name__} from {type(item).__name__}")


def joint_dofs(joint_type: str) -> int:
    """Number of controllable degrees of freedom of an articulated joint (else 0)."""
    return DOF_BY_JOINT_TYPE.get(joint_type, 0)

