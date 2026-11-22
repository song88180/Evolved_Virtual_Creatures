"""The neuron function set of the graph brain (Sims 1994), MuJoCo-free.

Every neuron has a function name, up to three weighted inputs, and (for the
stateful functions) a small private state. ``evaluate_neuron`` computes one
brain step of one neuron. Inputs ``x0, x1, x2`` are already scaled by their
weights. Results are always finite and bounded by ``SIGNAL_LIMIT``.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Tuple

SIGNAL_LIMIT = 1.0e3
EPSILON = 1.0e-6
TWO_PI = 2.0 * math.pi

# name -> number of inputs the function reads (the rest are ignored).
NEURON_ARITY: Dict[str, int] = {
    "sum": 3,
    "product": 2,
    "divide": 2,
    "sum-threshold": 3,
    "greater-than": 2,
    "sign-of": 1,
    "min": 2,
    "max": 2,
    "abs": 1,
    "if": 3,
    "interpolate": 3,
    "sin": 1,
    "cos": 1,
    "atan": 1,
    "log": 1,
    "expt": 2,
    "sigmoid": 1,
    "integrate": 1,
    "differentiate": 1,
    "smooth": 2,
    "memory": 2,
    "oscillate-wave": 2,
    "oscillate-saw": 2,
}
NEURON_FUNCTIONS: Tuple[str, ...] = tuple(NEURON_ARITY)
STATELESS_FUNCTIONS = frozenset(
    {
        "sum",
        "product",
        "divide",
        "sum-threshold",
        "greater-than",
        "sign-of",
        "min",
        "max",
        "abs",
        "if",
        "interpolate",
        "sin",
        "cos",
        "atan",
        "log",
        "expt",
        "sigmoid",
    }
)
MAX_INPUTS = 3


def clamp_signal(value: float) -> float:
    """Return ``value`` limited to ``±SIGNAL_LIMIT``; NaN becomes 0."""
    if value != value:
        return 0.0
    if value > SIGNAL_LIMIT:
        return SIGNAL_LIMIT
    if value < -SIGNAL_LIMIT:
        return -SIGNAL_LIMIT
    return value


def _divide(x0: float, x1: float) -> float:
    return 0.0 if abs(x1) < EPSILON else x0 / x1


def _log(x0: float) -> float:
    return math.log(max(abs(x0), EPSILON))


def _expt(x0: float, x1: float) -> float:
    try:
        return abs(x0) ** max(-8.0, min(8.0, x1))
    except (OverflowError, ZeroDivisionError):
        return SIGNAL_LIMIT


def _sigmoid(x0: float) -> float:
    x0 = max(-60.0, min(60.0, x0))
    return 1.0 / (1.0 + math.exp(-x0))


def _sign(x0: float) -> float:
    return (x0 > 0.0) - (x0 < 0.0)


def _interpolate(x0: float, x1: float, x2: float) -> float:
    t = max(0.0, min(1.0, x2))
    return x0 + (x1 - x0) * t


# Stateless functions: (x0, x1, x2) -> value.
_STATELESS: Dict[str, Callable[[float, float, float], float]] = {
    "sum": lambda x0, x1, x2: x0 + x1 + x2,
    "product": lambda x0, x1, x2: x0 * x1,
    "divide": lambda x0, x1, x2: _divide(x0, x1),
    "sum-threshold": lambda x0, x1, x2: 1.0 if x0 + x1 > x2 else 0.0,
    "greater-than": lambda x0, x1, x2: 1.0 if x0 > x1 else 0.0,
    "sign-of": lambda x0, x1, x2: _sign(x0),
    "min": lambda x0, x1, x2: min(x0, x1),
    "max": lambda x0, x1, x2: max(x0, x1),
    "abs": lambda x0, x1, x2: abs(x0),
    "if": lambda x0, x1, x2: x1 if x0 > 0.0 else x2,
    "interpolate": lambda x0, x1, x2: _interpolate(x0, x1, x2),
    "sin": lambda x0, x1, x2: math.sin(x0),
    "cos": lambda x0, x1, x2: math.cos(x0),
    "atan": lambda x0, x1, x2: math.atan(x0),
    "log": lambda x0, x1, x2: _log(x0),
    "expt": lambda x0, x1, x2: _expt(x0, x1),
    "sigmoid": lambda x0, x1, x2: _sigmoid(x0),
}


def initial_state(function: str) -> List[float]:
    """Return the fresh private state for ``function`` (empty if stateless)."""
    if function in STATELESS_FUNCTIONS:
        return []
    if function == "differentiate":
        return [0.0, 0.0]  # previous input, "has previous" flag
    return [0.0]  # integrator / smoother / memory value / oscillator phase


def evaluate_neuron(
    function: str,
    x0: float,
    x1: float,
    x2: float,
    state: List[float],
    dt: float,
) -> float:
    """Compute one brain step of a neuron and update its ``state`` in place.

    ``dt`` is the duration of the brain step in seconds.

    * ``sum``: ``x0 + x1 + x2``; ``product``: ``x0 * x1``; ``divide``:
      ``x0 / x1`` (0 when ``x1`` is ~0).
    * ``sum-threshold``: 1 if ``x0 + x1 > x2`` else 0; ``greater-than``: 1 if
      ``x0 > x1`` else 0; ``sign-of``: -1, 0 or 1.
    * ``min``, ``max``, ``abs``; ``if``: ``x1`` when ``x0 > 0`` else ``x2``;
      ``interpolate``: ``x0 + (x1 - x0) * clamp(x2, 0, 1)``.
    * ``sin``, ``cos``, ``atan``; ``log``: ``ln|x0|`` (floored); ``expt``:
      ``|x0| ** x1`` (exponent limited to ±8); ``sigmoid``: logistic.
    * ``integrate``: running sum of ``x0 * dt``; ``differentiate``: change of
      ``x0`` per second; ``smooth``: low-pass of ``x0`` with rate
      ``clamp(x1, 0, 1)`` per brain step; ``memory``: stores ``x0`` while
      ``x1 > 0`` and otherwise holds it.
    * ``oscillate-wave``: phase advances ``x0`` radians per second and the
      output is ``sin(phase + x1)``; ``oscillate-saw``: the same phase gives a
      sawtooth in [-1, 1).
    """
    stateless = _STATELESS.get(function)
    if stateless is not None:
        return clamp_signal(stateless(x0, x1, x2))
    if function == "integrate":
        state[0] = clamp_signal(state[0] + x0 * dt)
        return state[0]
    if function == "differentiate":
        if state[1]:
            value = (x0 - state[0]) / dt if dt > 0.0 else 0.0
        else:
            value = 0.0
        state[0] = x0
        state[1] = 1.0
        return clamp_signal(value)
    if function == "smooth":
        rate = max(0.0, min(1.0, x1))
        state[0] = clamp_signal(state[0] + (x0 - state[0]) * rate)
        return state[0]
    if function == "memory":
        if x1 > 0.0:
            state[0] = clamp_signal(x0)
        return state[0]
    if function == "oscillate-wave":
        state[0] = (state[0] + x0 * dt) % TWO_PI
        return math.sin(state[0] + x1)
    if function == "oscillate-saw":
        state[0] = (state[0] + x0 * dt) % TWO_PI
        cycles = (state[0] + x1) / TWO_PI
        return 2.0 * (cycles - math.floor(cycles)) - 1.0
    raise ValueError(f"Unknown neuron function {function!r}")


def evaluate_inputs(
    function: str,
    inputs: Sequence[float],
    state: List[float],
    dt: float,
) -> float:
    """Convenience wrapper: ``evaluate_neuron`` with an input sequence (padded with 0)."""
    padded = list(inputs) + [0.0] * (MAX_INPUTS - len(inputs))
    return evaluate_neuron(function, padded[0], padded[1], padded[2], state, dt)
