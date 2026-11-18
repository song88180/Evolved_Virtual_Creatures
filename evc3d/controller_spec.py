"""Controller specifications without the MuJoCo runtime.

Controller sizes, the ``ActuatorController`` record built by
``PhenotypeBuilder``, and neural-weight helpers. Genotype loading and mutation
use this module, so they do not import MuJoCo; ``evc3d.control`` applies
these controllers to a running simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


NEURAL_HIDDEN_SIZE = 16
NEURAL_INPUT_DIMS = {
    "hinge": 12,
    "slide": 12,
    "ball": 17,
}
NEURAL_OUTPUT_DIMS = {
    "hinge": 1,
    "slide": 1,
    "ball": 3,
}
BALL_NEURAL_AXES = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


@dataclass
class ActuatorController:
    motor_names: tuple[str, ...]
    joint_name: str
    joint_type: str
    control_mode: str
    ctrlranges: tuple[tuple[float, float], ...]
    neural_w1: tuple[tuple[float, ...], ...] = ()
    neural_b1: tuple[float, ...] = ()
    neural_w2: tuple[tuple[float, ...], ...] = ()
    neural_b2: tuple[float, ...] = ()
    amp: float = 0.0
    freq: float = 0.0  # sine angular frequency in rad/s (divide by 2*pi for Hz)
    phase: float = 0.0


def zero_neural_parameters(
    joint_type: str,
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
]:
    input_dim = NEURAL_INPUT_DIMS[joint_type]
    output_dim = NEURAL_OUTPUT_DIMS[joint_type]
    return (
        tuple((0.0,) * input_dim for _ in range(NEURAL_HIDDEN_SIZE)),
        (0.0,) * NEURAL_HIDDEN_SIZE,
        tuple((0.0,) * NEURAL_HIDDEN_SIZE for _ in range(output_dim)),
        (0.0,) * output_dim,
    )


def normalized_neural_parameters(
    joint_type: str,
    neural_w1: Sequence[Sequence[float]],
    neural_b1: Sequence[float],
    neural_w2: Sequence[Sequence[float]],
    neural_b2: Sequence[float],
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
]:
    if not neural_w1 and not neural_b1 and not neural_w2 and not neural_b2:
        return zero_neural_parameters(joint_type)

    input_dim = NEURAL_INPUT_DIMS[joint_type]
    output_dim = NEURAL_OUTPUT_DIMS[joint_type]
    w1 = _normalize_matrix(neural_w1, NEURAL_HIDDEN_SIZE, input_dim, "neural_w1")
    b1 = _normalize_vector(neural_b1, NEURAL_HIDDEN_SIZE, "neural_b1")
    w2 = _normalize_matrix(neural_w2, output_dim, NEURAL_HIDDEN_SIZE, "neural_w2")
    b2 = _normalize_vector(neural_b2, output_dim, "neural_b2")
    return w1, b1, w2, b2


def _normalize_matrix(
    matrix: Sequence[Sequence[float]],
    rows: int,
    columns: int,
    field_name: str,
) -> tuple[tuple[float, ...], ...]:
    if len(matrix) != rows:
        raise ValueError(f"{field_name} must contain {rows} rows")
    normalized = tuple(
        _normalize_vector(row, columns, f"{field_name} row")
        for row in matrix
    )
    return normalized


def _normalize_vector(
    vector: Sequence[float],
    length: int,
    field_name: str,
) -> tuple[float, ...]:
    if len(vector) != length:
        raise ValueError(f"{field_name} must contain {length} values")
    normalized = tuple(float(value) for value in vector)
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError(f"{field_name} values must be finite")
    return normalized
