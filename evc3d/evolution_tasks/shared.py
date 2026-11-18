"""Task definitions: results, rollout policy, and the ``TaskDefinition`` record.

Environment and scoring settings live in ``evolution_tasks.config``. This module
holds what a task module exports around them: its result type, how the shared
engine should roll it out, how its fitness is computed, and the helpers the
task modules share (``make_result``, ``failed_result``, ``excess_volume``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from functools import partial
from typing import Any, Callable, get_args, get_type_hints

from .config import (
    CONFIG_FIELD_OPTIONS,
    DEFAULT_UPRIGHT_ERROR_WEIGHT,
    ENVIRONMENT_FIELD_OPTIONS,
    EnvironmentFamily,
    EvaluationConfig,
)


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics produced by every locomotion task."""

    fitness: float
    origin_distance: float
    average_origin_speed: float
    forward_distance: float
    average_forward_speed: float
    sideways_drift_speed: float
    control_energy: float
    mean_angular_speed: float
    simulated_seconds: float
    actuator_count: int
    body_count: int
    total_volume: float


@dataclass(frozen=True)
class ResultField:
    """One task-specific result value printed by the evaluation CLI."""

    label: str
    attribute: str
    format_spec: str = ".6f"
    none_text: str | None = None


@dataclass(frozen=True)
class RolloutPolicy:
    """Declarative simulation behavior used by the shared evaluation engine."""

    passive_baseline: bool = False
    track_floor_contact: bool = False
    track_root_upright: bool = False
    horizontal_origin_distance: bool = False


@dataclass
class TaskDefinition:
    """Complete public definition exported by one task module."""

    name: str
    config_type: type
    result_type: type
    fitness_callback: Callable
    rollout_policy: RolloutPolicy
    environment: EnvironmentFamily
    title: str
    result_fields: tuple[ResultField, ...]
    order: int
    failed_task_callback: Callable | None = None
    config: EvaluationConfig | None = None
    _default_environment: EnvironmentFamily = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._default_environment = self.environment
        if self.failed_task_callback is None:
            self.failed_task_callback = partial(failed_result, self.result_type)
        if self.config is None:
            self.config = self.config_type()
        elif not isinstance(self.config, self.config_type):
            raise TypeError(
                f"Task {self.name!r} requires {self.config_type.__name__}, "
                f"not {type(self.config).__name__}"
            )

    def args_override(self, args: Any) -> TaskDefinition:
        """Reset task defaults, then apply explicit CLI environment and config values."""
        self.args_override_environment(args)
        self.args_override_config(args)
        return self

    def args_override_environment(self, args: Any) -> EnvironmentFamily:
        """Apply only explicitly supplied environment arguments."""
        overrides = {}
        for name in ENVIRONMENT_FIELD_OPTIONS:
            value = getattr(args, name, None)
            if value is not None:
                overrides[name] = tuple(value) if name == "fluid_coef" else value
        self.environment = replace(self._default_environment, **overrides)
        return self.environment

    def args_override_config(self, args: Any) -> EvaluationConfig:
        """Build a fresh task config and apply supported explicit CLI arguments."""
        config = self.config_type()
        supported = {item.name for item in fields(config)}
        candidates = {name: getattr(args, name, None) for name in CONFIG_FIELD_OPTIONS}
        if getattr(args, "upright_error", False):
            candidates["upright_weight"] = DEFAULT_UPRIGHT_ERROR_WEIGHT
        overrides = {
            name: value
            for name, value in candidates.items()
            if name in supported and value is not None
        }
        self.config = replace(config, **overrides)
        return self.config


TASK_REGISTRY: dict[str, TaskDefinition] = {}
_BUILTIN_TASKS_LOADED = False

DISALLOWED_COLLISION_REASON = "Disallowed non-parent self-collision detected."

SWIMMING_RESULT_FIELDS = (ResultField("Vertical drift speed", "vertical_drift_speed"),)
WALKING_RESULT_FIELDS = (
    ResultField("Height loss", "height_loss"),
    ResultField("Mean upright error", "mean_upright_error"),
)
FLYING_RESULT_FIELDS = (
    ResultField("Height loss", "height_loss"),
    ResultField("First ground contact", "first_ground_contact_time", ".2f", "none"),
    ResultField("Ground touch penalty", "ground_touch_penalty"),
    ResultField("No-ground-touch bonus", "no_ground_touch_bonus"),
    ResultField("Controlled fitness", "controlled_fitness"),
    ResultField("Passive fitness", "passive_fitness"),
    ResultField("Fitness gain", "fitness_gain"),
)


def failure_flags(reason: str) -> dict:
    collision = reason == DISALLOWED_COLLISION_REASON
    return dict(build_failed=not collision, disqualified=collision, failure_reason=reason)


def make_result(result_type: type, fitness: float, metrics: dict):
    """Build ``result_type`` from ``fitness`` and the rollout metrics it declares.

    The rollout reports every metric it can measure; each result type keeps the
    ones it has fields for. A declared field the rollout did not report raises
    ``TypeError``, so a task cannot silently score with a missing metric.
    """
    names = {item.name for item in fields(result_type)}
    return result_type(
        fitness=fitness,
        **{key: value for key, value in metrics.items() if key in names},
    )


def failed_result(result_type: type, config: EvaluationConfig, reason: str):
    """Return the result for a creature that failed to build or was disqualified.

    Fitness is ``config.build_failure_fitness``; every metric is zero (or None
    where the field allows it), so new result fields need no failure-case code.
    """
    hints = get_type_hints(result_type)
    values = {
        item.name: _empty_value(hints[item.name])
        for item in fields(result_type)
        if item.name != "fitness" and item.name not in FAILURE_FLAG_FIELDS
    }
    return result_type(
        fitness=config.build_failure_fitness, **values, **failure_flags(reason)
    )


FAILURE_FLAG_FIELDS = ("build_failed", "disqualified", "failure_reason")


def _empty_value(annotation: Any):
    if annotation is bool:
        return False
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    if type(None) in get_args(annotation):
        return None
    raise TypeError(f"Cannot build an empty result value for type {annotation!r}")


def excess_volume(total_volume: float, cutoff: float) -> float:
    """Return the creature volume above the unpenalized cutoff."""
    return max(0.0, total_volume - cutoff)
