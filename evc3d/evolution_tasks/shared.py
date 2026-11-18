"""Shared constants and registration metadata for built-in tasks."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class EnvironmentFamily:
    """Declarative MuJoCo physics and initial placement for one task."""

    name: str = "swimming"
    timestep: float = 0.01
    gravity: float = 0.0
    fluid_density: float | None = 1000.0
    fluid_viscosity: float | None = 0.001
    fluid_shape: str | None = None
    fluid_coef: Sequence[float] | None = None
    body_density: float = 500.0
    body_friction: Sequence[float] = (1.0, 0.5, 0.5)
    creature_contype: int = 0
    creature_conaffinity: int = 0
    self_collision_contype: int = 2
    self_collision_conaffinity: int = 2
    floor_position: Sequence[float] = (0.0, 0.0, -0.05)
    floor_size: Sequence[float] = (5.0, 5.0, 0.1)
    floor_contype: int = 0
    floor_conaffinity: int = 0
    initial_root_position: Sequence[float] = (0.0, 0.0, 0.6)
    initial_floor_clearance: float | None = None
    supports_scheduled_gravity: bool = False
    initialization_callback: Callable | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError("environment name must be a non-empty identifier")
        if self.timestep <= 0.0:
            raise ValueError("environment timestep must be positive")
        if self.fluid_density is not None and self.fluid_density < 0.0:
            raise ValueError("fluid_density must be non-negative")
        if self.fluid_viscosity is not None and self.fluid_viscosity < 0.0:
            raise ValueError("fluid_viscosity must be non-negative")
        if self.fluid_coef is not None and len(self.fluid_coef) != 5:
            raise ValueError("fluid_coef must contain exactly five values")
        if len(self.body_friction) != 3:
            raise ValueError("body_friction must contain exactly three values")
        if any(value < 0.0 for value in self.body_friction):
            raise ValueError("body_friction values must be non-negative")
        if len(self.floor_position) != 3 or len(self.floor_size) != 3:
            raise ValueError("floor position and size must contain three values")
        if len(self.initial_root_position) != 3:
            raise ValueError("initial_root_position must contain three values")
        if self.initial_floor_clearance is not None and self.initial_floor_clearance < 0:
            raise ValueError("initial_floor_clearance must be non-negative")
        collision_bits = (
            self.creature_contype,
            self.creature_conaffinity,
            self.self_collision_contype,
            self.self_collision_conaffinity,
            self.floor_contype,
            self.floor_conaffinity,
        )
        if any(value < 0 for value in collision_bits):
            raise ValueError("collision bits must be non-negative")


DEFAULT_ENVIRONMENT = EnvironmentFamily()


@dataclass(frozen=True)
class EvaluationConfig:
    """Common simulation, safety, and complexity settings for every task."""

    settle_seconds: float = 0.0
    max_creature_height: float = 0.0
    min_center_height_fraction: float = 0.0
    initial_floor_contact_policy: str = "allow"
    episode_seconds: float = 10.0
    max_node: int = 500
    self_collision: bool = False
    disallow_collision: bool = False
    target_direction: Sequence[float] = (1.0, 0.0, 0.0)
    body_count_weight: float = 0.001
    volume_weight: float = 0.01
    volume_penalty_cutoff: float = 0.1
    min_body_volume: float = 1e-6
    min_total_volume: float = 0.0
    max_volume: float = 1.0
    build_failure_fitness: float = -1_000.0
    max_abs_state_value: float = 1_000_000.0
    max_abs_velocity: float = 1_000.0
    max_abs_acceleration: float = 100_000.0


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
    failed_task_callback: Callable
    rollout_policy: RolloutPolicy
    environment: EnvironmentFamily
    title: str
    result_fields: tuple[ResultField, ...]
    order: int
    config: EvaluationConfig | None = None
    _default_environment: EnvironmentFamily = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._default_environment = self.environment
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
        for name in ("fluid_density", "fluid_viscosity", "fluid_shape", "fluid_coef"):
            value = getattr(args, name, None)
            if value is not None:
                overrides[name] = tuple(value) if name == "fluid_coef" else value
        self.environment = replace(self._default_environment, **overrides)
        return self.environment

    def args_override_config(self, args: Any) -> EvaluationConfig:
        """Build a fresh task config and apply supported explicit CLI arguments."""
        config = self.config_type()
        supported = {item.name for item in fields(config)}
        candidates = {
            "episode_seconds": getattr(args, "duration", None),
            "max_node": getattr(args, "max_node", None),
            "body_count_weight": getattr(args, "body_count_weight", None),
            "volume_weight": getattr(args, "volume_weight", None),
            "volume_penalty_cutoff": getattr(args, "volume_penalty_cutoff", None),
            "min_body_volume": getattr(args, "min_body_volume", None),
            "min_total_volume": getattr(args, "min_total_volume", None),
            "max_volume": getattr(args, "max_volume", None),
            "self_collision": getattr(args, "self_collision", None),
            "disallow_collision": getattr(args, "disallow_collision", None),
            "fitness_gain_fraction": getattr(args, "fitness_gain_fraction", None),
        }
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


DEFAULT_MIN_BODY_VOLUME = 1e-6
DEFAULT_MIN_TOTAL_VOLUME = 0.0
DEFAULT_FLYING_MIN_TOTAL_VOLUME = 1e-4
DEFAULT_UPRIGHT_ERROR_WEIGHT = 0.2
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


def excess_volume(total_volume: float, cutoff: float) -> float:
    """Return the creature volume above the unpenalized cutoff."""
    return max(0.0, total_volume - cutoff)
