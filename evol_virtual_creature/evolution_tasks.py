"""Built-in locomotion task configurations and evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .constants import EnvironmentFamily
from .evaluation import (
    ResultField,
    _evaluate_flying_away,
    _evaluate_origin_distance,
    _evaluate_walking_away,
    _evaluate_x_axis_flying,
    _evaluate_x_axis_swimming,
    _evaluate_x_axis_walking,
    register_task,
)
from .genotype import Genotype
from .phenotype import (
    DEFAULT_FLYING_FLUID_COEF,
    DEFAULT_FLYING_FLUID_DENSITY,
    DEFAULT_FLYING_FLUID_SHAPE,
    DEFAULT_FLYING_FLUID_VISCOSITY,
    DEFAULT_FLYING_GRAVITY,
)


_DEFAULT_MIN_BODY_VOLUME = 1e-6
_DEFAULT_MIN_TOTAL_VOLUME = 0.0
_DEFAULT_FLYING_MIN_TOTAL_VOLUME = 1e-4


@dataclass(frozen=True)
class SwimmingEvaluationConfig:
    """Weights and simulation settings for x-axis swimming fitness."""

    episode_seconds: float = 10.0
    max_node: int = 500
    self_collision: bool = False
    disallow_collision: bool = False
    target_direction: Sequence[float] = (1.0, 0.0, 0.0)
    forward_speed_weight: float = 1.0
    energy_weight: float = 1e-6
    sideways_drift_weight: float = 0.1
    vertical_drift_weight: float = 0.1
    angular_speed_weight: float = 0.01
    body_count_weight: float = 0.001
    volume_weight: float = 0.01
    volume_penalty_cutoff: float = 0.1
    min_body_volume: float = _DEFAULT_MIN_BODY_VOLUME
    min_total_volume: float = _DEFAULT_MIN_TOTAL_VOLUME
    max_volume: float = 1.0
    build_failure_fitness: float = -1_000.0
    max_abs_state_value: float = 1_000_000.0
    max_abs_velocity: float = 1_000.0
    max_abs_acceleration: float = 100_000.0


@dataclass(frozen=True)
class WalkingEvaluationConfig:
    """Weights and simulation settings for x-axis walking fitness."""

    episode_seconds: float = 10.0
    settle_seconds: float = 1.0
    max_node: int = 500
    self_collision: bool = False
    disallow_collision: bool = False
    target_direction: Sequence[float] = (1.0, 0.0, 0.0)
    forward_speed_weight: float = 1.0
    energy_weight: float = 1e-7
    sideways_drift_weight: float = 0.1
    angular_speed_weight: float = 0.01
    upright_weight: float = 0.0
    height_loss_weight: float = 0.2
    min_center_height_fraction: float = 0.5
    max_creature_height: float = 10.0
    body_count_weight: float = 0.001
    volume_weight: float = 0.01
    volume_penalty_cutoff: float = 0.1
    min_body_volume: float = _DEFAULT_MIN_BODY_VOLUME
    min_total_volume: float = _DEFAULT_MIN_TOTAL_VOLUME
    max_volume: float = 1.0
    build_failure_fitness: float = -1_000.0
    max_abs_state_value: float = 1_000_000.0
    max_abs_velocity: float = 1_000.0
    max_abs_acceleration: float = 100_000.0


@dataclass(frozen=True)
class SwimmingAwayEvaluationConfig:
    """Weights and simulation settings for swimming-away fitness."""

    episode_seconds: float = 10.0
    max_node: int = 500
    self_collision: bool = False
    disallow_collision: bool = False
    target_direction: Sequence[float] = (1.0, 0.0, 0.0)
    speed_weight: float = 1.0
    energy_weight: float = 1e-6
    angular_speed_weight: float = 0.01
    body_count_weight: float = 0.001
    volume_weight: float = 0.01
    volume_penalty_cutoff: float = 0.1
    min_body_volume: float = _DEFAULT_MIN_BODY_VOLUME
    min_total_volume: float = _DEFAULT_MIN_TOTAL_VOLUME
    max_volume: float = 1.0
    build_failure_fitness: float = -1_000.0
    max_abs_state_value: float = 1_000_000.0
    max_abs_velocity: float = 1_000.0
    max_abs_acceleration: float = 100_000.0


@dataclass(frozen=True)
class WalkingAwayEvaluationConfig:
    """Weights and simulation settings for walking-away fitness."""

    episode_seconds: float = 10.0
    settle_seconds: float = 1.0
    max_node: int = 500
    self_collision: bool = False
    disallow_collision: bool = False
    target_direction: Sequence[float] = (1.0, 0.0, 0.0)
    speed_weight: float = 1.0
    energy_weight: float = 1e-7
    angular_speed_weight: float = 0.01
    height_loss_weight: float = 0.2
    min_center_height_fraction: float = 0.5
    max_creature_height: float = 10.0
    body_count_weight: float = 0.001
    volume_weight: float = 0.01
    volume_penalty_cutoff: float = 0.1
    min_body_volume: float = _DEFAULT_MIN_BODY_VOLUME
    min_total_volume: float = _DEFAULT_MIN_TOTAL_VOLUME
    max_volume: float = 1.0
    build_failure_fitness: float = -1_000.0
    max_abs_state_value: float = 1_000_000.0
    max_abs_velocity: float = 1_000.0
    max_abs_acceleration: float = 100_000.0


@dataclass(frozen=True)
class FlyingEvaluationConfig:
    """Weights and simulation settings for x-axis flying fitness."""

    episode_seconds: float = 10.0
    fluid_density: float = DEFAULT_FLYING_FLUID_DENSITY
    fluid_viscosity: float = DEFAULT_FLYING_FLUID_VISCOSITY
    fluid_shape: str = DEFAULT_FLYING_FLUID_SHAPE
    fluid_coef: Sequence[float] = DEFAULT_FLYING_FLUID_COEF
    gravity: float = DEFAULT_FLYING_GRAVITY
    max_node: int = 500
    self_collision: bool = False
    disallow_collision: bool = False
    target_direction: Sequence[float] = (1.0, 0.0, 0.0)
    distance_weight: float = 0.1
    energy_weight: float = 1e-7
    angular_speed_weight: float = 0.01
    height_loss_weight: float = 0.5
    ground_touch_weight: float = 0.0
    no_ground_touch_bonus: float = 10
    fitness_gain_fraction: float = 0.5
    body_count_weight: float = 0.001
    volume_weight: float = 0.01
    volume_penalty_cutoff: float = 0.1
    min_body_volume: float = _DEFAULT_MIN_BODY_VOLUME
    min_total_volume: float = _DEFAULT_FLYING_MIN_TOTAL_VOLUME
    max_volume: float = 1.0
    build_failure_fitness: float = -1_000.0
    max_abs_state_value: float = 1_000_000.0
    max_abs_velocity: float = 1_000.0
    max_abs_acceleration: float = 100_000.0


@dataclass(frozen=True)
class FlyingAwayEvaluationConfig:
    """Weights and simulation settings for flying-away fitness."""

    episode_seconds: float = 10.0
    fluid_density: float = DEFAULT_FLYING_FLUID_DENSITY
    fluid_viscosity: float = DEFAULT_FLYING_FLUID_VISCOSITY
    fluid_shape: str = DEFAULT_FLYING_FLUID_SHAPE
    fluid_coef: Sequence[float] = DEFAULT_FLYING_FLUID_COEF
    gravity: float = DEFAULT_FLYING_GRAVITY
    max_node: int = 500
    self_collision: bool = False
    disallow_collision: bool = False
    target_direction: Sequence[float] = (1.0, 0.0, 0.0)
    speed_weight: float = 0.1
    energy_weight: float = 1e-7
    angular_speed_weight: float = 0.01
    height_loss_weight: float = 0.5
    ground_touch_weight: float = 0.0
    no_ground_touch_bonus: float = 10
    fitness_gain_fraction: float = 0.5
    body_count_weight: float = 0.001
    volume_weight: float = 0.01
    volume_penalty_cutoff: float = 0.1
    min_body_volume: float = _DEFAULT_MIN_BODY_VOLUME
    min_total_volume: float = _DEFAULT_FLYING_MIN_TOTAL_VOLUME
    max_volume: float = 1.0
    build_failure_fitness: float = -1_000.0
    max_abs_state_value: float = 1_000_000.0
    max_abs_velocity: float = 1_000.0
    max_abs_acceleration: float = 100_000.0


@dataclass(frozen=True)
class SwimmingEvaluationResult:
    fitness: float
    origin_distance: float
    average_origin_speed: float
    forward_distance: float
    average_forward_speed: float
    sideways_drift_speed: float
    vertical_drift_speed: float
    control_energy: float
    mean_angular_speed: float
    simulated_seconds: float
    actuator_count: int
    body_count: int
    total_volume: float
    build_failed: bool = False
    disqualified: bool = False
    failure_reason: str | None = None


@dataclass(frozen=True)
class WalkingEvaluationResult:
    fitness: float
    origin_distance: float
    average_origin_speed: float
    forward_distance: float
    average_forward_speed: float
    sideways_drift_speed: float
    height_loss: float
    control_energy: float
    mean_angular_speed: float
    mean_upright_error: float
    simulated_seconds: float
    actuator_count: int
    body_count: int
    total_volume: float
    build_failed: bool = False
    disqualified: bool = False
    failure_reason: str | None = None


@dataclass(frozen=True)
class OriginDistanceEvaluationResult(SwimmingEvaluationResult):
    """Result metrics for swimming-away evaluation."""


@dataclass(frozen=True)
class FlyingEvaluationResult:
    fitness: float
    origin_distance: float
    average_origin_speed: float
    forward_distance: float
    average_forward_speed: float
    sideways_drift_speed: float
    height_loss: float
    first_ground_contact_time: float | None
    ground_touch_penalty: float
    no_ground_touch_bonus: float
    controlled_fitness: float
    passive_fitness: float
    fitness_gain: float
    control_energy: float
    mean_angular_speed: float
    simulated_seconds: float
    actuator_count: int
    body_count: int
    total_volume: float
    build_failed: bool = False
    disqualified: bool = False
    failure_reason: str | None = None


_SWIMMING_RESULT_FIELDS = (ResultField("Vertical drift speed", "vertical_drift_speed"),)
_WALKING_RESULT_FIELDS = (
    ResultField("Height loss", "height_loss"),
    ResultField("Mean upright error", "mean_upright_error"),
)
_FLYING_RESULT_FIELDS = (
    ResultField("Height loss", "height_loss"),
    ResultField("First ground contact", "first_ground_contact_time", ".2f", "none"),
    ResultField("Ground touch penalty", "ground_touch_penalty"),
    ResultField("No-ground-touch bonus", "no_ground_touch_bonus"),
    ResultField("Controlled fitness", "controlled_fitness"),
    ResultField("Passive fitness", "passive_fitness"),
    ResultField("Fitness gain", "fitness_gain"),
)


@register_task(
    "swimming_x",
    config_type=SwimmingEvaluationConfig,
    environment_family=EnvironmentFamily.SWIMMING,
    title="X-axis swimming_x evaluation",
    result_fields=_SWIMMING_RESULT_FIELDS,
)
def evaluate_x_axis_swimming(
    genotype: Genotype, config: SwimmingEvaluationConfig | None = None
):
    return _evaluate_x_axis_swimming(genotype, config or SwimmingEvaluationConfig())


@register_task(
    "swimming_away",
    config_type=SwimmingAwayEvaluationConfig,
    environment_family=EnvironmentFamily.SWIMMING,
    title="Distance-from-origin swimming_away evaluation",
    result_fields=_SWIMMING_RESULT_FIELDS,
)
def evaluate_origin_distance(
    genotype: Genotype, config: SwimmingAwayEvaluationConfig | None = None
):
    return _evaluate_origin_distance(genotype, config or SwimmingAwayEvaluationConfig())


@register_task(
    "walking_x",
    config_type=WalkingEvaluationConfig,
    environment_family=EnvironmentFamily.WALKING,
    title="X-axis walking_x evaluation",
    result_fields=_WALKING_RESULT_FIELDS,
)
def evaluate_x_axis_walking(
    genotype: Genotype, config: WalkingEvaluationConfig | None = None
):
    return _evaluate_x_axis_walking(genotype, config or WalkingEvaluationConfig())


@register_task(
    "walking_away",
    config_type=WalkingAwayEvaluationConfig,
    environment_family=EnvironmentFamily.WALKING,
    title="Distance-from-origin walking_away evaluation",
    result_fields=_WALKING_RESULT_FIELDS,
)
def evaluate_walking_away(
    genotype: Genotype, config: WalkingAwayEvaluationConfig | None = None
):
    return _evaluate_walking_away(genotype, config or WalkingAwayEvaluationConfig())


@register_task(
    "flying_x",
    config_type=FlyingEvaluationConfig,
    environment_family=EnvironmentFamily.FLYING,
    title="X-axis flying_x evaluation",
    result_fields=_FLYING_RESULT_FIELDS,
)
def evaluate_x_axis_flying(
    genotype: Genotype, config: FlyingEvaluationConfig | None = None
):
    return _evaluate_x_axis_flying(genotype, config or FlyingEvaluationConfig())


@register_task(
    "flying_away",
    config_type=FlyingAwayEvaluationConfig,
    environment_family=EnvironmentFamily.FLYING,
    title="Distance-from-origin flying_away evaluation",
    result_fields=_FLYING_RESULT_FIELDS,
)
def evaluate_flying_away(
    genotype: Genotype, config: FlyingAwayEvaluationConfig | None = None
):
    return _evaluate_flying_away(genotype, config or FlyingAwayEvaluationConfig())
