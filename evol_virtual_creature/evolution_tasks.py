"""Built-in locomotion task configurations and evaluators."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .constants import EnvironmentFamily
from . import evaluation as evaluation_engine
from .evaluation import ResultField, register_task
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

# Swimming X task


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
    config = config or SwimmingEvaluationConfig()
    built = evaluation_engine._build_model(genotype, config)
    if isinstance(built, str):
        return _failed_swimming(config, built, SwimmingEvaluationResult)
    model, data, builder = built
    metrics = evaluation_engine._run_controlled_episode(model, data, builder, config)
    if isinstance(metrics, str):
        return _failed_swimming(config, metrics, SwimmingEvaluationResult)
    fitness = (
        config.forward_speed_weight * metrics["average_forward_speed"]
        - config.energy_weight * metrics["control_energy"]
        - config.sideways_drift_weight * metrics["sideways_drift_speed"]
        - config.vertical_drift_weight * metrics["vertical_drift_speed"]
        - config.angular_speed_weight * metrics["mean_angular_speed"]
        - config.body_count_weight * metrics["body_count"]
        - config.volume_weight * evaluation_engine._excess_volume(
            metrics["total_volume"], config.volume_penalty_cutoff
        )
    )
    if not math.isfinite(fitness):
        return _failed_swimming(
            config, "Simulation produced a non-finite fitness.", SwimmingEvaluationResult
        )
    return SwimmingEvaluationResult(fitness=fitness, **metrics)

# Swimming away task


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
class OriginDistanceEvaluationResult(SwimmingEvaluationResult):
    """Result metrics for swimming-away evaluation."""


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
    config = config or SwimmingAwayEvaluationConfig()
    built = evaluation_engine._build_model(genotype, config)
    if isinstance(built, str):
        return _failed_swimming(config, built, OriginDistanceEvaluationResult)
    model, data, builder = built
    metrics = evaluation_engine._run_controlled_episode(model, data, builder, config)
    if isinstance(metrics, str):
        return _failed_swimming(config, metrics, OriginDistanceEvaluationResult)
    fitness = (
        config.speed_weight * metrics["average_origin_speed"]
        - config.energy_weight * metrics["control_energy"]
        - config.angular_speed_weight * metrics["mean_angular_speed"]
        - config.body_count_weight * metrics["body_count"]
        - config.volume_weight * evaluation_engine._excess_volume(
            metrics["total_volume"], config.volume_penalty_cutoff
        )
    )
    if not math.isfinite(fitness):
        return _failed_swimming(
            config,
            "Simulation produced a non-finite fitness.",
            OriginDistanceEvaluationResult,
        )
    return OriginDistanceEvaluationResult(fitness=fitness, **metrics)

# Walking X task

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
    config = config or WalkingEvaluationConfig()
    return _evaluate_walking(genotype, config, away=False)

# Walking away task

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
    config = config or WalkingAwayEvaluationConfig()
    return _evaluate_walking(genotype, config, away=True)

# Flying X task

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
    config = config or FlyingEvaluationConfig()
    return _evaluate_flying(genotype, config, "average_forward_speed")

# Flying away task

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
    config = config or FlyingAwayEvaluationConfig()
    return _evaluate_flying(genotype, config, "average_origin_speed")

# Shared task-family helpers


def _evaluate_walking(genotype: Genotype, config, *, away: bool):
    built = evaluation_engine._build_model(genotype, config)
    if isinstance(built, str):
        return _failed_walking(config, built)
    model, data, builder = built
    for initialize in (
        evaluation_engine.initialize_walking_model,
        evaluation_engine._walking_height_failure_reason,
        evaluation_engine.settle_walking_model,
    ):
        failure = (
            initialize(model, data)
            if initialize is evaluation_engine.initialize_walking_model
            else initialize(model, data, config)
        )
        if failure is not None:
            return _failed_walking(config, failure)
    data.time = 0.0
    metrics = evaluation_engine._run_controlled_episode(
        model,
        data,
        builder,
        config,
        root_body_name=f"{genotype.root}_1",
        horizontal_origin_distance=away,
    )
    if isinstance(metrics, str):
        return _failed_walking(config, metrics)
    metrics = {
        key: value
        for key, value in metrics.items()
        if key != "vertical_drift_speed"
    }
    progress = (
        config.speed_weight * metrics["average_origin_speed"]
        if away
        else config.forward_speed_weight * metrics["average_forward_speed"]
        - config.sideways_drift_weight * metrics["sideways_drift_speed"]
        - config.upright_weight * metrics["mean_upright_error"]
    )
    fitness = (
        progress
        - config.energy_weight * metrics["control_energy"]
        - config.angular_speed_weight * metrics["mean_angular_speed"]
        - config.height_loss_weight * metrics["height_loss"]
        - config.body_count_weight * metrics["body_count"]
        - config.volume_weight * evaluation_engine._excess_volume(
            metrics["total_volume"], config.volume_penalty_cutoff
        )
    )
    if not math.isfinite(fitness):
        return _failed_walking(config, "Simulation produced a non-finite fitness.")
    return WalkingEvaluationResult(fitness=fitness, **metrics)


def _evaluate_flying(genotype: Genotype, config, speed_metric: str):
    built = evaluation_engine._build_model(genotype, config)
    if isinstance(built, str):
        return _failed_flying(config, built)
    model, data, builder = built
    failure = evaluation_engine.initialize_flying_model(model, data)
    if failure is not None:
        return _failed_flying(config, failure)
    controlled = evaluation_engine._run_flying_episode(
        model,
        evaluation_engine._copy_simulation_state(model, data),
        builder,
        config,
        apply_controls=True,
    )
    if isinstance(controlled, str):
        return _failed_flying(config, controlled)
    passive = evaluation_engine._run_flying_episode(
        model,
        evaluation_engine._copy_simulation_state(model, data),
        builder,
        config,
        apply_controls=False,
    )
    if isinstance(passive, str):
        return _failed_flying(config, passive)
    controlled_fitness = _flying_fitness(config, controlled, speed_metric)
    passive_fitness = _flying_fitness(config, passive, speed_metric)
    gain = controlled_fitness - passive_fitness
    fitness = gain * config.fitness_gain_fraction + controlled_fitness * (
        1.0 - config.fitness_gain_fraction
    )
    controlled.update(
        controlled_fitness=controlled_fitness,
        passive_fitness=passive_fitness,
        fitness_gain=gain,
    )
    if not math.isfinite(fitness):
        return _failed_flying(config, "Simulation produced a non-finite fitness.")
    return FlyingEvaluationResult(fitness=fitness, **controlled)


def _flying_fitness(config, metrics: dict, speed_metric: str) -> float:
    speed_weight = (
        config.speed_weight
        if hasattr(config, "speed_weight")
        else config.distance_weight
    )
    return (
        speed_weight * metrics[speed_metric]
        + metrics["no_ground_touch_bonus"]
        - config.height_loss_weight * metrics["height_loss"]
        - metrics["ground_touch_penalty"]
        - config.energy_weight * metrics["control_energy"]
        - config.angular_speed_weight * metrics["mean_angular_speed"]
        - config.body_count_weight * metrics["body_count"]
        - config.volume_weight
        * evaluation_engine._excess_volume(
            metrics["total_volume"], config.volume_penalty_cutoff
        )
    )


def _failure_flags(reason: str) -> dict:
    collision = reason == evaluation_engine.DISALLOWED_COLLISION_REASON
    return dict(build_failed=not collision, disqualified=collision, failure_reason=reason)


def _failed_swimming(config, reason: str, result_type):
    return result_type(
        fitness=config.build_failure_fitness, origin_distance=0.0, average_origin_speed=0.0,
        forward_distance=0.0, average_forward_speed=0.0, sideways_drift_speed=0.0,
        vertical_drift_speed=0.0, control_energy=0.0, mean_angular_speed=0.0,
        simulated_seconds=0.0, actuator_count=0, body_count=0, total_volume=0.0,
        **_failure_flags(reason),
    )


def _failed_walking(config, reason: str):
    return WalkingEvaluationResult(
        fitness=config.build_failure_fitness, origin_distance=0.0, average_origin_speed=0.0,
        forward_distance=0.0, average_forward_speed=0.0, sideways_drift_speed=0.0,
        height_loss=0.0, control_energy=0.0, mean_angular_speed=0.0,
        mean_upright_error=0.0, simulated_seconds=0.0, actuator_count=0,
        body_count=0, total_volume=0.0, **_failure_flags(reason),
    )


def _failed_flying(config, reason: str):
    return FlyingEvaluationResult(
        fitness=config.build_failure_fitness, origin_distance=0.0, average_origin_speed=0.0,
        forward_distance=0.0, average_forward_speed=0.0, sideways_drift_speed=0.0,
        height_loss=0.0, first_ground_contact_time=None, ground_touch_penalty=0.0,
        no_ground_touch_bonus=0.0, controlled_fitness=0.0, passive_fitness=0.0,
        fitness_gain=0.0, control_energy=0.0, mean_angular_speed=0.0,
        simulated_seconds=0.0, actuator_count=0, body_count=0, total_volume=0.0,
        **_failure_flags(reason),
    )
