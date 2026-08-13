"""Shared constants and rollout helpers for built-in tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from ..genotype import Genotype


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
    supports_fluid_overrides: bool = False
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


class EvaluationConfig(Protocol):
    """Settings required by shared evaluation and rendering infrastructure."""

    episode_seconds: float
    max_node: int
    self_collision: bool
    disallow_collision: bool
    build_failure_fitness: float
    max_abs_state_value: float
    max_abs_velocity: float
    max_abs_acceleration: float
    max_volume: float
    environment: EnvironmentFamily
    settle_seconds: float
    max_creature_height: float
    min_center_height_fraction: float
    initial_floor_contact_policy: str


@dataclass(frozen=True)
class ResultField:
    """One task-specific result value printed by the evaluation CLI."""

    label: str
    attribute: str
    format_spec: str = ".6f"
    none_text: str | None = None


@dataclass(frozen=True)
class TaskDefinition:
    """Complete public definition exported by one task module."""

    name: str
    config_type: type
    result_type: type
    evaluator: Callable
    environment: EnvironmentFamily
    title: str
    result_fields: tuple[ResultField, ...]
    order: int


TASK_REGISTRY: dict[str, TaskDefinition] = {}
_TASKS_BY_CONFIG_TYPE: dict[type, TaskDefinition] = {}
_BUILTIN_TASKS_LOADED = False


DEFAULT_MIN_BODY_VOLUME = 1e-6
DEFAULT_MIN_TOTAL_VOLUME = 0.0
DEFAULT_FLYING_MIN_TOTAL_VOLUME = 1e-4

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


def evaluate_walking(
    genotype: Genotype, config, *, away: bool, result_type: Callable
):
    from .. import evaluation as evaluation_engine

    built = evaluation_engine._build_model(genotype, config)
    if isinstance(built, str):
        return failed_walking(config, built, result_type)
    model, data, builder = built
    failure = evaluation_engine.initialize_model(model, data, config)
    if failure is not None:
        return failed_walking(config, failure, result_type)
    metrics = evaluation_engine._run_controlled_episode(
        model,
        data,
        builder,
        config,
        root_body_name=f"{genotype.root}_1",
        horizontal_origin_distance=away,
    )
    if isinstance(metrics, str):
        return failed_walking(config, metrics, result_type)
    metrics = {key: value for key, value in metrics.items() if key != "vertical_drift_speed"}
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
        - config.volume_weight
        * evaluation_engine._excess_volume(metrics["total_volume"], config.volume_penalty_cutoff)
    )
    if not math.isfinite(fitness):
        return failed_walking(config, "Simulation produced a non-finite fitness.", result_type)
    return result_type(fitness=fitness, **metrics)


def evaluate_flying(genotype: Genotype, config, speed_metric: str, result_type: Callable):
    from .. import evaluation as evaluation_engine

    built = evaluation_engine._build_model(genotype, config)
    if isinstance(built, str):
        return failed_flying(config, built, result_type)
    model, data, builder = built
    failure = evaluation_engine.initialize_model(model, data, config)
    if failure is not None:
        return failed_flying(config, failure, result_type)
    controlled = evaluation_engine._run_flying_episode(
        model, evaluation_engine._copy_simulation_state(model, data), builder, config,
        apply_controls=True,
    )
    if isinstance(controlled, str):
        return failed_flying(config, controlled, result_type)
    passive = evaluation_engine._run_flying_episode(
        model, evaluation_engine._copy_simulation_state(model, data), builder, config,
        apply_controls=False,
    )
    if isinstance(passive, str):
        return failed_flying(config, passive, result_type)
    controlled_fitness = flying_fitness(config, controlled, speed_metric)
    passive_fitness = flying_fitness(config, passive, speed_metric)
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
        return failed_flying(config, "Simulation produced a non-finite fitness.", result_type)
    return result_type(fitness=fitness, **controlled)


def flying_fitness(config, metrics: dict, speed_metric: str) -> float:
    from .. import evaluation as evaluation_engine

    speed_weight = config.speed_weight if hasattr(config, "speed_weight") else config.distance_weight
    return (
        speed_weight * metrics[speed_metric]
        + metrics["no_ground_touch_bonus"]
        - config.height_loss_weight * metrics["height_loss"]
        - metrics["ground_touch_penalty"]
        - config.energy_weight * metrics["control_energy"]
        - config.angular_speed_weight * metrics["mean_angular_speed"]
        - config.body_count_weight * metrics["body_count"]
        - config.volume_weight
        * evaluation_engine._excess_volume(metrics["total_volume"], config.volume_penalty_cutoff)
    )


def failure_flags(reason: str) -> dict:
    from .. import evaluation as evaluation_engine

    collision = reason == evaluation_engine.DISALLOWED_COLLISION_REASON
    return dict(build_failed=not collision, disqualified=collision, failure_reason=reason)


def failed_swimming(config, reason: str, result_type: Callable):
    return result_type(
        fitness=config.build_failure_fitness, origin_distance=0.0, average_origin_speed=0.0,
        forward_distance=0.0, average_forward_speed=0.0, sideways_drift_speed=0.0,
        vertical_drift_speed=0.0, control_energy=0.0, mean_angular_speed=0.0,
        simulated_seconds=0.0, actuator_count=0, body_count=0, total_volume=0.0,
        **failure_flags(reason),
    )


def failed_walking(config, reason: str, result_type: Callable):
    return result_type(
        fitness=config.build_failure_fitness, origin_distance=0.0, average_origin_speed=0.0,
        forward_distance=0.0, average_forward_speed=0.0, sideways_drift_speed=0.0,
        height_loss=0.0, control_energy=0.0, mean_angular_speed=0.0,
        mean_upright_error=0.0, simulated_seconds=0.0, actuator_count=0,
        body_count=0, total_volume=0.0, **failure_flags(reason),
    )


def failed_flying(config, reason: str, result_type: Callable):
    return result_type(
        fitness=config.build_failure_fitness, origin_distance=0.0, average_origin_speed=0.0,
        forward_distance=0.0, average_forward_speed=0.0, sideways_drift_speed=0.0,
        height_loss=0.0, first_ground_contact_time=None, ground_touch_penalty=0.0,
        no_ground_touch_bonus=0.0, controlled_fitness=0.0, passive_fitness=0.0,
        fitness_gain=0.0, control_energy=0.0, mean_angular_speed=0.0,
        simulated_seconds=0.0, actuator_count=0, body_count=0, total_volume=0.0,
        **failure_flags(reason),
    )
