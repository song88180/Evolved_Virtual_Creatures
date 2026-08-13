"""Distance-from-origin swimming task."""

from dataclasses import dataclass, replace
from typing import Sequence

from .. import evaluation as evaluation_engine
from .shared import DEFAULT_ENVIRONMENT, DEFAULT_MIN_BODY_VOLUME, DEFAULT_MIN_TOTAL_VOLUME, SWIMMING_RESULT_FIELDS, EnvironmentFamily, RolloutPolicy, TaskDefinition, failure_flags


TASK_ENVIRONMENT = replace(DEFAULT_ENVIRONMENT, name="swimming_away")


@dataclass(frozen=True)
class SwimmingAwayEvaluationConfig:
    environment: EnvironmentFamily = TASK_ENVIRONMENT
    settle_seconds: float = 0.0
    max_creature_height: float = 0.0
    min_center_height_fraction: float = 0.0
    initial_floor_contact_policy: str = "allow"
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
    min_body_volume: float = DEFAULT_MIN_BODY_VOLUME
    min_total_volume: float = DEFAULT_MIN_TOTAL_VOLUME
    max_volume: float = 1.0
    build_failure_fitness: float = -1_000.0
    max_abs_state_value: float = 1_000_000.0
    max_abs_velocity: float = 1_000.0
    max_abs_acceleration: float = 100_000.0


@dataclass(frozen=True)
class OriginDistanceEvaluationResult:
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


def fitness_callback(config, metrics: dict, _passive_metrics: dict | None):
    fitness = (
        config.speed_weight * metrics["average_origin_speed"]
        - config.energy_weight * metrics["control_energy"]
        - config.angular_speed_weight * metrics["mean_angular_speed"]
        - config.body_count_weight * metrics["body_count"]
        - config.volume_weight * evaluation_engine._excess_volume(
            metrics["total_volume"], config.volume_penalty_cutoff
        )
    )
    return OriginDistanceEvaluationResult(fitness=fitness, **metrics)


def failed_task_callback(config, reason: str):
    return OriginDistanceEvaluationResult(
        fitness=config.build_failure_fitness, origin_distance=0.0,
        average_origin_speed=0.0, forward_distance=0.0,
        average_forward_speed=0.0, sideways_drift_speed=0.0,
        vertical_drift_speed=0.0, control_energy=0.0, mean_angular_speed=0.0,
        simulated_seconds=0.0, actuator_count=0, body_count=0, total_volume=0.0,
        **failure_flags(reason),
    )


TASK_DEFINITION = TaskDefinition(
    name="swimming_away",
    config_type=SwimmingAwayEvaluationConfig,
    result_type=OriginDistanceEvaluationResult,
    fitness_callback=fitness_callback,
    failed_task_callback=failed_task_callback,
    rollout_policy=RolloutPolicy(),
    environment=TASK_ENVIRONMENT,
    title="Distance-from-origin swimming_away evaluation",
    result_fields=SWIMMING_RESULT_FIELDS,
    order=20,
)
