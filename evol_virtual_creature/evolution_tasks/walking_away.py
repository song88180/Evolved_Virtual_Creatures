"""Distance-from-origin walking task."""

from dataclasses import dataclass, replace
from typing import Sequence

from .. import evaluation as evaluation_engine
from .shared import DEFAULT_ENVIRONMENT, DEFAULT_MIN_BODY_VOLUME, DEFAULT_MIN_TOTAL_VOLUME, WALKING_RESULT_FIELDS, EnvironmentFamily, RolloutPolicy, TaskDefinition, failure_flags


TASK_ENVIRONMENT = replace(
    DEFAULT_ENVIRONMENT, name="walking_away", gravity=-9.81,
    fluid_density=None, fluid_viscosity=None, body_friction=(1.0, 0.005, 0.0001),
    creature_contype=2, creature_conaffinity=1, self_collision_conaffinity=3,
    floor_contype=1, floor_conaffinity=2, initial_floor_clearance=0.05,
)


@dataclass(frozen=True)
class WalkingAwayEvaluationConfig:
    environment: EnvironmentFamily = TASK_ENVIRONMENT
    initial_floor_contact_policy: str = "penetration"
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
    min_body_volume: float = DEFAULT_MIN_BODY_VOLUME
    min_total_volume: float = DEFAULT_MIN_TOTAL_VOLUME
    max_volume: float = 1.0
    build_failure_fitness: float = -1_000.0
    max_abs_state_value: float = 1_000_000.0
    max_abs_velocity: float = 1_000.0
    max_abs_acceleration: float = 100_000.0


@dataclass(frozen=True)
class WalkingAwayEvaluationResult:
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


def fitness_callback(config, metrics: dict, _passive_metrics: dict | None):
    metrics = {key: value for key, value in metrics.items() if key != "vertical_drift_speed"}
    fitness = (
        config.speed_weight * metrics["average_origin_speed"]
        - config.energy_weight * metrics["control_energy"]
        - config.angular_speed_weight * metrics["mean_angular_speed"]
        - config.height_loss_weight * metrics["height_loss"]
        - config.body_count_weight * metrics["body_count"]
        - config.volume_weight * evaluation_engine._excess_volume(
            metrics["total_volume"], config.volume_penalty_cutoff
        )
    )
    return WalkingAwayEvaluationResult(fitness=fitness, **metrics)


def failed_task_callback(config, reason: str):
    return WalkingAwayEvaluationResult(
        fitness=config.build_failure_fitness, origin_distance=0.0,
        average_origin_speed=0.0, forward_distance=0.0,
        average_forward_speed=0.0, sideways_drift_speed=0.0, height_loss=0.0,
        control_energy=0.0, mean_angular_speed=0.0, mean_upright_error=0.0,
        simulated_seconds=0.0, actuator_count=0, body_count=0, total_volume=0.0,
        **failure_flags(reason),
    )


TASK_DEFINITION = TaskDefinition(
    name="walking_away",
    config_type=WalkingAwayEvaluationConfig,
    result_type=WalkingAwayEvaluationResult,
    fitness_callback=fitness_callback,
    failed_task_callback=failed_task_callback,
    rollout_policy=RolloutPolicy(
        track_root_upright=True, horizontal_origin_distance=True
    ),
    environment=TASK_ENVIRONMENT,
    title="Distance-from-origin walking_away evaluation",
    result_fields=WALKING_RESULT_FIELDS,
    order=40,
)
