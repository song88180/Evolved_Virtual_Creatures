"""X-axis walking task."""

from dataclasses import dataclass, replace
from .. import evaluation as evaluation_engine
from .shared import DEFAULT_ENVIRONMENT, WALKING_RESULT_FIELDS, EnvironmentFamily, EvaluationConfig, EvaluationResult, RolloutPolicy, TaskDefinition, failure_flags


TASK_ENVIRONMENT = replace(
    DEFAULT_ENVIRONMENT, name="walking_x", gravity=-9.81,
    fluid_density=None, fluid_viscosity=None, body_friction=(1.0, 0.005, 0.0001),
    creature_contype=2, creature_conaffinity=1, self_collision_conaffinity=3,
    floor_contype=1, floor_conaffinity=2, initial_floor_clearance=0.05,
)


@dataclass(frozen=True)
class WalkingEvaluationConfig(EvaluationConfig):
    environment: EnvironmentFamily = TASK_ENVIRONMENT
    initial_floor_contact_policy: str = "penetration"
    settle_seconds: float = 1.0
    forward_speed_weight: float = 1.0
    energy_weight: float = 1e-7
    sideways_drift_weight: float = 0.1
    angular_speed_weight: float = 0.01
    upright_weight: float = 0.0
    height_loss_weight: float = 0.2
    min_center_height_fraction: float = 0.5
    max_creature_height: float = 10.0


@dataclass(frozen=True)
class WalkingEvaluationResult(EvaluationResult):
    height_loss: float
    mean_upright_error: float
    build_failed: bool = False
    disqualified: bool = False
    failure_reason: str | None = None


def fitness_callback(config, metrics: dict, _passive_metrics: dict | None):
    metrics = {key: value for key, value in metrics.items() if key != "vertical_drift_speed"}
    fitness = (
        config.forward_speed_weight * metrics["average_forward_speed"]
        - config.sideways_drift_weight * metrics["sideways_drift_speed"]
        - config.upright_weight * metrics["mean_upright_error"]
        - config.energy_weight * metrics["control_energy"]
        - config.angular_speed_weight * metrics["mean_angular_speed"]
        - config.height_loss_weight * metrics["height_loss"]
        - config.body_count_weight * metrics["body_count"]
        - config.volume_weight * evaluation_engine._excess_volume(
            metrics["total_volume"], config.volume_penalty_cutoff
        )
    )
    return WalkingEvaluationResult(fitness=fitness, **metrics)


def failed_task_callback(config, reason: str):
    return WalkingEvaluationResult(
        fitness=config.build_failure_fitness, origin_distance=0.0,
        average_origin_speed=0.0, forward_distance=0.0,
        average_forward_speed=0.0, sideways_drift_speed=0.0, height_loss=0.0,
        control_energy=0.0, mean_angular_speed=0.0, mean_upright_error=0.0,
        simulated_seconds=0.0, actuator_count=0, body_count=0, total_volume=0.0,
        **failure_flags(reason),
    )


TASK_DEFINITION = TaskDefinition(
    name="walking_x",
    config_type=WalkingEvaluationConfig,
    result_type=WalkingEvaluationResult,
    fitness_callback=fitness_callback,
    failed_task_callback=failed_task_callback,
    rollout_policy=RolloutPolicy(track_root_upright=True),
    environment=TASK_ENVIRONMENT,
    title="X-axis walking_x evaluation",
    result_fields=WALKING_RESULT_FIELDS,
    order=30,
)
