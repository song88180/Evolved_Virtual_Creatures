"""Distance-from-origin swimming task."""

from dataclasses import dataclass
from .shared import SWIMMING_RESULT_FIELDS, EnvironmentFamily, EvaluationConfig, EvaluationResult, RolloutPolicy, TaskDefinition, excess_volume, failure_flags


TASK_ENVIRONMENT = EnvironmentFamily(name="swimming_away")


@dataclass(frozen=True)
class SwimmingAwayEvaluationConfig(EvaluationConfig):
    environment: EnvironmentFamily = TASK_ENVIRONMENT
    speed_weight: float = 1.0
    energy_weight: float = 1e-6
    angular_speed_weight: float = 0.01


@dataclass(frozen=True)
class OriginDistanceEvaluationResult(EvaluationResult):
    vertical_drift_speed: float
    build_failed: bool = False
    disqualified: bool = False
    failure_reason: str | None = None


def fitness_callback(config, metrics: dict, _passive_metrics: dict | None):
    fitness = (
        config.speed_weight * metrics["average_origin_speed"]
        - config.energy_weight * metrics["control_energy"]
        - config.angular_speed_weight * metrics["mean_angular_speed"]
        - config.body_count_weight * metrics["body_count"]
        - config.volume_weight * excess_volume(
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
