"""X-axis swimming task."""

from dataclasses import dataclass, replace
from .shared import DEFAULT_ENVIRONMENT, SWIMMING_RESULT_FIELDS, EnvironmentFamily, EvaluationConfig, EvaluationResult, RolloutPolicy, TaskDefinition, excess_volume, failure_flags


TASK_ENVIRONMENT = replace(DEFAULT_ENVIRONMENT, name="swimming_x")


@dataclass(frozen=True)
class SwimmingEvaluationConfig(EvaluationConfig):
    environment: EnvironmentFamily = TASK_ENVIRONMENT
    forward_speed_weight: float = 1.0
    energy_weight: float = 1e-6
    sideways_drift_weight: float = 0.1
    vertical_drift_weight: float = 0.1
    angular_speed_weight: float = 0.01


@dataclass(frozen=True)
class SwimmingEvaluationResult(EvaluationResult):
    vertical_drift_speed: float
    build_failed: bool = False
    disqualified: bool = False
    failure_reason: str | None = None


def fitness_callback(config, metrics: dict, _passive_metrics: dict | None):
    fitness = (
        config.forward_speed_weight * metrics["average_forward_speed"]
        - config.energy_weight * metrics["control_energy"]
        - config.sideways_drift_weight * metrics["sideways_drift_speed"]
        - config.vertical_drift_weight * metrics["vertical_drift_speed"]
        - config.angular_speed_weight * metrics["mean_angular_speed"]
        - config.body_count_weight * metrics["body_count"]
        - config.volume_weight * excess_volume(
            metrics["total_volume"], config.volume_penalty_cutoff
        )
    )
    return SwimmingEvaluationResult(fitness=fitness, **metrics)


def failed_task_callback(config, reason: str):
    return SwimmingEvaluationResult(
        fitness=config.build_failure_fitness, origin_distance=0.0,
        average_origin_speed=0.0, forward_distance=0.0,
        average_forward_speed=0.0, sideways_drift_speed=0.0,
        vertical_drift_speed=0.0, control_energy=0.0, mean_angular_speed=0.0,
        simulated_seconds=0.0, actuator_count=0, body_count=0, total_volume=0.0,
        **failure_flags(reason),
    )


TASK_DEFINITION = TaskDefinition(
    name="swimming_x",
    config_type=SwimmingEvaluationConfig,
    result_type=SwimmingEvaluationResult,
    fitness_callback=fitness_callback,
    failed_task_callback=failed_task_callback,
    rollout_policy=RolloutPolicy(),
    environment=TASK_ENVIRONMENT,
    title="X-axis swimming_x evaluation",
    result_fields=SWIMMING_RESULT_FIELDS,
    order=10,
)
