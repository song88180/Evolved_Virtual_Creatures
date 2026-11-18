"""X-axis swimming task."""

from dataclasses import dataclass

from .config import EnvironmentFamily, EvaluationConfig
from .families import SwimmingResult
from .shared import (
    SWIMMING_RESULT_FIELDS,
    RolloutPolicy,
    TaskDefinition,
    excess_volume,
    make_result,
)

TASK_ENVIRONMENT = EnvironmentFamily(name="swimming_x")


@dataclass(frozen=True)
class SwimmingEvaluationConfig(EvaluationConfig):
    forward_speed_weight: float = 1.0
    energy_weight: float = 1e-6
    sideways_drift_weight: float = 0.1
    vertical_drift_weight: float = 0.1
    angular_speed_weight: float = 0.01


@dataclass(frozen=True)
class SwimmingEvaluationResult(SwimmingResult):
    """Result of the X-axis swimming task."""


def fitness_callback(config, metrics: dict, _passive_metrics: dict | None):
    fitness = (
        config.forward_speed_weight * metrics["average_forward_speed"]
        - config.energy_weight * metrics["control_energy"]
        - config.sideways_drift_weight * metrics["sideways_drift_speed"]
        - config.vertical_drift_weight * metrics["vertical_drift_speed"]
        - config.angular_speed_weight * metrics["mean_angular_speed"]
        - config.body_count_weight * metrics["body_count"]
        - config.volume_weight
        * excess_volume(metrics["total_volume"], config.volume_penalty_cutoff)
    )
    return make_result(SwimmingEvaluationResult, fitness, metrics)


TASK_DEFINITION = TaskDefinition(
    name="swimming_x",
    config_type=SwimmingEvaluationConfig,
    result_type=SwimmingEvaluationResult,
    fitness_callback=fitness_callback,
    rollout_policy=RolloutPolicy(),
    environment=TASK_ENVIRONMENT,
    title="X-axis swimming_x evaluation",
    result_fields=SWIMMING_RESULT_FIELDS,
    order=10,
)
