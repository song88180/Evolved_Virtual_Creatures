"""Distance-from-origin swimming task."""

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

TASK_ENVIRONMENT = EnvironmentFamily(name="swimming_away")


@dataclass(frozen=True)
class SwimmingAwayEvaluationConfig(EvaluationConfig):
    origin_speed_weight: float = 1.0
    energy_weight: float = 1e-6
    angular_speed_weight: float = 0.01


@dataclass(frozen=True)
class OriginDistanceEvaluationResult(SwimmingResult):
    """Result of the distance-from-origin swimming task."""


def fitness_callback(config, metrics: dict, _passive_metrics: dict | None):
    fitness = (
        config.origin_speed_weight * metrics["average_origin_speed"]
        - config.energy_weight * metrics["control_energy"]
        - config.angular_speed_weight * metrics["mean_angular_speed"]
        - config.body_count_weight * metrics["body_count"]
        - config.volume_weight
        * excess_volume(metrics["total_volume"], config.volume_penalty_cutoff)
    )
    return make_result(OriginDistanceEvaluationResult, fitness, metrics)


TASK_DEFINITION = TaskDefinition(
    name="swimming_away",
    config_type=SwimmingAwayEvaluationConfig,
    result_type=OriginDistanceEvaluationResult,
    fitness_callback=fitness_callback,
    rollout_policy=RolloutPolicy(),
    environment=TASK_ENVIRONMENT,
    title="Distance-from-origin swimming_away evaluation",
    result_fields=SWIMMING_RESULT_FIELDS,
    order=20,
)
