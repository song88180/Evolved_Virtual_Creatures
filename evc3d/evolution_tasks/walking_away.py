"""Distance-from-origin walking task."""

from dataclasses import dataclass

from .families import WalkingConfig, WalkingResult, walking_environment
from .shared import (
    WALKING_RESULT_FIELDS,
    RolloutPolicy,
    TaskDefinition,
    excess_volume,
    make_result,
)

TASK_ENVIRONMENT = walking_environment("walking_away")


@dataclass(frozen=True)
class WalkingAwayEvaluationConfig(WalkingConfig):
    origin_speed_weight: float = 1.0


@dataclass(frozen=True)
class WalkingAwayEvaluationResult(WalkingResult):
    """Result of the distance-from-origin walking task."""


def fitness_callback(config, metrics: dict, _passive_metrics: dict | None):
    fitness = (
        config.origin_speed_weight * metrics["average_origin_speed"]
        - config.energy_weight * metrics["control_energy"]
        - config.angular_speed_weight * metrics["mean_angular_speed"]
        - config.height_loss_weight * metrics["height_loss"]
        - config.body_count_weight * metrics["body_count"]
        - config.volume_weight
        * excess_volume(metrics["total_volume"], config.volume_penalty_cutoff)
    )
    return make_result(WalkingAwayEvaluationResult, fitness, metrics)


TASK_DEFINITION = TaskDefinition(
    name="walking_away",
    config_type=WalkingAwayEvaluationConfig,
    result_type=WalkingAwayEvaluationResult,
    fitness_callback=fitness_callback,
    rollout_policy=RolloutPolicy(
        track_root_upright=True, horizontal_origin_distance=True
    ),
    environment=TASK_ENVIRONMENT,
    title="Distance-from-origin walking_away evaluation",
    result_fields=WALKING_RESULT_FIELDS,
    order=40,
)
