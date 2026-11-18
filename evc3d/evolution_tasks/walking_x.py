"""X-axis walking task."""

from dataclasses import dataclass

from .families import WalkingConfig, WalkingResult, walking_environment
from .shared import (
    WALKING_RESULT_FIELDS,
    RolloutPolicy,
    TaskDefinition,
    excess_volume,
    make_result,
)

TASK_ENVIRONMENT = walking_environment("walking_x")


@dataclass(frozen=True)
class WalkingEvaluationConfig(WalkingConfig):
    forward_speed_weight: float = 1.0
    sideways_drift_weight: float = 0.1
    upright_weight: float = 0.0


@dataclass(frozen=True)
class WalkingEvaluationResult(WalkingResult):
    """Result of the X-axis walking task."""


def fitness_callback(config, metrics: dict, _passive_metrics: dict | None):
    fitness = (
        config.forward_speed_weight * metrics["average_forward_speed"]
        - config.sideways_drift_weight * metrics["sideways_drift_speed"]
        - config.upright_weight * metrics["mean_upright_error"]
        - config.energy_weight * metrics["control_energy"]
        - config.angular_speed_weight * metrics["mean_angular_speed"]
        - config.height_loss_weight * metrics["height_loss"]
        - config.body_count_weight * metrics["body_count"]
        - config.volume_weight
        * excess_volume(metrics["total_volume"], config.volume_penalty_cutoff)
    )
    return make_result(WalkingEvaluationResult, fitness, metrics)


TASK_DEFINITION = TaskDefinition(
    name="walking_x",
    config_type=WalkingEvaluationConfig,
    result_type=WalkingEvaluationResult,
    fitness_callback=fitness_callback,
    rollout_policy=RolloutPolicy(track_root_upright=True),
    environment=TASK_ENVIRONMENT,
    title="X-axis walking_x evaluation",
    result_fields=WALKING_RESULT_FIELDS,
    order=30,
)
