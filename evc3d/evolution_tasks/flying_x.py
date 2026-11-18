"""X-axis flying task."""

from dataclasses import dataclass

from .families import (
    FlyingConfig,
    FlyingResult,
    blend_flying_fitness,
    flying_environment,
    flying_episode_fitness,
)
from .shared import FLYING_RESULT_FIELDS, RolloutPolicy, TaskDefinition

TASK_ENVIRONMENT = flying_environment("flying_x")


@dataclass(frozen=True)
class FlyingEvaluationConfig(FlyingConfig):
    forward_speed_weight: float = 0.1


@dataclass(frozen=True)
class FlyingEvaluationResult(FlyingResult):
    """Result of the X-axis flying task."""


def _episode_fitness(config, metrics: dict) -> float:
    speed_term = config.forward_speed_weight * metrics["average_forward_speed"]
    return flying_episode_fitness(config, metrics, speed_term)


def fitness_callback(config, controlled_metrics: dict, passive_metrics: dict | None):
    return blend_flying_fitness(
        config,
        controlled_metrics,
        passive_metrics,
        _episode_fitness,
        FlyingEvaluationResult,
    )


TASK_DEFINITION = TaskDefinition(
    name="flying_x",
    config_type=FlyingEvaluationConfig,
    result_type=FlyingEvaluationResult,
    fitness_callback=fitness_callback,
    rollout_policy=RolloutPolicy(
        passive_baseline=True,
        track_floor_contact=True,
        horizontal_origin_distance=True,
    ),
    environment=TASK_ENVIRONMENT,
    title="X-axis flying_x evaluation",
    result_fields=FLYING_RESULT_FIELDS,
    order=50,
)
