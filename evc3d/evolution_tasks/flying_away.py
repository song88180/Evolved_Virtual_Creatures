"""Distance-from-origin flying task."""

from dataclasses import dataclass

from .families import (
    FlyingConfig,
    FlyingResult,
    blend_flying_fitness,
    flying_environment,
    flying_episode_fitness,
)
from .shared import FLYING_RESULT_FIELDS, RolloutPolicy, TaskDefinition

TASK_ENVIRONMENT = flying_environment("flying_away")


@dataclass(frozen=True)
class FlyingAwayEvaluationConfig(FlyingConfig):
    origin_speed_weight: float = 0.1


@dataclass(frozen=True)
class FlyingAwayEvaluationResult(FlyingResult):
    """Result of the distance-from-origin flying task."""


def _episode_fitness(config, metrics: dict) -> float:
    speed_term = config.origin_speed_weight * metrics["average_origin_speed"]
    return flying_episode_fitness(config, metrics, speed_term)


def fitness_callback(config, controlled_metrics: dict, passive_metrics: dict | None):
    return blend_flying_fitness(
        config,
        controlled_metrics,
        passive_metrics,
        _episode_fitness,
        FlyingAwayEvaluationResult,
    )


TASK_DEFINITION = TaskDefinition(
    name="flying_away",
    config_type=FlyingAwayEvaluationConfig,
    result_type=FlyingAwayEvaluationResult,
    fitness_callback=fitness_callback,
    rollout_policy=RolloutPolicy(
        passive_baseline=True,
        track_floor_contact=True,
        horizontal_origin_distance=True,
    ),
    environment=TASK_ENVIRONMENT,
    title="Distance-from-origin flying_away evaluation",
    result_fields=FLYING_RESULT_FIELDS,
    order=60,
)
