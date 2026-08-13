"""Distance-from-origin swimming task."""

from dataclasses import dataclass
import math
from typing import Sequence

from .. import evaluation as evaluation_engine
from ..constants import EnvironmentFamily
from ..genotype import Genotype
from .shared import DEFAULT_MIN_BODY_VOLUME, DEFAULT_MIN_TOTAL_VOLUME, SWIMMING_RESULT_FIELDS, TaskDefinition, failed_swimming


@dataclass(frozen=True)
class SwimmingAwayEvaluationConfig:
    episode_seconds: float = 10.0
    max_node: int = 500
    self_collision: bool = False
    disallow_collision: bool = False
    target_direction: Sequence[float] = (1.0, 0.0, 0.0)
    speed_weight: float = 1.0
    energy_weight: float = 1e-6
    angular_speed_weight: float = 0.01
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
class OriginDistanceEvaluationResult:
    fitness: float
    origin_distance: float
    average_origin_speed: float
    forward_distance: float
    average_forward_speed: float
    sideways_drift_speed: float
    vertical_drift_speed: float
    control_energy: float
    mean_angular_speed: float
    simulated_seconds: float
    actuator_count: int
    body_count: int
    total_volume: float
    build_failed: bool = False
    disqualified: bool = False
    failure_reason: str | None = None


def evaluate_origin_distance(genotype: Genotype, config: SwimmingAwayEvaluationConfig | None = None):
    config = config or SwimmingAwayEvaluationConfig()
    built = evaluation_engine._build_model(genotype, config)
    if isinstance(built, str):
        return failed_swimming(config, built, OriginDistanceEvaluationResult)
    model, data, builder = built
    metrics = evaluation_engine._run_controlled_episode(model, data, builder, config)
    if isinstance(metrics, str):
        return failed_swimming(config, metrics, OriginDistanceEvaluationResult)
    fitness = (
        config.speed_weight * metrics["average_origin_speed"]
        - config.energy_weight * metrics["control_energy"]
        - config.angular_speed_weight * metrics["mean_angular_speed"]
        - config.body_count_weight * metrics["body_count"]
        - config.volume_weight * evaluation_engine._excess_volume(
            metrics["total_volume"], config.volume_penalty_cutoff
        )
    )
    if not math.isfinite(fitness):
        return failed_swimming(config, "Simulation produced a non-finite fitness.", OriginDistanceEvaluationResult)
    return OriginDistanceEvaluationResult(fitness=fitness, **metrics)


TASK_DEFINITION = TaskDefinition(
    name="swimming_away",
    config_type=SwimmingAwayEvaluationConfig,
    result_type=OriginDistanceEvaluationResult,
    evaluator=evaluate_origin_distance,
    environment_family=EnvironmentFamily.SWIMMING,
    title="Distance-from-origin swimming_away evaluation",
    result_fields=SWIMMING_RESULT_FIELDS,
    order=20,
)
