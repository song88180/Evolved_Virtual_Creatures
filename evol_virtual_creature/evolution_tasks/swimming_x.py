"""X-axis swimming task."""

from dataclasses import dataclass, replace
import math
from typing import Sequence

from .. import evaluation as evaluation_engine
from ..genotype import Genotype
from .shared import DEFAULT_ENVIRONMENT, DEFAULT_MIN_BODY_VOLUME, DEFAULT_MIN_TOTAL_VOLUME, SWIMMING_RESULT_FIELDS, EnvironmentFamily, TaskDefinition, failed_swimming


TASK_ENVIRONMENT = replace(DEFAULT_ENVIRONMENT, name="swimming_x")


@dataclass(frozen=True)
class SwimmingEvaluationConfig:
    environment: EnvironmentFamily = TASK_ENVIRONMENT
    settle_seconds: float = 0.0
    max_creature_height: float = 0.0
    min_center_height_fraction: float = 0.0
    initial_floor_contact_policy: str = "allow"
    episode_seconds: float = 10.0
    max_node: int = 500
    self_collision: bool = False
    disallow_collision: bool = False
    target_direction: Sequence[float] = (1.0, 0.0, 0.0)
    forward_speed_weight: float = 1.0
    energy_weight: float = 1e-6
    sideways_drift_weight: float = 0.1
    vertical_drift_weight: float = 0.1
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
class SwimmingEvaluationResult:
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


def evaluate_x_axis_swimming(genotype: Genotype, config: SwimmingEvaluationConfig | None = None):
    config = config or SwimmingEvaluationConfig()
    built = evaluation_engine._build_model(genotype, config)
    if isinstance(built, str):
        return failed_swimming(config, built, SwimmingEvaluationResult)
    model, data, builder = built
    metrics = evaluation_engine._run_controlled_episode(model, data, builder, config)
    if isinstance(metrics, str):
        return failed_swimming(config, metrics, SwimmingEvaluationResult)
    fitness = (
        config.forward_speed_weight * metrics["average_forward_speed"]
        - config.energy_weight * metrics["control_energy"]
        - config.sideways_drift_weight * metrics["sideways_drift_speed"]
        - config.vertical_drift_weight * metrics["vertical_drift_speed"]
        - config.angular_speed_weight * metrics["mean_angular_speed"]
        - config.body_count_weight * metrics["body_count"]
        - config.volume_weight * evaluation_engine._excess_volume(
            metrics["total_volume"], config.volume_penalty_cutoff
        )
    )
    if not math.isfinite(fitness):
        return failed_swimming(config, "Simulation produced a non-finite fitness.", SwimmingEvaluationResult)
    return SwimmingEvaluationResult(fitness=fitness, **metrics)


TASK_DEFINITION = TaskDefinition(
    name="swimming_x",
    config_type=SwimmingEvaluationConfig,
    result_type=SwimmingEvaluationResult,
    evaluator=evaluate_x_axis_swimming,
    environment=TASK_ENVIRONMENT,
    title="X-axis swimming_x evaluation",
    result_fields=SWIMMING_RESULT_FIELDS,
    order=10,
)
