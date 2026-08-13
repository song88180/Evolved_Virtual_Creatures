"""Distance-from-origin walking task."""

from dataclasses import dataclass
from typing import Sequence

from ..constants import EnvironmentFamily
from ..evaluation import TaskDefinition
from ..genotype import Genotype
from .shared import DEFAULT_MIN_BODY_VOLUME, DEFAULT_MIN_TOTAL_VOLUME, WALKING_RESULT_FIELDS, evaluate_walking


@dataclass(frozen=True)
class WalkingAwayEvaluationConfig:
    episode_seconds: float = 10.0
    settle_seconds: float = 1.0
    max_node: int = 500
    self_collision: bool = False
    disallow_collision: bool = False
    target_direction: Sequence[float] = (1.0, 0.0, 0.0)
    speed_weight: float = 1.0
    energy_weight: float = 1e-7
    angular_speed_weight: float = 0.01
    height_loss_weight: float = 0.2
    min_center_height_fraction: float = 0.5
    max_creature_height: float = 10.0
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
class WalkingAwayEvaluationResult:
    fitness: float
    origin_distance: float
    average_origin_speed: float
    forward_distance: float
    average_forward_speed: float
    sideways_drift_speed: float
    height_loss: float
    control_energy: float
    mean_angular_speed: float
    mean_upright_error: float
    simulated_seconds: float
    actuator_count: int
    body_count: int
    total_volume: float
    build_failed: bool = False
    disqualified: bool = False
    failure_reason: str | None = None


def evaluate_walking_away(genotype: Genotype, config: WalkingAwayEvaluationConfig | None = None):
    config = config or WalkingAwayEvaluationConfig()
    return evaluate_walking(genotype, config, away=True, result_type=WalkingAwayEvaluationResult)


TASK_DEFINITION = TaskDefinition(
    name="walking_away",
    config_type=WalkingAwayEvaluationConfig,
    result_type=WalkingAwayEvaluationResult,
    evaluator=evaluate_walking_away,
    environment_family=EnvironmentFamily.WALKING,
    title="Distance-from-origin walking_away evaluation",
    result_fields=WALKING_RESULT_FIELDS,
    order=40,
)
