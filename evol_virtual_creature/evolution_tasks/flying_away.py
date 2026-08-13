"""Distance-from-origin flying task."""

from dataclasses import dataclass
from typing import Sequence

from ..constants import EnvironmentFamily
from ..evaluation import register_task
from ..genotype import Genotype
from ..phenotype import DEFAULT_FLYING_FLUID_COEF, DEFAULT_FLYING_FLUID_DENSITY, DEFAULT_FLYING_FLUID_SHAPE, DEFAULT_FLYING_FLUID_VISCOSITY, DEFAULT_FLYING_GRAVITY
from .shared import DEFAULT_FLYING_MIN_TOTAL_VOLUME, DEFAULT_MIN_BODY_VOLUME, FLYING_RESULT_FIELDS, evaluate_flying


@dataclass(frozen=True)
class FlyingAwayEvaluationConfig:
    episode_seconds: float = 10.0
    fluid_density: float = DEFAULT_FLYING_FLUID_DENSITY
    fluid_viscosity: float = DEFAULT_FLYING_FLUID_VISCOSITY
    fluid_shape: str = DEFAULT_FLYING_FLUID_SHAPE
    fluid_coef: Sequence[float] = DEFAULT_FLYING_FLUID_COEF
    gravity: float = DEFAULT_FLYING_GRAVITY
    max_node: int = 500
    self_collision: bool = False
    disallow_collision: bool = False
    target_direction: Sequence[float] = (1.0, 0.0, 0.0)
    speed_weight: float = 0.1
    energy_weight: float = 1e-7
    angular_speed_weight: float = 0.01
    height_loss_weight: float = 0.5
    ground_touch_weight: float = 0.0
    no_ground_touch_bonus: float = 10
    fitness_gain_fraction: float = 0.5
    body_count_weight: float = 0.001
    volume_weight: float = 0.01
    volume_penalty_cutoff: float = 0.1
    min_body_volume: float = DEFAULT_MIN_BODY_VOLUME
    min_total_volume: float = DEFAULT_FLYING_MIN_TOTAL_VOLUME
    max_volume: float = 1.0
    build_failure_fitness: float = -1_000.0
    max_abs_state_value: float = 1_000_000.0
    max_abs_velocity: float = 1_000.0
    max_abs_acceleration: float = 100_000.0


@dataclass(frozen=True)
class FlyingAwayEvaluationResult:
    fitness: float
    origin_distance: float
    average_origin_speed: float
    forward_distance: float
    average_forward_speed: float
    sideways_drift_speed: float
    height_loss: float
    first_ground_contact_time: float | None
    ground_touch_penalty: float
    no_ground_touch_bonus: float
    controlled_fitness: float
    passive_fitness: float
    fitness_gain: float
    control_energy: float
    mean_angular_speed: float
    simulated_seconds: float
    actuator_count: int
    body_count: int
    total_volume: float
    build_failed: bool = False
    disqualified: bool = False
    failure_reason: str | None = None


@register_task("flying_away", config_type=FlyingAwayEvaluationConfig,
               environment_family=EnvironmentFamily.FLYING,
               title="Distance-from-origin flying_away evaluation",
               result_fields=FLYING_RESULT_FIELDS, order=60)
def evaluate_flying_away(genotype: Genotype, config: FlyingAwayEvaluationConfig | None = None):
    config = config or FlyingAwayEvaluationConfig()
    return evaluate_flying(genotype, config, "average_origin_speed", FlyingAwayEvaluationResult)
