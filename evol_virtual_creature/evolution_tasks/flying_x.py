"""X-axis flying task."""

from dataclasses import dataclass, replace
from typing import Sequence

from ..genotype import Genotype
from .shared import DEFAULT_ENVIRONMENT, DEFAULT_FLYING_MIN_TOTAL_VOLUME, DEFAULT_MIN_BODY_VOLUME, FLYING_RESULT_FIELDS, EnvironmentFamily, TaskDefinition, evaluate_flying


TASK_ENVIRONMENT = replace(
    DEFAULT_ENVIRONMENT, name="flying_x", gravity=-9.81, fluid_density=1.225,
    fluid_viscosity=1.8e-5, fluid_shape="ellipsoid",
    fluid_coef=(0.5, 0.25, 1.5, 1.0, 1.0),
    body_friction=(1.0, 0.005, 0.0001), creature_contype=2,
    creature_conaffinity=1, self_collision_conaffinity=3,
    floor_contype=1, floor_conaffinity=2, initial_floor_clearance=5.0,
    supports_fluid_overrides=True, supports_scheduled_gravity=True,
)


@dataclass(frozen=True)
class FlyingEvaluationConfig:
    environment: EnvironmentFamily = TASK_ENVIRONMENT
    settle_seconds: float = 0.0
    max_creature_height: float = 0.0
    min_center_height_fraction: float = 0.0
    initial_floor_contact_policy: str = "contact"
    episode_seconds: float = 10.0
    max_node: int = 500
    self_collision: bool = False
    disallow_collision: bool = False
    target_direction: Sequence[float] = (1.0, 0.0, 0.0)
    distance_weight: float = 0.1
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
class FlyingEvaluationResult:
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


def evaluate_x_axis_flying(genotype: Genotype, config: FlyingEvaluationConfig | None = None):
    config = config or FlyingEvaluationConfig()
    return evaluate_flying(genotype, config, "average_forward_speed", FlyingEvaluationResult)


TASK_DEFINITION = TaskDefinition(
    name="flying_x",
    config_type=FlyingEvaluationConfig,
    result_type=FlyingEvaluationResult,
    evaluator=evaluate_x_axis_flying,
    environment=TASK_ENVIRONMENT,
    title="X-axis flying_x evaluation",
    result_fields=FLYING_RESULT_FIELDS,
    order=50,
)
