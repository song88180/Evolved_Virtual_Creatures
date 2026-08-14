"""X-axis flying task."""

from dataclasses import dataclass
from .shared import DEFAULT_FLYING_MIN_TOTAL_VOLUME, FLYING_RESULT_FIELDS, EnvironmentFamily, EvaluationConfig, EvaluationResult, RolloutPolicy, TaskDefinition, excess_volume, failure_flags


TASK_ENVIRONMENT = EnvironmentFamily(
    name="flying_x", gravity=-9.81, fluid_density=1.225,
    fluid_viscosity=1.8e-5, fluid_shape="ellipsoid",
    fluid_coef=(0.5, 0.25, 1.5, 1.0, 1.0),
    body_friction=(1.0, 0.005, 0.0001), creature_contype=2,
    creature_conaffinity=1, self_collision_conaffinity=3,
    floor_contype=1, floor_conaffinity=2, initial_floor_clearance=5.0,
    supports_fluid_overrides=True, supports_scheduled_gravity=True,
)


@dataclass(frozen=True)
class FlyingEvaluationConfig(EvaluationConfig):
    environment: EnvironmentFamily = TASK_ENVIRONMENT
    initial_floor_contact_policy: str = "contact"
    distance_weight: float = 0.1
    energy_weight: float = 1e-7
    angular_speed_weight: float = 0.01
    height_loss_weight: float = 0.5
    ground_touch_weight: float = 0.0
    no_ground_touch_bonus: float = 10
    fitness_gain_fraction: float = 0.5
    min_total_volume: float = DEFAULT_FLYING_MIN_TOTAL_VOLUME


@dataclass(frozen=True)
class FlyingEvaluationResult(EvaluationResult):
    height_loss: float
    first_ground_contact_time: float | None
    ground_touch_penalty: float
    no_ground_touch_bonus: float
    controlled_fitness: float
    passive_fitness: float
    fitness_gain: float
    build_failed: bool = False
    disqualified: bool = False
    failure_reason: str | None = None


def _episode_fitness(config, metrics: dict) -> float:
    return (
        config.distance_weight * metrics["average_forward_speed"]
        + metrics["no_ground_touch_bonus"]
        - config.height_loss_weight * metrics["height_loss"]
        - metrics["ground_touch_penalty"]
        - config.energy_weight * metrics["control_energy"]
        - config.angular_speed_weight * metrics["mean_angular_speed"]
        - config.body_count_weight * metrics["body_count"]
        - config.volume_weight * excess_volume(
            metrics["total_volume"], config.volume_penalty_cutoff
        )
    )


def fitness_callback(config, controlled_metrics: dict, passive_metrics: dict | None):
    assert passive_metrics is not None
    controlled_fitness = _episode_fitness(config, controlled_metrics)
    passive_fitness = _episode_fitness(config, passive_metrics)
    gain = controlled_fitness - passive_fitness
    fitness = (
        gain * config.fitness_gain_fraction
        + controlled_fitness * (1.0 - config.fitness_gain_fraction)
    )
    metrics = dict(
        controlled_metrics,
        controlled_fitness=controlled_fitness,
        passive_fitness=passive_fitness,
        fitness_gain=gain,
    )
    return FlyingEvaluationResult(fitness=fitness, **metrics)


def failed_task_callback(config, reason: str):
    return FlyingEvaluationResult(
        fitness=config.build_failure_fitness, origin_distance=0.0,
        average_origin_speed=0.0, forward_distance=0.0,
        average_forward_speed=0.0, sideways_drift_speed=0.0, height_loss=0.0,
        first_ground_contact_time=None, ground_touch_penalty=0.0,
        no_ground_touch_bonus=0.0, controlled_fitness=0.0, passive_fitness=0.0,
        fitness_gain=0.0, control_energy=0.0, mean_angular_speed=0.0,
        simulated_seconds=0.0, actuator_count=0, body_count=0, total_volume=0.0,
        **failure_flags(reason),
    )


TASK_DEFINITION = TaskDefinition(
    name="flying_x",
    config_type=FlyingEvaluationConfig,
    result_type=FlyingEvaluationResult,
    fitness_callback=fitness_callback,
    failed_task_callback=failed_task_callback,
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
