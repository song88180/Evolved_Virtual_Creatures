from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence, TypeAlias

import mujoco
import numpy as np

from .genotype import Genotype
from .graph_analysis import PhenotypeBuildAbort
from .phenotype import (
    DEFAULT_FLYING_FLUID_COEF,
    DEFAULT_FLYING_FLUID_DENSITY,
    DEFAULT_FLYING_FLUID_SHAPE,
    DEFAULT_FLYING_FLUID_VISCOSITY,
    DEFAULT_FLYING_GRAVITY,
    ActuatorController,
    PhenotypeBuilder,
)


DISALLOWED_COLLISION_REASON = "Disallowed non-parent self-collision detected."
NUMERICAL_INSTABILITY_REASON = "Simulation became numerically unstable."
INITIAL_FLOOR_OVERLAP_REASON = "Creature overlaps the floor at initialization."
MINIMUM_BODY_VOLUME_REASON = "Creature body volume is below the minimum allowed volume."
MINIMUM_TOTAL_VOLUME_REASON = (
    "Creature total volume is below the minimum allowed volume."
)
WALKING_CENTER_HEIGHT_DROP_REASON = (
    "Creature center of mass dropped below the walking height threshold."
)
MAXIMUM_CREATURE_HEIGHT_REASON = "Creature height exceeds the maximum allowed height."
_WALKING_FLOOR_CLEARANCE = 0.05
_FLYING_FLOOR_CLEARANCE = 5.0
_DEFAULT_MIN_BODY_VOLUME = 1e-6
_DEFAULT_MIN_TOTAL_VOLUME = 0.0
_DEFAULT_FLYING_MIN_TOTAL_VOLUME = 1e-4


@dataclass(frozen=True)
class SwimmingEvaluationConfig:
    """Weights and simulation settings for x-axis swimming fitness."""

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
    min_body_volume: float = _DEFAULT_MIN_BODY_VOLUME
    min_total_volume: float = _DEFAULT_MIN_TOTAL_VOLUME
    max_volume: float = 1.0
    build_failure_fitness: float = -1_000.0
    max_abs_state_value: float = 1_000_000.0
    max_abs_velocity: float = 1_000.0
    max_abs_acceleration: float = 100_000.0


@dataclass(frozen=True)
class WalkingEvaluationConfig:
    """Weights and simulation settings for x-axis walking fitness."""

    episode_seconds: float = 10.0
    settle_seconds: float = 1.0
    max_node: int = 500
    self_collision: bool = False
    disallow_collision: bool = False
    target_direction: Sequence[float] = (1.0, 0.0, 0.0)
    forward_speed_weight: float = 1.0
    energy_weight: float = 1e-7
    sideways_drift_weight: float = 0.1
    angular_speed_weight: float = 0.01
    upright_weight: float = 0.2
    height_loss_weight: float = 0.2
    min_center_height_fraction: float = 0.5
    max_creature_height: float = 10.0
    body_count_weight: float = 0.001
    volume_weight: float = 0.01
    volume_penalty_cutoff: float = 0.1
    min_body_volume: float = _DEFAULT_MIN_BODY_VOLUME
    min_total_volume: float = _DEFAULT_MIN_TOTAL_VOLUME
    max_volume: float = 1.0
    build_failure_fitness: float = -1_000.0
    max_abs_state_value: float = 1_000_000.0
    max_abs_velocity: float = 1_000.0
    max_abs_acceleration: float = 100_000.0


@dataclass(frozen=True)
class SwimmingAwayEvaluationConfig:
    """Weights and simulation settings for swimming-away fitness."""

    episode_seconds: float = 10.0
    max_node: int = 500
    self_collision: bool = False
    disallow_collision: bool = False
    target_direction: Sequence[float] = (1.0, 0.0, 0.0)
    distance_weight: float = 1.0
    energy_weight: float = 1e-6
    angular_speed_weight: float = 0.01
    body_count_weight: float = 0.001
    volume_weight: float = 0.01
    volume_penalty_cutoff: float = 0.1
    min_body_volume: float = _DEFAULT_MIN_BODY_VOLUME
    min_total_volume: float = _DEFAULT_MIN_TOTAL_VOLUME
    max_volume: float = 1.0
    build_failure_fitness: float = -1_000.0
    max_abs_state_value: float = 1_000_000.0
    max_abs_velocity: float = 1_000.0
    max_abs_acceleration: float = 100_000.0


@dataclass(frozen=True)
class WalkingAwayEvaluationConfig:
    """Weights and simulation settings for walking-away fitness."""

    episode_seconds: float = 10.0
    settle_seconds: float = 1.0
    max_node: int = 500
    self_collision: bool = False
    disallow_collision: bool = False
    target_direction: Sequence[float] = (1.0, 0.0, 0.0)
    distance_weight: float = 1.0
    energy_weight: float = 1e-7
    angular_speed_weight: float = 0.01
    height_loss_weight: float = 0.2
    min_center_height_fraction: float = 0.5
    max_creature_height: float = 10.0
    body_count_weight: float = 0.001
    volume_weight: float = 0.01
    volume_penalty_cutoff: float = 0.1
    min_body_volume: float = _DEFAULT_MIN_BODY_VOLUME
    min_total_volume: float = _DEFAULT_MIN_TOTAL_VOLUME
    max_volume: float = 1.0
    build_failure_fitness: float = -1_000.0
    max_abs_state_value: float = 1_000_000.0
    max_abs_velocity: float = 1_000.0
    max_abs_acceleration: float = 100_000.0


@dataclass(frozen=True)
class FlyingEvaluationConfig:
    """Weights and simulation settings for x-axis flying fitness."""

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
    min_body_volume: float = _DEFAULT_MIN_BODY_VOLUME
    min_total_volume: float = _DEFAULT_FLYING_MIN_TOTAL_VOLUME
    max_volume: float = 1.0
    build_failure_fitness: float = -1_000.0
    max_abs_state_value: float = 1_000_000.0
    max_abs_velocity: float = 1_000.0
    max_abs_acceleration: float = 100_000.0


@dataclass(frozen=True)
class FlyingAwayEvaluationConfig:
    """Weights and simulation settings for flying-away fitness."""

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
    min_body_volume: float = _DEFAULT_MIN_BODY_VOLUME
    min_total_volume: float = _DEFAULT_FLYING_MIN_TOTAL_VOLUME
    max_volume: float = 1.0
    build_failure_fitness: float = -1_000.0
    max_abs_state_value: float = 1_000_000.0
    max_abs_velocity: float = 1_000.0
    max_abs_acceleration: float = 100_000.0


EvaluationConfig: TypeAlias = (
    SwimmingEvaluationConfig
    | WalkingEvaluationConfig
    | SwimmingAwayEvaluationConfig
    | WalkingAwayEvaluationConfig
    | FlyingEvaluationConfig
    | FlyingAwayEvaluationConfig
)


@dataclass(frozen=True)
class SwimmingEvaluationResult:
    fitness: float
    origin_distance: float
    average_origin_speed: float
    forward_distance: float
    average_forward_speed: float
    sideways_drift: float
    vertical_drift: float
    control_energy: float
    mean_angular_speed: float
    simulated_seconds: float
    actuator_count: int
    body_count: int
    total_volume: float
    build_failed: bool = False
    disqualified: bool = False
    failure_reason: str | None = None


@dataclass(frozen=True)
class WalkingEvaluationResult:
    fitness: float
    origin_distance: float
    average_origin_speed: float
    forward_distance: float
    average_forward_speed: float
    sideways_drift: float
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


@dataclass(frozen=True)
class OriginDistanceEvaluationResult:
    fitness: float
    origin_distance: float
    average_origin_speed: float
    forward_distance: float
    average_forward_speed: float
    sideways_drift: float
    vertical_drift: float
    control_energy: float
    mean_angular_speed: float
    simulated_seconds: float
    actuator_count: int
    body_count: int
    total_volume: float
    build_failed: bool = False
    disqualified: bool = False
    failure_reason: str | None = None


@dataclass(frozen=True)
class FlyingEvaluationResult:
    fitness: float
    origin_distance: float
    average_origin_speed: float
    forward_distance: float
    average_forward_speed: float
    sideways_drift: float
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


def evaluate_x_axis_swimming(
    genotype: Genotype,
    config: SwimmingEvaluationConfig | None = None,
) -> SwimmingEvaluationResult:
    """Score a genotype by how well it swims in the positive x direction."""
    config = config or SwimmingEvaluationConfig()
    built = _build_model(genotype, config, "swimming_x")
    if isinstance(built, str):
        return _failed_swimming(config, built)
    model, data, builder = built

    metrics = _run_controlled_episode(model, data, builder, config)
    if isinstance(metrics, str):
        return _failed_swimming(config, metrics)

    fitness = (
        config.forward_speed_weight * metrics["average_forward_speed"]
        - config.energy_weight * metrics["control_energy"]
        - config.sideways_drift_weight * metrics["sideways_drift"]
        - config.vertical_drift_weight * metrics["vertical_drift"]
        - config.angular_speed_weight * metrics["mean_angular_speed"]
        - config.body_count_weight * metrics["body_count"]
        - config.volume_weight * _excess_volume(
            metrics["total_volume"], config.volume_penalty_cutoff
        )
    )
    if not math.isfinite(fitness):
        return _failed_swimming(config, "Simulation produced a non-finite fitness.")

    return SwimmingEvaluationResult(fitness=fitness, **metrics)


def evaluate_origin_distance(
    genotype: Genotype,
    config: SwimmingAwayEvaluationConfig | None = None,
) -> OriginDistanceEvaluationResult:
    """Score a genotype by final root distance from its starting position."""
    config = config or SwimmingAwayEvaluationConfig()
    built = _build_model(genotype, config, "swimming_away")
    if isinstance(built, str):
        return _failed_origin_distance(config, built)
    model, data, builder = built

    metrics = _run_controlled_episode(model, data, builder, config)
    if isinstance(metrics, str):
        return _failed_origin_distance(config, metrics)

    fitness = (
        config.distance_weight * metrics["average_origin_speed"]
        - config.energy_weight * metrics["control_energy"]
        - config.angular_speed_weight * metrics["mean_angular_speed"]
        - config.body_count_weight * metrics["body_count"]
        - config.volume_weight * _excess_volume(
            metrics["total_volume"], config.volume_penalty_cutoff
        )
    )
    if not math.isfinite(fitness):
        return _failed_origin_distance(
            config, "Simulation produced a non-finite fitness."
        )

    return OriginDistanceEvaluationResult(fitness=fitness, **metrics)


def evaluate_x_axis_flying(
    genotype: Genotype,
    config: FlyingEvaluationConfig | None = None,
) -> FlyingEvaluationResult:
    """Score flight by horizontal pre-contact speed while penalizing altitude loss."""
    config = config or FlyingEvaluationConfig()
    built = _build_model(genotype, config, "flying_x")
    if isinstance(built, str):
        return _failed_flying(config, built)
    model, data, builder = built

    failure = initialize_flying_model(model, data)
    if failure is not None:
        return _failed_flying(config, failure)

    evaluated = _flying_metrics_with_passive_baseline(
        model,
        data,
        builder,
        config,
        speed_metric="average_forward_speed",
    )
    if isinstance(evaluated, str):
        return _failed_flying(config, evaluated)
    metrics, fitness = evaluated
    if not math.isfinite(fitness):
        return _failed_flying(config, "Simulation produced a non-finite fitness.")

    return FlyingEvaluationResult(fitness=fitness, **metrics)


def evaluate_flying_away(
    genotype: Genotype,
    config: FlyingAwayEvaluationConfig | None = None,
) -> FlyingEvaluationResult:
    """Score flight by horizontal pre-contact speed from the starting point."""
    config = config or FlyingAwayEvaluationConfig()
    built = _build_model(genotype, config, "flying_away")
    if isinstance(built, str):
        return _failed_flying(config, built)
    model, data, builder = built

    failure = initialize_flying_model(model, data)
    if failure is not None:
        return _failed_flying(config, failure)

    evaluated = _flying_metrics_with_passive_baseline(
        model,
        data,
        builder,
        config,
        speed_metric="average_origin_speed",
    )
    if isinstance(evaluated, str):
        return _failed_flying(config, evaluated)
    metrics, fitness = evaluated
    if not math.isfinite(fitness):
        return _failed_flying(config, "Simulation produced a non-finite fitness.")

    return FlyingEvaluationResult(fitness=fitness, **metrics)

def _flying_metrics_with_passive_baseline(
    model: mujoco.MjModel,
    initialized_data: mujoco.MjData,
    builder: PhenotypeBuilder,
    config: FlyingEvaluationConfig | FlyingAwayEvaluationConfig,
    speed_metric: str,
):
    controlled_data = _copy_simulation_state(model, initialized_data)
    passive_data = _copy_simulation_state(model, initialized_data)

    controlled_metrics = _run_flying_episode(
        model, controlled_data, builder, config, apply_controls=True
    )
    if isinstance(controlled_metrics, str):
        return controlled_metrics
    passive_metrics = _run_flying_episode(
        model, passive_data, builder, config, apply_controls=False
    )
    if isinstance(passive_metrics, str):
        return passive_metrics

    controlled_fitness = _flying_fitness(config, controlled_metrics, speed_metric)
    passive_fitness = _flying_fitness(config, passive_metrics, speed_metric)
    fitness_gain = controlled_fitness - passive_fitness
    gain_fraction = config.fitness_gain_fraction
    fitness = (
        fitness_gain * gain_fraction
        + controlled_fitness * (1.0 - gain_fraction)
    )
    controlled_metrics["controlled_fitness"] = controlled_fitness
    controlled_metrics["passive_fitness"] = passive_fitness
    controlled_metrics["fitness_gain"] = fitness_gain
    return controlled_metrics, fitness


def _copy_simulation_state(
    model: mujoco.MjModel, source: mujoco.MjData
) -> mujoco.MjData:
    copied = mujoco.MjData(model)
    copied.time = source.time
    copied.qpos[:] = source.qpos
    copied.qvel[:] = source.qvel
    copied.ctrl[:] = source.ctrl
    if copied.act.size:
        copied.act[:] = source.act
    if copied.mocap_pos.size:
        copied.mocap_pos[:] = source.mocap_pos
    if copied.mocap_quat.size:
        copied.mocap_quat[:] = source.mocap_quat
    mujoco.mj_forward(model, copied)
    return copied


def evaluate_walking_away(
    genotype: Genotype,
    config: WalkingAwayEvaluationConfig | None = None,
) -> WalkingEvaluationResult:
    """Score ground locomotion by final root distance from its starting position."""
    config = config or WalkingAwayEvaluationConfig()
    built = _build_model(genotype, config, "walking_away")
    if isinstance(built, str):
        return _failed_walking(config, built)
    model, data, builder = built

    failure = initialize_walking_model(model, data)
    if failure is not None:
        return _failed_walking(config, failure)
    failure = _walking_height_failure_reason(model, data, config)
    if failure is not None:
        return _failed_walking(config, failure)
    failure = settle_walking_model(model, data, config)
    if failure is not None:
        return _failed_walking(config, failure)
    data.time = 0.0

    metrics = _run_controlled_episode(
        model,
        data,
        builder,
        config,
        root_body_name=f"{genotype.root}_1",
        horizontal_origin_distance=True,
    )
    if isinstance(metrics, str):
        return _failed_walking(config, metrics)

    walking_metrics = {
        key: value
        for key, value in metrics.items()
        if key != "vertical_drift"
    }
    fitness = (
        config.distance_weight * walking_metrics["average_origin_speed"]
        - config.energy_weight * walking_metrics["control_energy"]
        - config.angular_speed_weight * walking_metrics["mean_angular_speed"]
        - config.height_loss_weight * walking_metrics["height_loss"]
        - config.body_count_weight * walking_metrics["body_count"]
        - config.volume_weight * _excess_volume(
            walking_metrics["total_volume"], config.volume_penalty_cutoff
        )
    )
    if not math.isfinite(fitness):
        return _failed_walking(config, "Simulation produced a non-finite fitness.")

    return WalkingEvaluationResult(fitness=fitness, **walking_metrics)


def evaluate_x_axis_walking(
    genotype: Genotype,
    config: WalkingEvaluationConfig | None = None,
) -> WalkingEvaluationResult:
    """Score positive-x ground locomotion while penalizing falling and rolling."""
    config = config or WalkingEvaluationConfig()
    built = _build_model(genotype, config, "walking_x")
    if isinstance(built, str):
        return _failed_walking(config, built)
    model, data, builder = built

    failure = initialize_walking_model(model, data)
    if failure is not None:
        return _failed_walking(config, failure)
    failure = _walking_height_failure_reason(model, data, config)
    if failure is not None:
        return _failed_walking(config, failure)
    failure = settle_walking_model(model, data, config)
    if failure is not None:
        return _failed_walking(config, failure)
    data.time = 0.0

    metrics = _run_controlled_episode(
        model,
        data,
        builder,
        config,
        root_body_name=f"{genotype.root}_1",
    )
    if isinstance(metrics, str):
        return _failed_walking(config, metrics)

    walking_metrics = {
        key: value
        for key, value in metrics.items()
        if key != "vertical_drift"
    }
    fitness = (
        config.forward_speed_weight * walking_metrics["average_forward_speed"]
        - config.energy_weight * walking_metrics["control_energy"]
        - config.sideways_drift_weight * walking_metrics["sideways_drift"]
        - config.angular_speed_weight * walking_metrics["mean_angular_speed"]
        - config.upright_weight * walking_metrics["mean_upright_error"]
        - config.height_loss_weight * walking_metrics["height_loss"]
        - config.body_count_weight * walking_metrics["body_count"]
        - config.volume_weight * _excess_volume(
            walking_metrics["total_volume"], config.volume_penalty_cutoff
        )
    )
    if not math.isfinite(fitness):
        return _failed_walking(config, "Simulation produced a non-finite fitness.")

    return WalkingEvaluationResult(fitness=fitness, **walking_metrics)


def evaluate_for_task(genotype: Genotype, config: EvaluationConfig):
    """Dispatch evaluation from the concrete task configuration type."""
    if isinstance(config, FlyingAwayEvaluationConfig):
        return evaluate_flying_away(genotype, config)
    if isinstance(config, FlyingEvaluationConfig):
        return evaluate_x_axis_flying(genotype, config)
    if isinstance(config, WalkingAwayEvaluationConfig):
        return evaluate_walking_away(genotype, config)
    if isinstance(config, WalkingEvaluationConfig):
        return evaluate_x_axis_walking(genotype, config)
    if isinstance(config, SwimmingAwayEvaluationConfig):
        return evaluate_origin_distance(genotype, config)
    return evaluate_x_axis_swimming(genotype, config)


def task_for_config(config: EvaluationConfig) -> str:
    if isinstance(config, FlyingAwayEvaluationConfig):
        return "flying_away"
    if isinstance(config, FlyingEvaluationConfig):
        return "flying_x"
    if isinstance(config, WalkingAwayEvaluationConfig):
        return "walking_away"
    if isinstance(config, WalkingEvaluationConfig):
        return "walking_x"
    if isinstance(config, SwimmingAwayEvaluationConfig):
        return "swimming_away"
    return "swimming_x"


def initialize_walking_model(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> str | None:
    """Place a walking creature above the floor and reject penetration."""
    mujoco.mj_forward(model, data)
    floor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
    )
    if floor_id < 0:
        return None

    free_joint_ids = np.flatnonzero(
        model.jnt_type == mujoco.mjtJoint.mjJNT_FREE
    )
    if free_joint_ids.size == 0:
        return "Walking creature has no free root joint."
    root_qpos_adr = int(model.jnt_qposadr[int(free_joint_ids[0])])
    floor_z = float(data.geom_xpos[floor_id, 2])
    lowest_z = min(
        _geom_lowest_world_z(model, data, geom_id)
        for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id] != 0
    )
    required_shift = floor_z + _WALKING_FLOOR_CLEARANCE - lowest_z
    if required_shift > 0.0:
        data.qpos[root_qpos_adr + 2] += required_shift
    mujoco.mj_forward(model, data)
    if _has_floor_penetration(model, data, floor_id):
        return INITIAL_FLOOR_OVERLAP_REASON
    return None


def initialize_flying_model(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> str | None:
    """Place a flying creature above the floor before scoring starts."""
    mujoco.mj_forward(model, data)
    floor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
    )
    if floor_id < 0:
        return None

    free_joint_ids = np.flatnonzero(
        model.jnt_type == mujoco.mjtJoint.mjJNT_FREE
    )
    if free_joint_ids.size == 0:
        return "Flying creature has no free root joint."
    root_qpos_adr = int(model.jnt_qposadr[int(free_joint_ids[0])])
    floor_z = float(data.geom_xpos[floor_id, 2])
    lowest_z = min(
        _geom_lowest_world_z(model, data, geom_id)
        for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id] != 0
    )
    required_shift = floor_z + _FLYING_FLOOR_CLEARANCE - lowest_z
    if required_shift > 0.0:
        data.qpos[root_qpos_adr + 2] += required_shift
    mujoco.mj_forward(model, data)
    if _has_floor_contact(model, data, floor_id):
        return INITIAL_FLOOR_OVERLAP_REASON
    return None


def _walking_height_failure_reason(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: WalkingEvaluationConfig | WalkingAwayEvaluationConfig,
) -> str | None:
    max_height = getattr(config, "max_creature_height", 0.0)
    if max_height <= 0.0:
        return None
    height = _creature_height(model, data)
    if height <= max_height:
        return None
    return (
        f"{MAXIMUM_CREATURE_HEIGHT_REASON} Creature height is "
        f"{height:.6f} m; maximum is {max_height:.6f} m."
    )


def _creature_height(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    creature_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id] != 0
    ]
    if not creature_geom_ids:
        return 0.0
    lowest_z = min(
        _geom_lowest_world_z(model, data, geom_id)
        for geom_id in creature_geom_ids
    )
    highest_z = max(
        _geom_highest_world_z(model, data, geom_id)
        for geom_id in creature_geom_ids
    )
    return max(0.0, highest_z - lowest_z)


def _geom_highest_world_z(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> float:
    return float(data.geom_xpos[geom_id, 2]) + _geom_vertical_half_extent(
        model, data, geom_id
    )


def _geom_vertical_half_extent(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> float:
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    geom_type = model.geom_type[geom_id]
    size = model.geom_size[geom_id]
    if geom_type in {
        mujoco.mjtGeom.mjGEOM_BOX,
        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    }:
        return float(np.abs(rotation[2]) @ size[:3])
    if geom_type in {
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
    }:
        radius = float(size[0])
        half_length = float(size[1])
        axis_vertical = float(rotation[2, 2])
        radial_vertical = math.sqrt(max(0.0, 1.0 - axis_vertical * axis_vertical))
        return abs(axis_vertical) * half_length + radius * radial_vertical
    return float(np.max(size[:3]))


def _geom_lowest_world_z(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> float:
    return float(data.geom_xpos[geom_id, 2]) - _geom_vertical_half_extent(
        model, data, geom_id
    )


def _has_floor_penetration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    floor_id: int,
) -> bool:
    for contact in data.contact:
        if floor_id in contact.geom and float(contact.dist) < 0.0:
            return True
    return False


def settle_walking_model(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: WalkingEvaluationConfig | WalkingAwayEvaluationConfig,
) -> str | None:
    """Let a walking creature fall onto the floor before controls and scoring."""
    settle_steps = max(0, math.ceil(config.settle_seconds / model.opt.timestep))
    for _ in range(settle_steps):
        previous_time = float(data.time)
        mujoco.mj_step(model, data)
        if (
            config.disallow_collision
            and _has_nonparent_self_collision(model, data)
        ):
            return DISALLOWED_COLLISION_REASON
        failure = simulation_failure_reason(data, previous_time, config)
        if failure is not None:
            return f"{failure} while settling."
    return None


def _build_model(genotype: Genotype, config: EvaluationConfig, task: str):
    try:
        builder = PhenotypeBuilder(
            genotype,
            max_node=config.max_node,
            task=task,
            self_collision=(
                config.self_collision or config.disallow_collision
            ),
            fluid_density=getattr(config, "fluid_density", None),
            fluid_viscosity=getattr(config, "fluid_viscosity", None),
            fluid_shape=getattr(config, "fluid_shape", None),
            fluid_coef=getattr(config, "fluid_coef", None),
            gravity=getattr(config, "gravity", None),
        )
        mjcf = builder.build()
    except PhenotypeBuildAbort as error:
        return str(error)
    try:
        model = mujoco.MjModel.from_xml_string(mjcf)
    except ValueError as error:
        return str(error)
    volume_failure = _creature_volume_failure_reason(model, config)
    if volume_failure is not None:
        return volume_failure
    total_volume = _creature_volume(model)
    if total_volume > config.max_volume:
        return (
            f"Creature volume {total_volume:.6f} m^3 exceeds maximum "
            f"allowed volume {config.max_volume:.6f} m^3."
        )
    return model, mujoco.MjData(model), builder


def _run_flying_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    builder: PhenotypeBuilder,
    config: FlyingEvaluationConfig | FlyingAwayEvaluationConfig,
    apply_controls: bool = True,
):
    actuator_ids = _actuator_ids(model, builder.actuator_controllers)
    body_count = max(model.nbody - 1, 0)
    total_volume = _creature_volume(model)
    target_direction = _normalized_target_direction(config.target_direction)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    initial_center_of_mass = _creature_center_of_mass(model, data)
    previous_time = data.time
    control_energy = 0.0
    actuator_gear_norms = np.linalg.norm(
        model.actuator_gear[actuator_ids],
        axis=1,
    )
    angular_speed_sum = 0.0
    sample_count = 0
    first_ground_contact_time = None
    distance_measurement_position = None
    if floor_id >= 0 and _has_floor_contact(model, data, floor_id):
        first_ground_contact_time = 0.0
        distance_measurement_position = initial_center_of_mass.copy()

    max_steps = max(1, math.ceil(config.episode_seconds / model.opt.timestep))
    for _ in range(max_steps):
        if data.time >= config.episode_seconds:
            break
        if apply_controls:
            _apply_open_loop_controller(data, actuator_ids, builder.actuator_controllers)
        else:
            data.ctrl[actuator_ids] = 0.0
        ctrl = data.ctrl.copy()
        mujoco.mj_step(model, data)
        if (
            config.disallow_collision
            and _has_nonparent_self_collision(model, data)
        ):
            return DISALLOWED_COLLISION_REASON
        if (
            floor_id >= 0
            and first_ground_contact_time is None
            and _has_floor_contact(model, data, floor_id)
        ):
            first_ground_contact_time = float(data.time)
            distance_measurement_position = _creature_center_of_mass(model, data)
        failure = simulation_failure_reason(data, previous_time, config)
        if failure is not None:
            return failure

        dt = data.time - previous_time
        previous_time = data.time
        control_effort = ctrl[actuator_ids] * actuator_gear_norms
        control_energy += float(dt * (control_effort @ control_effort))
        if model.nv >= 6:
            angular_speed_sum += float(np.linalg.norm(data.qvel[3:6]))
        sample_count += 1

    final_center_of_mass = _creature_center_of_mass(model, data)
    if distance_measurement_position is None:
        distance_measurement_position = final_center_of_mass
    displacement = distance_measurement_position - initial_center_of_mass
    horizontal_displacement = displacement.copy()
    horizontal_displacement[2] = 0.0
    forward_distance = float(horizontal_displacement @ np.asarray(target_direction))
    origin_distance = float(np.linalg.norm(horizontal_displacement))
    simulated_seconds = max(float(data.time), model.opt.timestep)
    measurement_seconds = simulated_seconds
    ground_touch_penalty = 0.0
    no_ground_touch_bonus = config.no_ground_touch_bonus
    if first_ground_contact_time is not None:
        measurement_seconds = max(first_ground_contact_time, model.opt.timestep)
        touch_fraction = min(first_ground_contact_time / config.episode_seconds, 1.0)
        ground_touch_penalty = config.ground_touch_weight * (1.0 - touch_fraction)
        no_ground_touch_bonus = 0.0
    height_loss = max(
        0.0, float(initial_center_of_mass[2] - final_center_of_mass[2])
    )
    lateral = horizontal_displacement - forward_distance * np.asarray(target_direction)

    return {
        "origin_distance": origin_distance,
        "average_origin_speed": origin_distance / measurement_seconds,
        "forward_distance": forward_distance,
        "average_forward_speed": forward_distance / measurement_seconds,
        "sideways_drift": abs(float(lateral[1])),
        "height_loss": height_loss,
        "first_ground_contact_time": first_ground_contact_time,
        "ground_touch_penalty": ground_touch_penalty,
        "no_ground_touch_bonus": no_ground_touch_bonus,
        "control_energy": control_energy,
        "mean_angular_speed": angular_speed_sum / max(sample_count, 1),
        "simulated_seconds": simulated_seconds,
        "actuator_count": len(actuator_ids),
        "body_count": body_count,
        "total_volume": total_volume,
    }


def _flying_fitness(
    config: FlyingEvaluationConfig | FlyingAwayEvaluationConfig,
    metrics: dict,
    speed_metric: str,
) -> float:
    return (
        config.distance_weight * metrics[speed_metric]
        + metrics["no_ground_touch_bonus"]
        - config.height_loss_weight * metrics["height_loss"]
        - metrics["ground_touch_penalty"]
        - config.energy_weight * metrics["control_energy"]
        - config.angular_speed_weight * metrics["mean_angular_speed"]
        - config.body_count_weight * metrics["body_count"]
        - config.volume_weight * _excess_volume(
            metrics["total_volume"], config.volume_penalty_cutoff
        )
    )


def _run_controlled_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    builder: PhenotypeBuilder,
    config: EvaluationConfig,
    root_body_name: str | None = None,
    horizontal_origin_distance: bool = False,
):
    actuator_ids = _actuator_ids(model, builder.actuator_controllers)
    body_count = max(model.nbody - 1, 0)
    total_volume = _creature_volume(model)
    target_direction = _normalized_target_direction(config.target_direction)
    initial_center_of_mass = _creature_center_of_mass(model, data)
    previous_time = data.time
    control_energy = 0.0
    actuator_gear_norms = np.linalg.norm(
        model.actuator_gear[actuator_ids],
        axis=1,
    )
    angular_speed_sum = 0.0
    upright_error_sum = 0.0
    sample_count = 0
    root_body_id = -1
    if root_body_name is not None:
        root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)

    max_steps = max(1, math.ceil(config.episode_seconds / model.opt.timestep))
    for _ in range(max_steps):
        if data.time >= config.episode_seconds:
            break
        _apply_open_loop_controller(data, actuator_ids, builder.actuator_controllers)
        ctrl = data.ctrl.copy()
        mujoco.mj_step(model, data)
        if (
            config.disallow_collision
            and _has_nonparent_self_collision(model, data)
        ):
            return DISALLOWED_COLLISION_REASON
        failure = simulation_failure_reason(data, previous_time, config)
        if failure is not None:
            return failure

        dt = data.time - previous_time
        previous_time = data.time
        control_effort = ctrl[actuator_ids] * actuator_gear_norms
        control_energy += float(dt * (control_effort @ control_effort))
        if model.nv >= 6:
            angular_speed_sum += float(np.linalg.norm(data.qvel[3:6]))
        if root_body_id >= 0:
            upright_error_sum += max(0.0, 1.0 - float(data.xmat[root_body_id, 8]))
        sample_count += 1

    final_center_of_mass = _creature_center_of_mass(model, data)
    displacement = final_center_of_mass - initial_center_of_mass
    forward_distance = float(displacement @ target_direction)
    simulated_seconds = max(float(data.time), model.opt.timestep)
    average_forward_speed = forward_distance / simulated_seconds
    origin_displacement = displacement.copy()
    if horizontal_origin_distance:
        origin_displacement[2] = 0.0
    origin_distance = float(np.linalg.norm(origin_displacement))
    average_origin_speed = origin_distance / simulated_seconds
    lateral = displacement - forward_distance * np.asarray(target_direction)

    metrics = {
        "origin_distance": origin_distance,
        "average_origin_speed": average_origin_speed,
        "forward_distance": forward_distance,
        "average_forward_speed": average_forward_speed,
        "sideways_drift": abs(float(lateral[1])),
        "vertical_drift": abs(float(displacement[2])),
        "control_energy": control_energy,
        "mean_angular_speed": angular_speed_sum / max(sample_count, 1),
        "simulated_seconds": simulated_seconds,
        "actuator_count": len(actuator_ids),
        "body_count": body_count,
        "total_volume": total_volume,
    }
    if root_body_id >= 0:
        min_center_height = (
            initial_center_of_mass[2]
            * getattr(config, "min_center_height_fraction", 0.0)
        )
        if final_center_of_mass[2] < min_center_height:
            return WALKING_CENTER_HEIGHT_DROP_REASON
        metrics["height_loss"] = max(
            0.0, float(initial_center_of_mass[2] - final_center_of_mass[2])
        )
        metrics["mean_upright_error"] = upright_error_sum / max(sample_count, 1)
    return metrics


def _failed_swimming(config: SwimmingEvaluationConfig, reason: str):
    return SwimmingEvaluationResult(
        fitness=config.build_failure_fitness,
        origin_distance=0.0,
        average_origin_speed=0.0,
        forward_distance=0.0,
        average_forward_speed=0.0,
        sideways_drift=0.0,
        vertical_drift=0.0,
        control_energy=0.0,
        mean_angular_speed=0.0,
        simulated_seconds=0.0,
        actuator_count=0,
        body_count=0,
        total_volume=0.0,
        build_failed=reason != DISALLOWED_COLLISION_REASON,
        disqualified=reason == DISALLOWED_COLLISION_REASON,
        failure_reason=reason,
    )


def _failed_origin_distance(config: SwimmingAwayEvaluationConfig, reason: str):
    return OriginDistanceEvaluationResult(
        fitness=config.build_failure_fitness,
        origin_distance=0.0,
        average_origin_speed=0.0,
        forward_distance=0.0,
        average_forward_speed=0.0,
        sideways_drift=0.0,
        vertical_drift=0.0,
        control_energy=0.0,
        mean_angular_speed=0.0,
        simulated_seconds=0.0,
        actuator_count=0,
        body_count=0,
        total_volume=0.0,
        build_failed=reason != DISALLOWED_COLLISION_REASON,
        disqualified=reason == DISALLOWED_COLLISION_REASON,
        failure_reason=reason,
    )


def _failed_flying(
    config: FlyingEvaluationConfig | FlyingAwayEvaluationConfig, reason: str
):
    return FlyingEvaluationResult(
        fitness=config.build_failure_fitness,
        origin_distance=0.0,
        average_origin_speed=0.0,
        forward_distance=0.0,
        average_forward_speed=0.0,
        sideways_drift=0.0,
        height_loss=0.0,
        first_ground_contact_time=None,
        ground_touch_penalty=0.0,
        no_ground_touch_bonus=0.0,
        controlled_fitness=0.0,
        passive_fitness=0.0,
        fitness_gain=0.0,
        control_energy=0.0,
        mean_angular_speed=0.0,
        simulated_seconds=0.0,
        actuator_count=0,
        body_count=0,
        total_volume=0.0,
        build_failed=reason != DISALLOWED_COLLISION_REASON,
        disqualified=reason == DISALLOWED_COLLISION_REASON,
        failure_reason=reason,
    )


def _failed_walking(config: WalkingEvaluationConfig | WalkingAwayEvaluationConfig, reason: str):
    return WalkingEvaluationResult(
        fitness=config.build_failure_fitness,
        origin_distance=0.0,
        average_origin_speed=0.0,
        forward_distance=0.0,
        average_forward_speed=0.0,
        sideways_drift=0.0,
        height_loss=0.0,
        control_energy=0.0,
        mean_angular_speed=0.0,
        mean_upright_error=0.0,
        simulated_seconds=0.0,
        actuator_count=0,
        body_count=0,
        total_volume=0.0,
        build_failed=reason != DISALLOWED_COLLISION_REASON,
        disqualified=reason == DISALLOWED_COLLISION_REASON,
        failure_reason=reason,
    )


def _normalized_target_direction(target_direction: Sequence[float]):
    if len(target_direction) != 3:
        raise ValueError("target_direction must have three components")
    norm = math.sqrt(sum(component * component for component in target_direction))
    if norm == 0.0:
        raise ValueError("target_direction must be non-zero")
    return tuple(component / norm for component in target_direction)


def _excess_volume(total_volume: float, cutoff: float) -> float:
    return max(0.0, total_volume - cutoff)


def _creature_center_of_mass(
    model: mujoco.MjModel, data: mujoco.MjData
) -> np.ndarray:
    body_ids = np.arange(1, model.nbody)
    masses = model.body_mass[body_ids]
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        return data.qpos[:3].copy()
    return (data.xipos[body_ids] * masses[:, None]).sum(axis=0) / total_mass


def _has_floor_contact(
    model: mujoco.MjModel, data: mujoco.MjData, floor_id: int
) -> bool:
    for contact in data.contact:
        if floor_id not in contact.geom:
            continue
        first_geom, second_geom = map(int, contact.geom)
        other_geom = second_geom if first_geom == floor_id else first_geom
        if other_geom >= 0 and model.geom_bodyid[other_geom] != 0:
            return True
    return False


def _creature_volume(model: mujoco.MjModel) -> float:
    return sum(_creature_body_volumes(model).values())


def _creature_body_volumes(model: mujoco.MjModel) -> dict[int, float]:
    body_volumes = {body_id: 0.0 for body_id in range(1, model.nbody)}
    if not body_volumes:
        return body_volumes

    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        if body_id == 0:
            continue
        geom_type = model.geom_type[geom_id]
        size = model.geom_size[geom_id]
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            volume = 8.0 * float(np.prod(size[:3]))
        elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            volume = 4.0 / 3.0 * math.pi * float(np.prod(size[:3]))
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            radius = float(size[0])
            half_length = float(size[1])
            volume = (
                2.0 * math.pi * radius * radius * half_length
                + 4.0 / 3.0 * math.pi * radius ** 3
            )
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            radius = float(size[0])
            half_length = float(size[1])
            volume = 2.0 * math.pi * radius * radius * half_length
        else:
            continue
        body_volumes[body_id] += volume
    return body_volumes


def _creature_volume_failure_reason(
    model: mujoco.MjModel,
    config: EvaluationConfig,
) -> str | None:
    min_body_volume = getattr(config, "min_body_volume", 0.0)
    body_volumes = _creature_body_volumes(model)
    if min_body_volume > 0.0:
        for body_id, body_volume in body_volumes.items():
            if body_volume >= min_body_volume:
                continue
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name is None:
                body_name = f"body {body_id}"
            return (
                f"{MINIMUM_BODY_VOLUME_REASON} {body_name!r} has volume "
                f"{body_volume:.6g} m^3; minimum is {min_body_volume:.6g} m^3."
            )

    min_total_volume = getattr(config, "min_total_volume", 0.0)
    total_volume = sum(body_volumes.values())
    if total_volume < min_total_volume:
        return (
            f"{MINIMUM_TOTAL_VOLUME_REASON} Creature has volume "
            f"{total_volume:.6g} m^3; minimum is {min_total_volume:.6g} m^3."
        )
    return None


def _actuator_ids(model: mujoco.MjModel, controllers: list[ActuatorController]):
    return [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, controller.motor_name)
        for controller in controllers
    ]


def _has_nonparent_self_collision(
    model: mujoco.MjModel, data: mujoco.MjData
) -> bool:
    for contact in data.contact:
        first_geom, second_geom = map(int, contact.geom)
        if first_geom < 0 or second_geom < 0:
            continue
        first_body = int(model.geom_bodyid[first_geom])
        second_body = int(model.geom_bodyid[second_geom])
        if first_body == 0 or second_body == 0 or first_body == second_body:
            continue
        if (
            model.body_parentid[first_body] == second_body
            or model.body_parentid[second_body] == first_body
        ):
            continue
        return True
    return False


def simulation_failure_reason(
    data: mujoco.MjData,
    previous_time: float,
    config: EvaluationConfig,
) -> str | None:
    if not math.isfinite(float(data.time)) or data.time <= previous_time:
        return NUMERICAL_INSTABILITY_REASON
    if not _array_is_stable(data.qpos, config.max_abs_state_value):
        return NUMERICAL_INSTABILITY_REASON
    if not _array_is_stable(data.qvel, config.max_abs_velocity):
        return NUMERICAL_INSTABILITY_REASON
    if not _array_is_stable(data.qacc, config.max_abs_acceleration):
        return NUMERICAL_INSTABILITY_REASON
    return None


def _array_is_stable(values, max_abs_value: float):
    return bool(np.all(np.isfinite(values)) and np.all(np.abs(values) <= max_abs_value))


def _apply_open_loop_controller(
    data: mujoco.MjData,
    actuator_ids: list[int],
    actuator_controllers: list[ActuatorController],
):
    for actuator_id, controller in zip(actuator_ids, actuator_controllers):
        data.ctrl[actuator_id] = controller.amp * math.sin(
            controller.freq * data.time + controller.phase
        )
