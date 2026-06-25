from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence, TypeAlias

import mujoco
import numpy as np

from .genotype import Genotype
from .graph_analysis import PhenotypeBuildAbort
from .phenotype import ActuatorController, PhenotypeBuilder


DISALLOWED_COLLISION_REASON = "Disallowed non-parent self-collision detected."
NUMERICAL_INSTABILITY_REASON = "Simulation became numerically unstable."
INITIAL_FLOOR_OVERLAP_REASON = "Creature overlaps the floor at initialization."
_WALKING_FLOOR_CLEARANCE = 0.05


@dataclass(frozen=True)
class SwimmingEvaluationConfig:
    """Weights and simulation settings for x-axis swimming fitness."""

    episode_seconds: float = 10.0
    max_node: int = 500
    self_collision: bool = False
    disallow_collision: bool = False
    target_direction: Sequence[float] = (1.0, 0.0, 0.0)
    forward_speed_weight: float = 1.0
    energy_weight: float = 0.001
    sideways_drift_weight: float = 0.1
    vertical_drift_weight: float = 0.1
    angular_speed_weight: float = 0.01
    body_count_weight: float = 0.001
    volume_weight: float = 0.01
    volume_penalty_cutoff: float = 0.1
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
    energy_weight: float = 0.001
    sideways_drift_weight: float = 0.1
    angular_speed_weight: float = 0.01
    upright_weight: float = 0.2
    height_loss_weight: float = 0.2
    body_count_weight: float = 0.001
    volume_weight: float = 0.01
    volume_penalty_cutoff: float = 0.1
    max_volume: float = 1.0
    build_failure_fitness: float = -1_000.0
    max_abs_state_value: float = 1_000_000.0
    max_abs_velocity: float = 1_000.0
    max_abs_acceleration: float = 100_000.0


EvaluationConfig: TypeAlias = SwimmingEvaluationConfig | WalkingEvaluationConfig


@dataclass(frozen=True)
class SwimmingEvaluationResult:
    fitness: float
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


def evaluate_x_axis_swimming(
    genotype: Genotype,
    config: SwimmingEvaluationConfig | None = None,
) -> SwimmingEvaluationResult:
    """Score a genotype by how well it swims in the positive x direction."""
    config = config or SwimmingEvaluationConfig()
    built = _build_model(genotype, config, "swimming")
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


def evaluate_x_axis_walking(
    genotype: Genotype,
    config: WalkingEvaluationConfig | None = None,
) -> WalkingEvaluationResult:
    """Score positive-x ground locomotion while penalizing falling and rolling."""
    config = config or WalkingEvaluationConfig()
    built = _build_model(genotype, config, "walking")
    if isinstance(built, str):
        return _failed_walking(config, built)
    model, data, builder = built

    failure = initialize_walking_model(model, data)
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
    if isinstance(config, WalkingEvaluationConfig):
        return evaluate_x_axis_walking(genotype, config)
    return evaluate_x_axis_swimming(genotype, config)


def task_for_config(config: EvaluationConfig) -> str:
    return "walking" if isinstance(config, WalkingEvaluationConfig) else "swimming"


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
        _box_lowest_world_z(model, data, geom_id)
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


def _box_lowest_world_z(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> float:
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    vertical_half_extent = float(
        np.abs(rotation[2]) @ model.geom_size[geom_id, :3]
    )
    return float(data.geom_xpos[geom_id, 2] - vertical_half_extent)


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
    config: WalkingEvaluationConfig,
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
        )
        mjcf = builder.build()
    except PhenotypeBuildAbort as error:
        return str(error)
    model = mujoco.MjModel.from_xml_string(mjcf)
    total_volume = _creature_volume(model)
    if total_volume > config.max_volume:
        return (
            f"Creature volume {total_volume:.6f} m^3 exceeds maximum "
            f"allowed volume {config.max_volume:.6f} m^3."
        )
    return model, mujoco.MjData(model), builder


def _run_controlled_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    builder: PhenotypeBuilder,
    config: EvaluationConfig,
    root_body_name: str | None = None,
):
    actuator_ids = _actuator_ids(model, builder.actuator_controllers)
    body_count = max(model.nbody - 1, 0)
    total_volume = _creature_volume(model)
    target_direction = _normalized_target_direction(config.target_direction)
    initial_position = data.qpos[:3].copy()
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

    final_position = data.qpos[:3].copy()
    displacement = final_position - initial_position
    forward_distance = float(displacement @ target_direction)
    simulated_seconds = max(float(data.time), model.opt.timestep)
    average_forward_speed = forward_distance / simulated_seconds
    lateral = displacement - forward_distance * np.asarray(target_direction)

    metrics = {
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
        metrics["height_loss"] = max(0.0, float(initial_position[2] - final_position[2]))
        metrics["mean_upright_error"] = upright_error_sum / max(sample_count, 1)
    return metrics


def _failed_swimming(config: SwimmingEvaluationConfig, reason: str):
    return SwimmingEvaluationResult(
        fitness=config.build_failure_fitness,
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


def _failed_walking(config: WalkingEvaluationConfig, reason: str):
    return WalkingEvaluationResult(
        fitness=config.build_failure_fitness,
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


def _creature_volume(model: mujoco.MjModel) -> float:
    total_volume = 0.0
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == 0:
            continue
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        total_volume += 8.0 * float(np.prod(model.geom_size[geom_id, :3]))
    return total_volume


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
