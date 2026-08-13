from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import math
from typing import Callable, Protocol, Sequence

import mujoco
import numpy as np

from .genotype import Genotype
from .constants import EnvironmentFamily
from .graph_analysis import PhenotypeBuildAbort
from .control import (
    ActuatorController,
    actuator_ids_for_controllers,
    apply_controller,
    apply_open_loop_controller,
)
from .phenotype import PhenotypeBuilder


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
DEFAULT_UPRIGHT_ERROR_WEIGHT = 0.2


class EvaluationConfig(Protocol):
    """Settings required by shared evaluation and rendering infrastructure."""

    episode_seconds: float
    max_node: int
    self_collision: bool
    disallow_collision: bool
    build_failure_fitness: float
    max_abs_state_value: float
    max_abs_velocity: float
    max_abs_acceleration: float
    max_volume: float


@dataclass(frozen=True)
class ResultField:
    """One task-specific result value printed by the evaluation CLI."""

    label: str
    attribute: str
    format_spec: str = ".6f"
    none_text: str | None = None


@dataclass(frozen=True)
class TaskDefinition:
    """Config, evaluator, environment, and CLI presentation for one task."""

    name: str
    config_type: type
    evaluator: Callable
    environment_family: EnvironmentFamily
    title: str
    result_fields: tuple[ResultField, ...]
    order: int


TASK_REGISTRY: dict[str, TaskDefinition] = {}
_TASKS_BY_CONFIG_TYPE: dict[type, TaskDefinition] = {}
_BUILTIN_TASKS_LOADED = False


def register_task(
    name: str,
    *,
    config_type: type,
    environment_family: EnvironmentFamily,
    title: str,
    result_fields: tuple[ResultField, ...],
    order: int = 1_000,
):
    """Register an evaluator and all metadata needed by task consumers."""
    if name in TASK_REGISTRY:
        raise ValueError(f"Task name {name!r} is already registered")
    if config_type in _TASKS_BY_CONFIG_TYPE:
        raise ValueError(
            f"Config type {config_type.__name__} is already registered"
        )

    def decorator(evaluator: Callable) -> Callable:
        definition = TaskDefinition(
            name=name,
            config_type=config_type,
            evaluator=evaluator,
            environment_family=environment_family,
            title=title,
            result_fields=result_fields,
            order=order,
        )
        TASK_REGISTRY[name] = definition
        _TASKS_BY_CONFIG_TYPE[config_type] = definition
        return evaluator

    return decorator


def _load_builtin_tasks() -> None:
    """Import the built-in task catalog once so its decorators register tasks."""
    global _BUILTIN_TASKS_LOADED
    if not _BUILTIN_TASKS_LOADED:
        import_module(".evolution_tasks", package=__package__)
        _BUILTIN_TASKS_LOADED = True


def task_definition(task: str) -> TaskDefinition:
    """Return canonical metadata for a task name."""
    _load_builtin_tasks()
    return TASK_REGISTRY[task]


def task_definition_for_config(config: EvaluationConfig) -> TaskDefinition:
    """Return canonical task metadata for a concrete config instance."""
    _load_builtin_tasks()
    definition = _TASKS_BY_CONFIG_TYPE.get(type(config))
    if definition is None:
        raise TypeError(f"Unsupported evaluation config type: {type(config).__name__}")
    return definition


def evaluate_for_task(genotype: Genotype, config: EvaluationConfig):
    """Dispatch evaluation through the canonical task registry."""
    return task_definition_for_config(config).evaluator(genotype, config)


def task_for_config(config: EvaluationConfig) -> str:
    """Return the canonical task name for a concrete config instance."""
    return task_definition_for_config(config).name


def task_names() -> tuple[str, ...]:
    """Return registered task names in presentation order."""
    _load_builtin_tasks()
    return tuple(
        definition.name
        for definition in sorted(
            TASK_REGISTRY.values(),
            key=lambda definition: (definition.order, definition.name),
        )
    )


def _copy_simulation_state(
    model: mujoco.MjModel, source: mujoco.MjData
) -> mujoco.MjData:
    """Copy the dynamic MuJoCo state for an independent rollout."""
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
    config: EvaluationConfig,
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
    config: EvaluationConfig,
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


def _build_model(genotype: Genotype, config: EvaluationConfig):
    try:
        definition = task_definition_for_config(config)
        builder = PhenotypeBuilder(
            genotype,
            max_node=config.max_node,
            environment_family=definition.environment_family,
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
    config: EvaluationConfig,
    apply_controls: bool = True,
):
    actuator_ids = _actuator_ids(model, builder.actuator_controllers)
    body_count = max(model.nbody - 1, 0)
    total_volume = _creature_volume(model)
    target_direction = _normalized_target_direction(config.target_direction)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    mujoco.mj_forward(model, data)
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
            apply_controller(model, data, builder.actuator_controllers)
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
        "sideways_drift_speed": abs(float(lateral[1])) / measurement_seconds,
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
    mujoco.mj_forward(model, data)
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
        apply_controller(model, data, builder.actuator_controllers)
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
        "sideways_drift_speed": abs(float(lateral[1])) / simulated_seconds,
        "vertical_drift_speed": abs(float(displacement[2])) / simulated_seconds,
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
    return actuator_ids_for_controllers(model, controllers)


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
    apply_open_loop_controller(data, actuator_ids, actuator_controllers)
