"""Simulation mechanics shared by scoring and viewing.

Placing and settling a creature at the episode start, copying simulation
state, floor and height geometry, and the physics safety checks (numerical
instability and non-parent self-collision). ``evc3d.evaluation`` adds the
scoring rules and metrics on top; ``evc3d.scene`` uses these directly for the
viewer and MP4 recording, so all of them start and step identically.
"""

from __future__ import annotations

import math
from typing import Callable

import mujoco
import numpy as np

from .evolution_tasks.config import EnvironmentFamily, EvaluationConfig
from .evolution_tasks.shared import DISALLOWED_COLLISION_REASON

NUMERICAL_INSTABILITY_REASON = "Simulation became numerically unstable."


def start_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: EvaluationConfig,
    environment: EnvironmentFamily,
    start_check: Callable[[mujoco.MjModel, mujoco.MjData, EvaluationConfig], str | None] | None = None,
) -> str | None:
    """Prepare the episode start; return a failure reason, or None.

    Places the creature (``place_creature``), runs ``start_check`` on the placed
    state if given, settles (``settle_model``), and resets time to zero.
    Scoring passes its start rules as ``start_check``; viewing passes none, so
    any buildable creature starts from the same state the scored episode does.
    ``start_check`` must only read the state.
    """
    failure = place_creature(model, data, config, environment)
    if failure is not None:
        return failure
    if start_check is not None:
        failure = start_check(model, data, config)
        if failure is not None:
            return failure
    failure = settle_model(model, data, config)
    if failure is None:
        data.time = 0.0
    return failure


def copy_simulation_state(
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


def place_creature(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: EvaluationConfig,
    environment: EnvironmentFamily,
) -> str | None:
    """Apply the environment's floor clearance and initialization callback."""
    mujoco.mj_forward(model, data)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    clearance = environment.initial_floor_clearance
    if floor_id >= 0 and clearance is not None:
        free_joint_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if free_joint_ids.size == 0:
            return f"{environment.name} creature has no free root joint."
        root_qpos_adr = int(model.jnt_qposadr[int(free_joint_ids[0])])
        creature_geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] != 0]
        floor_z = float(data.geom_xpos[floor_id, 2])
        lowest_z = min(_geom_lowest_world_z(model, data, i) for i in creature_geoms)
        required_shift = floor_z + clearance - lowest_z
        if required_shift > 0.0:
            data.qpos[root_qpos_adr + 2] += required_shift
        mujoco.mj_forward(model, data)
    callback = environment.initialization_callback
    if callback is not None:
        failure = callback(model, data, config)
        if failure is not None:
            return failure
        mujoco.mj_forward(model, data)
    return None


def settle_model(
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
            and has_nonparent_self_collision(model, data)
        ):
            return DISALLOWED_COLLISION_REASON
        failure = simulation_failure_reason(data, previous_time, config)
        if failure is not None:
            return f"{failure} while settling."
    return None


def creature_height(model: mujoco.MjModel, data: mujoco.MjData) -> float:
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


def has_floor_contact(
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


def has_floor_penetration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    floor_id: int,
) -> bool:
    for contact in data.contact:
        if floor_id in contact.geom and float(contact.dist) < 0.0:
            return True
    return False


def has_nonparent_self_collision(
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
