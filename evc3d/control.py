"""Apply actuator controllers to a running MuJoCo simulation."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .controller_spec import ActuatorController


def actuator_ids_for_controllers(
    model: mujoco.MjModel,
    actuator_controllers: list[ActuatorController],
) -> list[int]:
    return [
        actuator_id
        for controller in actuator_controllers
        for actuator_id in _actuator_ids_for_controller(model, controller)
    ]


def apply_controller(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_controllers: list[ActuatorController],
) -> None:
    for controller in actuator_controllers:
        actuator_ids = _actuator_ids_for_controller(model, controller)
        if controller.control_mode == "sine":
            _apply_sine_controller(data, actuator_ids, controller)
        elif controller.control_mode == "neural":
            _apply_neural_controller(model, data, actuator_ids, controller)
        else:
            raise ValueError(f"Unknown control mode: {controller.control_mode!r}")


def _apply_sine_controller(
    data: mujoco.MjData,
    actuator_ids: list[int],
    controller: ActuatorController,
) -> None:
    value = controller.amp * math.sin(controller.freq * data.time + controller.phase)
    for actuator_id in actuator_ids:
        data.ctrl[actuator_id] = value


def _apply_neural_controller(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_ids: list[int],
    controller: ActuatorController,
) -> None:
    observation = _neural_observation(model, data, controller)
    hidden = np.tanh(
        np.asarray(controller.neural_w1, dtype=float) @ observation
        + np.asarray(controller.neural_b1, dtype=float)
    )
    raw_outputs = np.tanh(
        np.asarray(controller.neural_w2, dtype=float) @ hidden
        + np.asarray(controller.neural_b2, dtype=float)
    )
    for actuator_id, raw_output, ctrlrange in zip(
        actuator_ids,
        raw_outputs,
        controller.ctrlranges,
    ):
        data.ctrl[actuator_id] = _scale_unit_output(float(raw_output), ctrlrange)


def _neural_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: ActuatorController,
) -> np.ndarray:
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        controller.joint_name,
    )
    if joint_id < 0:
        raise ValueError(f"Unknown joint for controller: {controller.joint_name!r}")

    qpos_adr = int(model.jnt_qposadr[joint_id])
    qvel_adr = int(model.jnt_dofadr[joint_id])
    if controller.joint_type == "ball":
        joint_position = data.qpos[qpos_adr:qpos_adr + 4]
        joint_velocity = data.qvel[qvel_adr:qvel_adr + 3]
    else:
        joint_position = data.qpos[qpos_adr:qpos_adr + 1]
        joint_velocity = data.qvel[qvel_adr:qvel_adr + 1]

    root_body_id = 1 if model.nbody > 1 else 0
    root_up = data.xmat[root_body_id].reshape(3, 3)[:, 2]
    root_angular_velocity = data.cvel[root_body_id, :3]
    root_linear_velocity = data.cvel[root_body_id, 3:6]

    return np.concatenate(
        (
            np.asarray((1.0,), dtype=float),
            np.asarray(joint_position, dtype=float),
            np.asarray(joint_velocity, dtype=float),
            np.asarray(root_up, dtype=float),
            np.asarray(root_linear_velocity, dtype=float),
            np.asarray(root_angular_velocity, dtype=float),
        )
    )


def _actuator_ids_for_controller(
    model: mujoco.MjModel,
    controller: ActuatorController,
) -> list[int]:
    return [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, motor_name)
        for motor_name in controller.motor_names
    ]


def _scale_unit_output(
    raw_output: float,
    ctrlrange: tuple[float, float],
) -> float:
    lower, upper = ctrlrange
    normalized = max(-1.0, min(1.0, raw_output))
    if lower <= 0.0 <= upper:
        if normalized >= 0.0:
            return normalized * upper
        return normalized * abs(lower)
    return lower + (normalized + 1.0) * 0.5 * (upper - lower)
