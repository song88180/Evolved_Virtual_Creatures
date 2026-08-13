from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import mujoco
import numpy as np


NEURAL_HIDDEN_SIZE = 16
NEURAL_INPUT_DIMS = {
    "hinge": 12,
    "slide": 12,
    "ball": 17,
}
NEURAL_OUTPUT_DIMS = {
    "hinge": 1,
    "slide": 1,
    "ball": 3,
}
BALL_NEURAL_AXES = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


@dataclass
class ActuatorController:
    motor_names: tuple[str, ...]
    joint_name: str
    joint_type: str
    control_mode: str
    ctrlranges: tuple[tuple[float, float], ...]
    neural_w1: tuple[tuple[float, ...], ...] = ()
    neural_b1: tuple[float, ...] = ()
    neural_w2: tuple[tuple[float, ...], ...] = ()
    neural_b2: tuple[float, ...] = ()
    amp: float = 0.0
    freq: float = 0.0
    phase: float = 0.0

    @property
    def motor_name(self) -> str:
        return self.motor_names[0]


def zero_neural_parameters(
    joint_type: str,
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
]:
    input_dim = NEURAL_INPUT_DIMS[joint_type]
    output_dim = NEURAL_OUTPUT_DIMS[joint_type]
    return (
        tuple((0.0,) * input_dim for _ in range(NEURAL_HIDDEN_SIZE)),
        (0.0,) * NEURAL_HIDDEN_SIZE,
        tuple((0.0,) * NEURAL_HIDDEN_SIZE for _ in range(output_dim)),
        (0.0,) * output_dim,
    )


def normalized_neural_parameters(
    joint_type: str,
    neural_w1: Sequence[Sequence[float]],
    neural_b1: Sequence[float],
    neural_w2: Sequence[Sequence[float]],
    neural_b2: Sequence[float],
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
]:
    if not neural_w1 and not neural_b1 and not neural_w2 and not neural_b2:
        return zero_neural_parameters(joint_type)

    input_dim = NEURAL_INPUT_DIMS[joint_type]
    output_dim = NEURAL_OUTPUT_DIMS[joint_type]
    w1 = _normalize_matrix(neural_w1, NEURAL_HIDDEN_SIZE, input_dim, "neural_w1")
    b1 = _normalize_vector(neural_b1, NEURAL_HIDDEN_SIZE, "neural_b1")
    w2 = _normalize_matrix(neural_w2, output_dim, NEURAL_HIDDEN_SIZE, "neural_w2")
    b2 = _normalize_vector(neural_b2, output_dim, "neural_b2")
    return w1, b1, w2, b2


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


def apply_open_loop_controller(
    data: mujoco.MjData,
    actuator_ids: list[int],
    actuator_controllers: list[ActuatorController],
) -> None:
    for actuator_id, controller in zip(actuator_ids, actuator_controllers):
        data.ctrl[actuator_id] = controller.amp * math.sin(
            controller.freq * data.time + controller.phase
        )


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


def _normalize_matrix(
    matrix: Sequence[Sequence[float]],
    rows: int,
    columns: int,
    field_name: str,
) -> tuple[tuple[float, ...], ...]:
    if len(matrix) != rows:
        raise ValueError(f"{field_name} must contain {rows} rows")
    normalized = tuple(
        _normalize_vector(row, columns, f"{field_name} row")
        for row in matrix
    )
    return normalized


def _normalize_vector(
    vector: Sequence[float],
    length: int,
    field_name: str,
) -> tuple[float, ...]:
    if len(vector) != length:
        raise ValueError(f"{field_name} must contain {length} values")
    normalized = tuple(float(value) for value in vector)
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError(f"{field_name} values must be finite")
    return normalized
