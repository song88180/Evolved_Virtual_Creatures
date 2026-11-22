"""Apply actuator controllers to a running MuJoCo simulation."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .brain_functions import clamp_signal, evaluate_neuron, initial_state
from .brain_genes import SENSOR_ANGLE
from .controller_spec import ActuatorController, BrainSpec

BRAIN_STEPS_PER_CONTROL_STEP = 2


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
    brain: "BrainRuntime | None" = None,
) -> None:
    """Write ``data.ctrl`` for one control step.

    Graph-mode controllers are driven together by ``brain`` (a fresh
    ``BrainRuntime`` per episode), so they are skipped in the per-joint loop.
    """
    if brain is not None:
        brain.step(model, data)
    for controller in actuator_controllers:
        if controller.control_mode == "graph":
            continue
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


def create_brain_runtime(
    model: mujoco.MjModel,
    spec: BrainSpec | None,
    physics_substeps: int = 1,
) -> "BrainRuntime | None":
    """Return a fresh brain for one episode, or None when there is no graph brain.

    The brain updates once per control step, which is ``physics_substeps``
    MuJoCo steps long.
    """
    if spec is None:
        return None
    return BrainRuntime(model, spec, model.opt.timestep * physics_substeps)


class BrainRuntime:
    """Runs a compiled graph brain: sensors -> neurons (2 brain steps) -> effectors.

    Holds all neuron state, so create a new one for every episode. Signals
    live in one list: neuron outputs first, then the sensors the brain uses.
    """

    def __init__(self, model: mujoco.MjModel, spec: BrainSpec, control_dt: float):
        self.spec = spec
        self.dt = control_dt / BRAIN_STEPS_PER_CONTROL_STEP
        self.neuron_count = len(spec.neurons)
        self.signals = [0.0] * (self.neuron_count + len(spec.sensors))
        self.states = [initial_state(neuron.function) for neuron in spec.neurons]
        self.actuator_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, effector.motor_name)
            for effector in spec.effectors
        ]
        self._angle_sensors: list[tuple[int, int, int, bool]] = []
        self._contact_sensors: dict[tuple[int, str], int] = {}
        self._contact_bodies: dict[int, int] = {}  # geom id -> body id
        for offset, sensor in enumerate(spec.sensors):
            slot = self.neuron_count + offset
            if sensor.kind == SENSOR_ANGLE:
                joint_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, sensor.joint_name
                )
                if joint_id < 0:
                    raise ValueError(f"Unknown joint for brain: {sensor.joint_name!r}")
                self._angle_sensors.append((
                    slot,
                    int(model.jnt_qposadr[joint_id]),
                    sensor.index,
                    sensor.joint_type == "ball",
                ))
            else:
                body_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, sensor.body_name
                )
                if body_id < 0:
                    raise ValueError(f"Unknown body for brain: {sensor.body_name!r}")
                self._contact_sensors[(body_id, sensor.face)] = slot
                start = int(model.body_geomadr[body_id])
                for geom_id in range(start, start + int(model.body_geomnum[body_id])):
                    self._contact_bodies[geom_id] = body_id
        self._contact_slots = list(self._contact_sensors.values())

    def step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Read sensors, run the brain steps, and write the effector controls."""
        signals = self.signals
        self._read_sensors(model, data)
        neurons = self.spec.neurons
        for _ in range(BRAIN_STEPS_PER_CONTROL_STEP):
            outputs = []
            for neuron, state in zip(neurons, self.states):
                (s0, w0), (s1, w1), (s2, w2) = neuron.inputs
                outputs.append(
                    evaluate_neuron(
                        neuron.function,
                        w0 if s0 < 0 else w0 * signals[s0],
                        w1 if s1 < 0 else w1 * signals[s1],
                        w2 if s2 < 0 else w2 * signals[s2],
                        state,
                        self.dt,
                    )
                )
            signals[: self.neuron_count] = outputs
        for effector, actuator_id in zip(self.spec.effectors, self.actuator_ids):
            value = (
                effector.weight
                if effector.slot < 0
                else effector.weight * signals[effector.slot]
            )
            data.ctrl[actuator_id] = _scale_unit_output(
                max(-1.0, min(1.0, clamp_signal(value))),
                effector.ctrlrange,
            )

    def _read_sensors(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        signals = self.signals
        for slot, qpos_adr, index, is_ball in self._angle_sensors:
            if is_ball:
                signals[slot] = _rotation_vector(data.qpos[qpos_adr:qpos_adr + 4])[index]
            else:
                signals[slot] = float(data.qpos[qpos_adr])
        if not self._contact_slots:
            return
        for slot in self._contact_slots:
            signals[slot] = -1.0
        contacts = data.contact
        for index in range(data.ncon):
            for geom_id, direction in (
                (int(contacts.geom1[index]), 1.0),
                (int(contacts.geom2[index]), -1.0),
            ):
                body_id = self._contact_bodies.get(geom_id)
                if body_id is None:
                    continue
                # The contact normal points from geom1 to geom2, so the
                # outward direction of this geom's surface is +/- the normal.
                outward = direction * contacts.frame[index, :3]
                face = _contact_face(data.xmat[body_id].reshape(3, 3), outward)
                slot = self._contact_sensors.get((body_id, face))
                if slot is not None:
                    signals[slot] = 1.0


def _contact_face(body_rotation: np.ndarray, outward: np.ndarray) -> str:
    """The body face (``+x`` ... ``-z``) a world-frame outward direction points at."""
    local = body_rotation.T @ outward
    axis = int(np.argmax(np.abs(local)))
    return ("+" if local[axis] >= 0.0 else "-") + "xyz"[axis]


def _rotation_vector(quaternion: np.ndarray) -> tuple[float, float, float]:
    """Axis-angle vector (radians) of a unit quaternion ``(w, x, y, z)``."""
    w, x, y, z = (float(value) for value in quaternion)
    sine = math.sqrt(x * x + y * y + z * z)
    if sine < 1e-12:
        return (0.0, 0.0, 0.0)
    if w < 0.0:
        w, x, y, z, = -w, -x, -y, -z
    angle = 2.0 * math.atan2(sine, w)
    scale = angle / sine
    return (x * scale, y * scale, z * scale)
