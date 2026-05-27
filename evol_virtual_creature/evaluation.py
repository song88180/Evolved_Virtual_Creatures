from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import mujoco

from .genotype import Genotype
from .graph_analysis import PhenotypeBuildAbort
from .phenotype import ActuatorController, PhenotypeBuilder


@dataclass(frozen=True)
class SwimmingEvaluationConfig:
    """Weights and simulation settings for x-axis swimming fitness."""

    episode_seconds: float = 10.0
    max_node: int = 500
    target_direction: Sequence[float] = (1.0, 0.0, 0.0)
    forward_speed_weight: float = 1.0
    energy_weight: float = 0.001
    sideways_drift_weight: float = 0.1
    vertical_drift_weight: float = 0.1
    angular_speed_weight: float = 0.01
    build_failure_fitness: float = -1_000.0


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
    build_failed: bool = False
    failure_reason: str | None = None


def evaluate_x_axis_swimming(
    genotype: Genotype,
    config: SwimmingEvaluationConfig | None = None,
) -> SwimmingEvaluationResult:
    """
    Score a genotype by how well it swims in the positive x direction.

    The current controller is open-loop: each generated actuator follows the
    sine-wave control parameters stored in its connection gene. The resulting
    fitness rewards average +x speed and penalizes wasted control effort,
    sideways/vertical drift, and excessive tumbling.
    """
    config = config or SwimmingEvaluationConfig()

    try:
        builder = PhenotypeBuilder(genotype, max_node=config.max_node)
        mjcf = builder.build()
    except PhenotypeBuildAbort as error:
        return _failed_evaluation(config, str(error))

    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)
    actuator_ids = _actuator_ids(model, builder.actuator_controllers)

    target_direction = _normalized_target_direction(config.target_direction)
    initial_position = data.qpos[:3].copy()
    previous_time = data.time
    control_energy = 0.0
    angular_speed_sum = 0.0
    sample_count = 0

    while data.time < config.episode_seconds:
        _apply_open_loop_controller(
            data=data,
            actuator_ids=actuator_ids,
            actuator_controllers=builder.actuator_controllers,
        )

        ctrl = data.ctrl.copy()
        mujoco.mj_step(model, data)

        dt = data.time - previous_time
        previous_time = data.time
        control_energy += float(dt * (ctrl @ ctrl))

        if model.nv >= 6:
            angular_speed_sum += float(
                math.sqrt(data.qvel[3] ** 2 + data.qvel[4] ** 2 + data.qvel[5] ** 2)
            )
            sample_count += 1

    final_position = data.qpos[:3].copy()
    displacement = final_position - initial_position
    forward_distance = float(displacement @ target_direction)
    simulated_seconds = max(float(data.time), model.opt.timestep)
    average_forward_speed = forward_distance / simulated_seconds

    lateral_displacement = [
        displacement[index] - forward_distance * target_direction[index]
        for index in range(3)
    ]
    sideways_drift = abs(float(lateral_displacement[1]))
    vertical_drift = abs(float(displacement[2]))
    mean_angular_speed = angular_speed_sum / max(sample_count, 1)

    fitness = (
        config.forward_speed_weight * average_forward_speed
        - config.energy_weight * control_energy
        - config.sideways_drift_weight * sideways_drift
        - config.vertical_drift_weight * vertical_drift
        - config.angular_speed_weight * mean_angular_speed
    )

    return SwimmingEvaluationResult(
        fitness=fitness,
        forward_distance=forward_distance,
        average_forward_speed=average_forward_speed,
        sideways_drift=sideways_drift,
        vertical_drift=vertical_drift,
        control_energy=control_energy,
        mean_angular_speed=mean_angular_speed,
        simulated_seconds=simulated_seconds,
        actuator_count=len(actuator_ids),
    )


def _failed_evaluation(
    config: SwimmingEvaluationConfig,
    failure_reason: str,
) -> SwimmingEvaluationResult:
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
        build_failed=True,
        failure_reason=failure_reason,
    )


def _normalized_target_direction(
    target_direction: Sequence[float],
) -> tuple[float, float, float]:
    if len(target_direction) != 3:
        raise ValueError("target_direction must have three components")

    norm = math.sqrt(sum(component * component for component in target_direction))
    if norm == 0.0:
        raise ValueError("target_direction must be non-zero")

    return tuple(component / norm for component in target_direction)


def _actuator_ids(
    model: mujoco.MjModel,
    actuator_controllers: list[ActuatorController],
) -> list[int]:
    return [
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            controller.motor_name,
        )
        for controller in actuator_controllers
    ]


def _apply_open_loop_controller(
    data: mujoco.MjData,
    actuator_ids: list[int],
    actuator_controllers: list[ActuatorController],
):
    for actuator_id, controller in zip(actuator_ids, actuator_controllers):
        data.ctrl[actuator_id] = controller.amp * math.sin(
            controller.freq * data.time + controller.phase
        )
