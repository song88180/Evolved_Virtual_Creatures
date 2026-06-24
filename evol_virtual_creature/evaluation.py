from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence, TypeAlias

import mujoco
import numpy as np

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
    body_count_weight: float = 0.001
    build_failure_fitness: float = -1_000.0
    max_abs_state_value: float = 1_000_000.0


@dataclass(frozen=True)
class WalkingEvaluationConfig:
    """Weights and simulation settings for x-axis walking fitness."""

    episode_seconds: float = 10.0
    settle_seconds: float = 1.0
    max_node: int = 500
    target_direction: Sequence[float] = (1.0, 0.0, 0.0)
    forward_speed_weight: float = 1.0
    energy_weight: float = 0.001
    sideways_drift_weight: float = 0.1
    angular_speed_weight: float = 0.01
    upright_weight: float = 0.2
    height_loss_weight: float = 0.2
    body_count_weight: float = 0.001
    build_failure_fitness: float = -1_000.0
    max_abs_state_value: float = 1_000_000.0


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
    build_failed: bool = False
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
    build_failed: bool = False
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


def settle_walking_model(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: WalkingEvaluationConfig,
) -> str | None:
    """Let a walking creature fall onto the floor before controls and scoring."""
    settle_steps = max(0, math.ceil(config.settle_seconds / model.opt.timestep))
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
        if not _simulation_state_is_stable(data, config.max_abs_state_value):
            return "Simulation became numerically unstable while settling."
    return None


def _build_model(genotype: Genotype, config: EvaluationConfig, task: str):
    try:
        builder = PhenotypeBuilder(genotype, max_node=config.max_node, task=task)
        mjcf = builder.build()
    except PhenotypeBuildAbort as error:
        return str(error)
    model = mujoco.MjModel.from_xml_string(mjcf)
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
    target_direction = _normalized_target_direction(config.target_direction)
    initial_position = data.qpos[:3].copy()
    previous_time = data.time
    control_energy = 0.0
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
        if not _simulation_state_is_stable(data, config.max_abs_state_value):
            return "Simulation became numerically unstable."

        dt = data.time - previous_time
        previous_time = data.time
        control_energy += float(dt * (ctrl @ ctrl))
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
        build_failed=True,
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
        build_failed=True,
        failure_reason=reason,
    )


def _normalized_target_direction(target_direction: Sequence[float]):
    if len(target_direction) != 3:
        raise ValueError("target_direction must have three components")
    norm = math.sqrt(sum(component * component for component in target_direction))
    if norm == 0.0:
        raise ValueError("target_direction must be non-zero")
    return tuple(component / norm for component in target_direction)


def _actuator_ids(model: mujoco.MjModel, controllers: list[ActuatorController]):
    return [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, controller.motor_name)
        for controller in controllers
    ]


def _simulation_state_is_stable(data: mujoco.MjData, max_abs_state_value: float):
    return all(
        _array_is_stable(values, max_abs_state_value)
        for values in (data.qpos, data.qvel, data.qacc)
    )


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
