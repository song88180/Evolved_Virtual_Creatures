"""Score a genotype on a task.

``evaluate_task`` builds the creature, enforces the task's scoring rules
(volume limits, start rules, collision disqualification), runs the metric
rollout, and returns the task's typed result. Shared mechanics (placement,
settling, safety checks) live in ``evc3d.simulation``; task loading lives in
``evc3d.evolution_tasks``.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Sequence

import mujoco
import numpy as np

from .control import (
    actuator_ids_for_controllers,
    apply_controller,
)
from .evolution_tasks import shared as task_shared
from .genotype import Genotype
from .graph_analysis import PhenotypeBuildAbort
from .phenotype import PhenotypeBuilder
from .simulation import (
    copy_simulation_state,
    creature_height,
    has_floor_contact,
    has_floor_penetration,
    has_nonparent_self_collision,
    simulation_failure_reason,
    start_episode,
)


INITIAL_FLOOR_OVERLAP_REASON = "Creature overlaps the floor at initialization."
MINIMUM_BODY_VOLUME_REASON = "Creature body volume is below the minimum allowed volume."
MINIMUM_TOTAL_VOLUME_REASON = (
    "Creature total volume is below the minimum allowed volume."
)
WALKING_CENTER_HEIGHT_DROP_REASON = (
    "Creature center of mass dropped below the walking height threshold."
)
MAXIMUM_CREATURE_HEIGHT_REASON = "Creature height exceeds the maximum allowed height."
DEFAULT_UPRIGHT_ERROR_WEIGHT = task_shared.DEFAULT_UPRIGHT_ERROR_WEIGHT


def evaluate_task(
    genotype: Genotype,
    definition: task_shared.TaskDefinition,
    config: task_shared.EvaluationConfig | None = None,
    environment: task_shared.EnvironmentFamily | None = None,
):
    """Build, simulate, and score a genotype using its registered task callbacks.

    ``config`` and ``environment`` override the definition's own values for
    this call only; ``definition`` itself (often a shared registry object) is
    never modified.
    """
    overrides = {}
    if config is not None:
        overrides["config"] = config
    if environment is not None:
        overrides["environment"] = environment
    if overrides:
        definition = replace(definition, **overrides)
    config = definition.config
    if not isinstance(config, definition.config_type):
        raise TypeError(
            f"Task {definition.name!r} requires {definition.config_type.__name__}, "
            f"not {type(config).__name__}"
        )

    def failed(reason: str):
        return definition.failed_task_callback(config, reason)

    environment = definition.environment
    built = _build_model(genotype, definition)
    if isinstance(built, str):
        return failed(built)
    model, data, builder = built
    # Scored start: place, enforce the start rules, and settle. Viewing calls
    # start_episode without start_check, so both start from the same state.
    failure = start_episode(
        model, data, config, environment, start_check=start_rule_failure
    )
    if failure is not None:
        return failed(failure)

    policy = definition.rollout_policy
    initial_state = data
    if policy.passive_baseline:
        initial_state = copy_simulation_state(model, data)
    controlled_metrics = _run_episode(
        model,
        initial_state,
        builder,
        config,
        policy,
        root_body_name=(builder.root_body_name if policy.track_root_upright else None),
    )
    if isinstance(controlled_metrics, str):
        return failed(controlled_metrics)

    passive_metrics = None
    if policy.passive_baseline:
        passive_metrics = _run_episode(
            model,
            copy_simulation_state(model, data),
            builder,
            config,
            policy,
            apply_controls=False,
        )
        if isinstance(passive_metrics, str):
            return failed(passive_metrics)

    result = definition.fitness_callback(
        config, controlled_metrics, passive_metrics
    )
    if not math.isfinite(result.fitness):
        return failed("Simulation produced a non-finite fitness.")
    return result


def start_rule_failure(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: task_shared.EvaluationConfig,
) -> str | None:
    """Check the task's rules for a valid start (floor contact, creature height).

    Read-only: inspects the placed state without modifying it.
    """
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    contact_policy = config.initial_floor_contact_policy
    if contact_policy == "allow" or floor_id < 0:
        violates_floor_rule = False
    elif contact_policy == "no_penetration":
        violates_floor_rule = has_floor_penetration(model, data, floor_id)
    elif contact_policy == "no_contact":
        violates_floor_rule = has_floor_contact(model, data, floor_id)
    else:
        # Configs reject unknown policies, but never let one score as "allow".
        raise ValueError(f"Unknown initial floor contact policy: {contact_policy!r}")
    if violates_floor_rule:
        return INITIAL_FLOOR_OVERLAP_REASON
    return _walking_height_failure_reason(model, data, config)


def _walking_height_failure_reason(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: task_shared.EvaluationConfig,
) -> str | None:
    max_height = config.max_creature_height
    if max_height <= 0.0:
        return None
    height = creature_height(model, data)
    if height <= max_height:
        return None
    return (
        f"{MAXIMUM_CREATURE_HEIGHT_REASON} Creature height is "
        f"{height:.6f} m; maximum is {max_height:.6f} m."
    )


def _build_model(
    genotype: Genotype,
    definition: task_shared.TaskDefinition,
):
    config = definition.config
    try:
        builder = PhenotypeBuilder(genotype, definition)
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


def _run_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    builder: PhenotypeBuilder,
    config: task_shared.EvaluationConfig,
    policy: task_shared.RolloutPolicy = task_shared.RolloutPolicy(),
    root_body_name: str | None = None,
    apply_controls: bool = True,
):
    actuator_ids = actuator_ids_for_controllers(model, builder.actuator_controllers)
    body_count = max(model.nbody - 1, 0)
    total_volume = _creature_volume(model)
    target_direction = _normalized_target_direction(config.target_direction)
    floor_id = -1
    if policy.track_floor_contact:
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
    upright_error_sum = 0.0
    sample_count = 0
    first_ground_contact_time = None
    distance_measurement_position = None
    if floor_id >= 0 and has_floor_contact(model, data, floor_id):
        first_ground_contact_time = 0.0
        distance_measurement_position = initial_center_of_mass.copy()
    root_body_id = -1
    if root_body_name is not None:
        root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)

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
            and has_nonparent_self_collision(model, data)
        ):
            return task_shared.DISALLOWED_COLLISION_REASON
        if (
            floor_id >= 0
            and first_ground_contact_time is None
            and has_floor_contact(model, data, floor_id)
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
        if root_body_id >= 0:
            upright_error_sum += max(0.0, 1.0 - float(data.xmat[root_body_id, 8]))
        sample_count += 1

    final_center_of_mass = _creature_center_of_mass(model, data)
    if distance_measurement_position is None:
        distance_measurement_position = final_center_of_mass
    displacement = distance_measurement_position - initial_center_of_mass
    if policy.horizontal_origin_distance:
        displacement = displacement.copy()
        displacement[2] = 0.0
    forward_distance = float(displacement @ target_direction)
    simulated_seconds = max(float(data.time), model.opt.timestep)
    measurement_seconds = simulated_seconds
    if first_ground_contact_time is not None:
        measurement_seconds = max(first_ground_contact_time, model.opt.timestep)
    average_forward_speed = forward_distance / measurement_seconds
    origin_distance = float(np.linalg.norm(displacement))
    average_origin_speed = origin_distance / measurement_seconds
    lateral = displacement - forward_distance * np.asarray(target_direction)

    metrics = {
        "origin_distance": origin_distance,
        "average_origin_speed": average_origin_speed,
        "forward_distance": forward_distance,
        "average_forward_speed": average_forward_speed,
        "sideways_drift_speed": abs(float(lateral[1])) / measurement_seconds,
        "control_energy": control_energy,
        "mean_angular_speed": angular_speed_sum / max(sample_count, 1),
        "simulated_seconds": simulated_seconds,
        "actuator_count": len(actuator_ids),
        "body_count": body_count,
        "total_volume": total_volume,
    }
    if policy.track_floor_contact:
        ground_touch_penalty = 0.0
        no_ground_touch_bonus = config.no_ground_touch_bonus
        if first_ground_contact_time is not None:
            touch_fraction = min(
                first_ground_contact_time / config.episode_seconds, 1.0
            )
            ground_touch_penalty = config.ground_touch_weight * (1.0 - touch_fraction)
            no_ground_touch_bonus = 0.0
        metrics.update(
            height_loss=max(
                0.0,
                float(initial_center_of_mass[2] - final_center_of_mass[2]),
            ),
            first_ground_contact_time=first_ground_contact_time,
            ground_touch_penalty=ground_touch_penalty,
            no_ground_touch_bonus=no_ground_touch_bonus,
        )
    else:
        metrics["vertical_drift_speed"] = abs(
            float(final_center_of_mass[2] - initial_center_of_mass[2])
        ) / simulated_seconds
    if root_body_id >= 0:
        min_center_height = (
            initial_center_of_mass[2]
            * config.min_center_height_fraction
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


def _creature_center_of_mass(
    model: mujoco.MjModel, data: mujoco.MjData
) -> np.ndarray:
    body_ids = np.arange(1, model.nbody)
    masses = model.body_mass[body_ids]
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        return data.qpos[:3].copy()
    return (data.xipos[body_ids] * masses[:, None]).sum(axis=0) / total_mass


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
    config: task_shared.EvaluationConfig,
) -> str | None:
    min_body_volume = config.min_body_volume
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

    min_total_volume = config.min_total_volume
    total_volume = sum(body_volumes.values())
    if total_volume < min_total_volume:
        return (
            f"{MINIMUM_TOTAL_VOLUME_REASON} Creature has volume "
            f"{total_volume:.6g} m^3; minimum is {min_total_volume:.6g} m^3."
        )
    return None
