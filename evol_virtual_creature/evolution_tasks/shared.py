"""Shared constants and rollout helpers for built-in tasks."""

from __future__ import annotations

import math
from typing import Callable

from .. import evaluation as evaluation_engine
from ..evaluation import ResultField
from ..genotype import Genotype


DEFAULT_MIN_BODY_VOLUME = 1e-6
DEFAULT_MIN_TOTAL_VOLUME = 0.0
DEFAULT_FLYING_MIN_TOTAL_VOLUME = 1e-4

SWIMMING_RESULT_FIELDS = (ResultField("Vertical drift speed", "vertical_drift_speed"),)
WALKING_RESULT_FIELDS = (
    ResultField("Height loss", "height_loss"),
    ResultField("Mean upright error", "mean_upright_error"),
)
FLYING_RESULT_FIELDS = (
    ResultField("Height loss", "height_loss"),
    ResultField("First ground contact", "first_ground_contact_time", ".2f", "none"),
    ResultField("Ground touch penalty", "ground_touch_penalty"),
    ResultField("No-ground-touch bonus", "no_ground_touch_bonus"),
    ResultField("Controlled fitness", "controlled_fitness"),
    ResultField("Passive fitness", "passive_fitness"),
    ResultField("Fitness gain", "fitness_gain"),
)


def evaluate_walking(
    genotype: Genotype, config, *, away: bool, result_type: Callable
):
    built = evaluation_engine._build_model(genotype, config)
    if isinstance(built, str):
        return failed_walking(config, built, result_type)
    model, data, builder = built
    for initialize in (
        evaluation_engine.initialize_walking_model,
        evaluation_engine._walking_height_failure_reason,
        evaluation_engine.settle_walking_model,
    ):
        failure = (
            initialize(model, data)
            if initialize is evaluation_engine.initialize_walking_model
            else initialize(model, data, config)
        )
        if failure is not None:
            return failed_walking(config, failure, result_type)
    data.time = 0.0
    metrics = evaluation_engine._run_controlled_episode(
        model,
        data,
        builder,
        config,
        root_body_name=f"{genotype.root}_1",
        horizontal_origin_distance=away,
    )
    if isinstance(metrics, str):
        return failed_walking(config, metrics, result_type)
    metrics = {key: value for key, value in metrics.items() if key != "vertical_drift_speed"}
    progress = (
        config.speed_weight * metrics["average_origin_speed"]
        if away
        else config.forward_speed_weight * metrics["average_forward_speed"]
        - config.sideways_drift_weight * metrics["sideways_drift_speed"]
        - config.upright_weight * metrics["mean_upright_error"]
    )
    fitness = (
        progress
        - config.energy_weight * metrics["control_energy"]
        - config.angular_speed_weight * metrics["mean_angular_speed"]
        - config.height_loss_weight * metrics["height_loss"]
        - config.body_count_weight * metrics["body_count"]
        - config.volume_weight
        * evaluation_engine._excess_volume(metrics["total_volume"], config.volume_penalty_cutoff)
    )
    if not math.isfinite(fitness):
        return failed_walking(config, "Simulation produced a non-finite fitness.", result_type)
    return result_type(fitness=fitness, **metrics)


def evaluate_flying(genotype: Genotype, config, speed_metric: str, result_type: Callable):
    built = evaluation_engine._build_model(genotype, config)
    if isinstance(built, str):
        return failed_flying(config, built, result_type)
    model, data, builder = built
    failure = evaluation_engine.initialize_flying_model(model, data)
    if failure is not None:
        return failed_flying(config, failure, result_type)
    controlled = evaluation_engine._run_flying_episode(
        model, evaluation_engine._copy_simulation_state(model, data), builder, config,
        apply_controls=True,
    )
    if isinstance(controlled, str):
        return failed_flying(config, controlled, result_type)
    passive = evaluation_engine._run_flying_episode(
        model, evaluation_engine._copy_simulation_state(model, data), builder, config,
        apply_controls=False,
    )
    if isinstance(passive, str):
        return failed_flying(config, passive, result_type)
    controlled_fitness = flying_fitness(config, controlled, speed_metric)
    passive_fitness = flying_fitness(config, passive, speed_metric)
    gain = controlled_fitness - passive_fitness
    fitness = gain * config.fitness_gain_fraction + controlled_fitness * (
        1.0 - config.fitness_gain_fraction
    )
    controlled.update(
        controlled_fitness=controlled_fitness,
        passive_fitness=passive_fitness,
        fitness_gain=gain,
    )
    if not math.isfinite(fitness):
        return failed_flying(config, "Simulation produced a non-finite fitness.", result_type)
    return result_type(fitness=fitness, **controlled)


def flying_fitness(config, metrics: dict, speed_metric: str) -> float:
    speed_weight = config.speed_weight if hasattr(config, "speed_weight") else config.distance_weight
    return (
        speed_weight * metrics[speed_metric]
        + metrics["no_ground_touch_bonus"]
        - config.height_loss_weight * metrics["height_loss"]
        - metrics["ground_touch_penalty"]
        - config.energy_weight * metrics["control_energy"]
        - config.angular_speed_weight * metrics["mean_angular_speed"]
        - config.body_count_weight * metrics["body_count"]
        - config.volume_weight
        * evaluation_engine._excess_volume(metrics["total_volume"], config.volume_penalty_cutoff)
    )


def failure_flags(reason: str) -> dict:
    collision = reason == evaluation_engine.DISALLOWED_COLLISION_REASON
    return dict(build_failed=not collision, disqualified=collision, failure_reason=reason)


def failed_swimming(config, reason: str, result_type: Callable):
    return result_type(
        fitness=config.build_failure_fitness, origin_distance=0.0, average_origin_speed=0.0,
        forward_distance=0.0, average_forward_speed=0.0, sideways_drift_speed=0.0,
        vertical_drift_speed=0.0, control_energy=0.0, mean_angular_speed=0.0,
        simulated_seconds=0.0, actuator_count=0, body_count=0, total_volume=0.0,
        **failure_flags(reason),
    )


def failed_walking(config, reason: str, result_type: Callable):
    return result_type(
        fitness=config.build_failure_fitness, origin_distance=0.0, average_origin_speed=0.0,
        forward_distance=0.0, average_forward_speed=0.0, sideways_drift_speed=0.0,
        height_loss=0.0, control_energy=0.0, mean_angular_speed=0.0,
        mean_upright_error=0.0, simulated_seconds=0.0, actuator_count=0,
        body_count=0, total_volume=0.0, **failure_flags(reason),
    )


def failed_flying(config, reason: str, result_type: Callable):
    return result_type(
        fitness=config.build_failure_fitness, origin_distance=0.0, average_origin_speed=0.0,
        forward_distance=0.0, average_forward_speed=0.0, sideways_drift_speed=0.0,
        height_loss=0.0, first_ground_contact_time=None, ground_touch_penalty=0.0,
        no_ground_touch_bonus=0.0, controlled_fitness=0.0, passive_fitness=0.0,
        fitness_gain=0.0, control_energy=0.0, mean_angular_speed=0.0,
        simulated_seconds=0.0, actuator_count=0, body_count=0, total_volume=0.0,
        **failure_flags(reason),
    )
