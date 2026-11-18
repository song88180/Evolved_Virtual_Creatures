"""Building blocks shared by the paired task modules.

Each locomotion family (swimming, walking, flying) has an ``_x`` task that
rewards progress along +X and an ``_away`` task that rewards speed away from the
start. The family's environment, config fields, and result fields are defined
once here; a task module adds only what differs: its speed reward and its name.

Fitness expressions stay explicit in each task module on purpose: they are the
part that defines a task, and keeping their term order fixed keeps scores
comparable across code versions.

Task modules must not import ``evc3d.evaluation`` (the engine imports them).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .config import (
    DEFAULT_FLYING_MIN_TOTAL_VOLUME,
    EnvironmentFamily,
    EvaluationConfig,
)
from .shared import EvaluationResult, excess_volume, make_result


# --- Environments ---------------------------------------------------------

def walking_environment(name: str) -> EnvironmentFamily:
    """Gravity and a contact floor; no fluid."""
    return EnvironmentFamily(
        name=name,
        gravity=-9.81,
        fluid_density=None,
        fluid_viscosity=None,
        body_friction=(1.0, 0.005, 0.0001),
        creature_contype=2,
        creature_conaffinity=1,
        self_collision_conaffinity=3,
        floor_contype=1,
        floor_conaffinity=2,
        initial_floor_clearance=0.05,
    )


def flying_environment(name: str) -> EnvironmentFamily:
    """Gravity and air drag, starting well above a contact floor."""
    return EnvironmentFamily(
        name=name,
        gravity=-9.81,
        fluid_density=1.225,
        fluid_viscosity=1.8e-5,
        fluid_shape="ellipsoid",
        fluid_coef=(0.5, 0.25, 1.5, 1.0, 1.0),
        body_friction=(1.0, 0.005, 0.0001),
        creature_contype=2,
        creature_conaffinity=1,
        self_collision_conaffinity=3,
        floor_contype=1,
        floor_conaffinity=2,
        initial_floor_clearance=5.0,
        supports_scheduled_gravity=True,
    )


# --- Result bases ---------------------------------------------------------
# Failure fields (build_failed, disqualified, failure_reason) default so a
# result can be built with metrics only; ``failed_result`` fills them in.

@dataclass(frozen=True)
class SwimmingResult(EvaluationResult):
    vertical_drift_speed: float
    build_failed: bool = False
    disqualified: bool = False
    failure_reason: str | None = None


@dataclass(frozen=True)
class WalkingResult(EvaluationResult):
    height_loss: float
    mean_upright_error: float
    build_failed: bool = False
    disqualified: bool = False
    failure_reason: str | None = None


@dataclass(frozen=True)
class FlyingResult(EvaluationResult):
    height_loss: float
    first_ground_contact_time: float | None
    ground_touch_penalty: float
    no_ground_touch_bonus: float
    controlled_fitness: float
    passive_fitness: float
    fitness_gain: float
    build_failed: bool = False
    disqualified: bool = False
    failure_reason: str | None = None


# --- Config bases ---------------------------------------------------------

@dataclass(frozen=True)
class WalkingConfig(EvaluationConfig):
    initial_floor_contact_policy: str = "no_penetration"
    settle_seconds: float = 1.0
    energy_weight: float = 1e-7
    angular_speed_weight: float = 0.01
    height_loss_weight: float = 0.2
    min_center_height_fraction: float = 0.5
    max_creature_height: float = 10.0


@dataclass(frozen=True)
class FlyingConfig(EvaluationConfig):
    initial_floor_contact_policy: str = "no_contact"
    energy_weight: float = 1e-7
    angular_speed_weight: float = 0.01
    height_loss_weight: float = 0.5
    ground_touch_weight: float = 0.0
    no_ground_touch_bonus: float = 10
    fitness_gain_fraction: float = 0.5
    min_total_volume: float = DEFAULT_FLYING_MIN_TOTAL_VOLUME

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.fitness_gain_fraction <= 1.0:
            raise ValueError("fitness_gain_fraction must be between 0 and 1")


# --- Flying scoring -------------------------------------------------------

def ground_touch_terms(
    config: FlyingConfig, first_ground_contact_time: float | None
) -> tuple[float, float]:
    """Return ``(ground_touch_penalty, no_ground_touch_bonus)`` for one episode.

    Never touching the floor earns the full bonus; touching it costs a penalty
    that shrinks the later the first contact happens.
    """
    if first_ground_contact_time is None:
        return 0.0, config.no_ground_touch_bonus
    touch_fraction = min(first_ground_contact_time / config.episode_seconds, 1.0)
    return config.ground_touch_weight * (1.0 - touch_fraction), 0.0


def flying_episode_fitness(
    config: FlyingConfig, metrics: dict, speed_term: float
) -> float:
    """Score one flying episode; ``speed_term`` is the task's weighted speed."""
    penalty, bonus = ground_touch_terms(config, metrics["first_ground_contact_time"])
    return (
        speed_term
        + bonus
        - config.height_loss_weight * metrics["height_loss"]
        - penalty
        - config.energy_weight * metrics["control_energy"]
        - config.angular_speed_weight * metrics["mean_angular_speed"]
        - config.body_count_weight * metrics["body_count"]
        - config.volume_weight
        * excess_volume(metrics["total_volume"], config.volume_penalty_cutoff)
    )


def blend_flying_fitness(
    config: FlyingConfig,
    controlled_metrics: dict,
    passive_metrics: dict | None,
    episode_fitness: Callable[[FlyingConfig, dict], float],
    result_type: type,
):
    """Blend the controlled score with its gain over the passive baseline."""
    assert passive_metrics is not None
    controlled_fitness = episode_fitness(config, controlled_metrics)
    passive_fitness = episode_fitness(config, passive_metrics)
    gain = controlled_fitness - passive_fitness
    fitness = (
        gain * config.fitness_gain_fraction
        + controlled_fitness * (1.0 - config.fitness_gain_fraction)
    )
    penalty, bonus = ground_touch_terms(
        config, controlled_metrics["first_ground_contact_time"]
    )
    metrics = dict(
        controlled_metrics,
        ground_touch_penalty=penalty,
        no_ground_touch_bonus=bonus,
        controlled_fitness=controlled_fitness,
        passive_fitness=passive_fitness,
        fitness_gain=gain,
    )
    return make_result(result_type, fitness, metrics)
