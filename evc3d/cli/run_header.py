"""The settings summary ``evc3d evolve`` prints before generation zero."""

from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any

from ..evolution_tasks import task_definition
from ..evolution_tasks.config import EvaluationConfig
from ..evolution_tasks.shared import TaskDefinition

RULE = "=" * 78
LABEL_WIDTH = 21


def format_run_header(
    args: argparse.Namespace,
    definition: TaskDefinition,
    *,
    run_dir: Path,
    title: str,
    seed_control_mode: str,
    loaded_control_mode: str,
    started: datetime,
    worker_count: int,
) -> str:
    """Return the multi-line run summary for the resolved task and options.

    A ``*`` after a value marks one that differs from the task default, so
    command-line overrides stand out.
    """
    default = task_definition(args.task)
    environment, config = definition.environment, definition.config
    default_environment, default_config = default.environment, default.config

    def env(name: str) -> str:
        return _value(getattr(environment, name), getattr(default_environment, name))

    def cfg(name: str) -> str:
        return _value(getattr(config, name), getattr(default_config, name))

    control = seed_control_mode
    if seed_control_mode != loaded_control_mode:
        control += f"     (converted from {loaded_control_mode}; old controller discarded)"

    lines = [
        _title_rule("evc3d evolve"),
        _row("Control mode", control),
        _row("Seed genotype", args.genotype),
        _row("Run directory", run_dir),
        _row("Started", started.strftime("%Y-%m-%d %H:%M:%S")),
        _row("Random seed", args.seed),
        _row("Task", definition.name),
        _row("Title", title),
        "",
        _section("Environment"),
        _row("Gravity (z)", env("gravity")),
        _row("Timestep", f"{env('timestep')} s"),
        _row(
            "Fluid",
            f"density={env('fluid_density')}  viscosity={env('fluid_viscosity')}  "
            f"shape={env('fluid_shape')}  coef={env('fluid_coef')}",
        ),
        _row("Body", f"density={env('body_density')}"),
        _row(
            "Collision bits",
            f"creature={env('creature_contype')}/{env('creature_conaffinity')}  "
            f"self={env('self_collision_contype')}/{env('self_collision_conaffinity')}  "
            f"floor={env('floor_contype')}/{env('floor_conaffinity')}"
            "      (contype/conaffinity)",
        ),
        _row("Initial root pos", env("initial_root_position")),
        _row("Floor clearance", env("initial_floor_clearance")),
    ]
    if args.gradual_gravity_change is not None:
        start, end = args.gradual_gravity_change
        lines.append(_row("Gravity schedule", f"{start} -> {end} over {args.generations} generations"))

    lines += [
        "",
        _section("Scoring"),
        _row("Episode", f"{cfg('episode_seconds')} s   (settle {cfg('settle_seconds')} s)"),
        _row("Max body count", f"{cfg('max_body_count')}        weight {cfg('body_count_weight')}"),
        _row(
            "Volume",
            f"weight {cfg('volume_weight')}  cutoff {cfg('volume_penalty_cutoff')}  "
            f"min/body {cfg('min_body_volume')}  "
            f"total [{cfg('min_total_volume')}, {cfg('max_volume')}]",
        ),
        _row(
            "Self-collision",
            f"{cfg('self_collision')}   disqualify on collision: {cfg('disallow_collision')}",
        ),
        _row("Start floor contact", cfg("initial_floor_contact_policy")),
        _row(
            "Height",
            f"max {cfg('max_creature_height')}   "
            f"min center fraction {cfg('min_center_height_fraction')}",
        ),
    ]
    task_fields = _task_specific_fields(config)
    if task_fields:
        lines.append(
            _row(
                "Task terms",
                "  ".join(
                    f"{name}={_value(getattr(config, name), getattr(default_config, name))}"
                    for name in task_fields
                ),
            )
        )

    lines += [
        "",
        _section("Evolution"),
        _row("Population", f"{args.population_size}        Generations : {args.generations} (+ generation 0)"),
        _row("Elites", f"{args.elite_count}          Tournament  : {args.tournament_size}"),
        _row(
            "Mutations",
            f"{args.min_mutations}-{args.max_mutations} per child      "
            f"Initial : {args.initial_mutations}",
        ),
        _row("Workers", f"{worker_count} processes"),
        _row("Verification", _verification_policy(args)),
        "",
        _section("Output"),
        _row("Checkpoints", _checkpoint_policy(args)),
        _row("Mutant types", "recorded (fitter / neutral / less_fit)" if args.record_mutant_type else "not recorded"),
        RULE,
    ]
    return "\n".join(lines)


def _verification_policy(args: argparse.Namespace) -> str:
    verify_top = getattr(args, "verify_top", 0)
    if not verify_top:
        return "off"
    return (
        f"top {verify_top} re-scored with {args.verify_integrator}, "
        f"{args.verify_substeps} substeps per control step"
    )


def _task_specific_fields(config: Any) -> list[str]:
    """Return config fields the task adds beyond ``EvaluationConfig``."""
    shared = {item.name for item in fields(EvaluationConfig)}
    return [item.name for item in fields(config) if item.name not in shared]


def _checkpoint_policy(args: argparse.Namespace) -> str:
    if args.latest_best_only:
        return "latest best genotype only"
    if args.save_genotype_every_n == 1:
        return "best genotype every generation"
    return f"best genotype every {args.save_genotype_every_n} generations"


def _value(value: Any, default: Any) -> str:
    """Format ``value``, appending ``*`` when it differs from ``default``."""
    return f"{_plain(value)}{' *' if _plain(value) != _plain(default) else ''}"


def _plain(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_plain(item) for item in value)
    return value


def _row(label: str, value: Any) -> str:
    return f"{label:<{LABEL_WIDTH}}: {value}"


def _title_rule(title: str) -> str:
    return f" {title} ".center(len(RULE), "=")


def _section(name: str) -> str:
    return f"--- {name} ".ljust(len(RULE), "-")
