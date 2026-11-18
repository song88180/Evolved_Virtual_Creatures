"""``evc3d evaluate``: score one genotype on a locomotion task."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from ..evaluation import DEFAULT_UPRIGHT_ERROR_WEIGHT, evaluate_task
from ..genotype_io import load_genotype_from_json
from .args import (
    HelpFormatter,
    add_environment_arguments,
    add_genotype_argument,
    add_scoring_arguments,
    resolve_task_definition,
    validate_environment_arguments,
    validate_scoring_arguments,
)

NAME = "evaluate"

__all__ = ["DEFAULT_UPRIGHT_ERROR_WEIGHT", "parse_args", "run"]


def add_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        NAME,
        help="Score one genotype on a task.",
        description=(
            "Evaluate a creature on swimming, walking, or flying task variants."
        ),
        formatter_class=HelpFormatter,
    )
    add_genotype_argument(parser, "Genotype JSON path.")
    add_environment_arguments(parser, "Locomotion task to evaluate.")
    add_scoring_arguments(parser, "Measured episode duration in seconds.")
    return parser


def validate(args: argparse.Namespace) -> None:
    validate_environment_arguments(args)
    validate_scoring_arguments(args)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse ``evc3d evaluate`` options (``argv`` excludes the command name)."""
    from . import parse_args as parse_cli_args

    return parse_cli_args([NAME, *(sys.argv[1:] if argv is None else argv)])


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


def run(args: argparse.Namespace) -> None:
    definition = resolve_task_definition(args)
    genotype = load_genotype_from_json(args.genotype)
    result = evaluate_task(genotype, definition)

    if result.disqualified:
        print(f"Disqualified: {result.failure_reason}")
        print(f"Fitness: {result.fitness:.6f}")
        return

    if result.build_failed:
        print(f"Build failed: {result.failure_reason}")
        print(f"Fitness: {result.fitness:.6f}")
        return

    print(definition.title)
    print(f"Genotype: {args.genotype}")
    print(f"Fitness: {result.fitness:.6f}")
    print(f"Origin distance: {result.origin_distance:.6f}")
    print(f"Average origin speed: {result.average_origin_speed:.6f}")
    print(f"Forward distance: {result.forward_distance:.6f}")
    print(f"Average forward speed: {result.average_forward_speed:.6f}")
    print(f"Sideways drift speed: {result.sideways_drift_speed:.6f}")
    for result_field in definition.result_fields:
        value = getattr(result, result_field.attribute)
        rendered = (
            result_field.none_text
            if value is None and result_field.none_text is not None
            else format(value, result_field.format_spec)
        )
        print(f"{result_field.label}: {rendered}")
    print(f"Control energy: {result.control_energy:.6f}")
    print(f"Mean angular speed: {result.mean_angular_speed:.6f}")
    print(f"Simulated seconds: {result.simulated_seconds:.2f}")
    print(f"Actuators: {result.actuator_count}")
    print(f"Total volume: {result.total_volume:.6f}")
    print(f"Bodies: {result.body_count}")

