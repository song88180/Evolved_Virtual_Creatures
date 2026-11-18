"""Argument groups shared by ``evc3d`` subcommands.

Each ``add_*_arguments`` function adds one named group to a subcommand parser.
Task options (environment and scoring) are validated by the task config and
environment themselves, through ``validate_task_options``; other groups have a
``validate_*`` function. Validators raise ``ValueError``, which
``evc3d.cli.parse_args`` reports as a usage error.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import re

from ..evolution_tasks import task_definition, task_names
from ..evolution_tasks.shared import (
    CONFIG_FIELD_OPTIONS,
    ENVIRONMENT_FIELD_OPTIONS,
    TaskDefinition,
)
from ..genotype import Genotype


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GENOTYPE_PATH = REPO_ROOT / "examples" / "example_genotype.json"
DEFAULT_TASK = "swimming_x"
DEFAULT_MAX_BODY_COUNT = 500


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Append concrete defaults unless the help text already states one."""

    def _get_help_string(self, action):
        if action.default is None or action.default is argparse.SUPPRESS:
            return action.help
        if action.help and "(default:" in action.help:
            return action.help
        return super()._get_help_string(action)


def add_genotype_argument(parser: argparse.ArgumentParser, help: str) -> None:
    parser.add_argument(
        "--genotype",
        type=Path,
        default=DEFAULT_GENOTYPE_PATH,
        help=f"{help} (default: examples/example_genotype.json)",
    )


def add_environment_arguments(
    parser: argparse.ArgumentParser,
    task_help: str,
) -> None:
    """Options that define the simulated world and the generated phenotype."""
    group = parser.add_argument_group("environment")
    group.add_argument(
        "--task",
        choices=task_names(),
        default=DEFAULT_TASK,
        help=task_help,
    )
    group.add_argument(
        "--max-body-count",
        type=int,
        default=DEFAULT_MAX_BODY_COUNT,
        help="Maximum number of generated bodies (after recursion and symmetry).",
    )
    group.add_argument(
        "--fluid-density",
        type=float,
        default=None,
        help="Fluid density in kg/m^3. (default: task environment default)",
    )
    group.add_argument(
        "--fluid-viscosity",
        type=float,
        default=None,
        help="Fluid viscosity in Pa*s. (default: task environment default)",
    )
    group.add_argument(
        "--fluid-shape",
        choices=("none", "ellipsoid"),
        default=None,
        help=(
            "MuJoCo fluid shape approximation for creature geoms. "
            "(default: task environment default)"
        ),
    )
    group.add_argument(
        "--fluid-coef",
        type=float,
        nargs=5,
        default=None,
        metavar=("BLUNT", "SLENDER", "ANGULAR", "KUTTA", "MAGNUS"),
        help=(
            "Five MuJoCo fluid coefficients used by creature geoms. "
            "(default: task environment default)"
        ),
    )
    group.add_argument(
        "--self-collision",
        action="store_true",
        help=(
            "Enable collisions between non-parent creature bodies; direct "
            "parent-child collisions remain filtered."
        ),
    )


def add_scoring_arguments(
    parser: argparse.ArgumentParser,
    duration_help: str,
) -> None:
    """Options that only affect how an episode is scored."""
    group = parser.add_argument_group("scoring")
    group.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help=duration_help,
    )
    group.add_argument(
        "--body-count-weight",
        type=float,
        default=None,
        help="Fitness penalty per generated body. (default: task default)",
    )
    group.add_argument(
        "--volume-weight",
        type=float,
        default=None,
        help=(
            "Fitness penalty per cubic meter of generated creature volume. "
            "(default: task default)"
        ),
    )
    group.add_argument(
        "--volume-penalty-cutoff",
        type=float,
        default=None,
        help=(
            "Creature volume in cubic meters allowed without a fitness penalty. "
            "(default: task default)"
        ),
    )
    group.add_argument(
        "--min-body-volume",
        type=float,
        default=None,
        help=(
            "Minimum required volume for each generated body section in cubic meters. "
            "(default: task default)"
        ),
    )
    group.add_argument(
        "--min-total-volume",
        type=float,
        default=None,
        help=(
            "Minimum required total generated creature volume in cubic meters. "
            "(default: task default)"
        ),
    )
    group.add_argument(
        "--max-volume",
        type=float,
        default=None,
        help=(
            "Maximum allowed generated creature volume in cubic meters. "
            "(default: task default)"
        ),
    )
    group.add_argument(
        "--fitness-gain-fraction",
        type=float,
        default=None,
        help=(
            "Blend fraction for controlled-minus-passive flying fitness gain. "
            "(default: task default)"
        ),
    )
    group.add_argument(
        "--upright-error",
        action="store_true",
        help="Enable the root-upright fitness penalty for walking_x tasks.",
    )
    group.add_argument(
        "--disallow-collision",
        action="store_true",
        help=(
            "Assign low fitness if any non-parent creature bodies collide; "
            "self-collision detection is enabled automatically."
        ),
    )


def add_mutation_arguments(parser: argparse.ArgumentParser) -> None:
    """Options that restrict which genotype mutations may be applied."""
    group = parser.add_argument_group("mutation")
    group.add_argument(
        "--topology-mutation-rate-min",
        type=float,
        default=Genotype.DEFAULT_TOPOLOGY_MUTATION_RATE_MIN,
        help=(
            "Minimum independent mutation rate for topology-changing mutation "
            "operators."
        ),
    )
    group.add_argument(
        "--disallow-topology-mutations",
        action="store_true",
        help=(
            "Only mutate properties of existing node and connection genes; "
            "do not add, remove, replace, or relink genes."
        ),
    )
    group.add_argument(
        "--disallow-root-mutation",
        action="store_true",
        help="Do not mutate the root node gene.",
    )
    group.add_argument(
        "--allow-slide-joint",
        action="store_true",
        help="Allow mutations to create slide-jointed body parts.",
    )


def validate_mutation_arguments(args: argparse.Namespace) -> None:
    if not 0.0 <= args.topology_mutation_rate_min <= 1.0:
        raise ValueError("--topology-mutation-rate-min must be between 0 and 1")


def mutation_options(args: argparse.Namespace) -> dict:
    """Keyword arguments for ``Genotype.mutation`` from the mutation group."""
    return {
        "allow_topology_mutations": not args.disallow_topology_mutations,
        "allow_slide_joint": args.allow_slide_joint,
        "allow_root_mutation": not args.disallow_root_mutation,
        "topology_mutation_rate_min": args.topology_mutation_rate_min,
    }


def resolve_task_definition(args: argparse.Namespace) -> TaskDefinition:
    """Return a private copy of the task definition with CLI overrides applied.

    The registry's definition is shared by everything in the process, so the
    overrides are applied to a copy and never leak into later task lookups.
    """
    return replace(task_definition(args.task)).args_override(args)


def validate_task_options(args: argparse.Namespace) -> None:
    """Build the task definition from parsed options to validate them.

    The task config and environment enforce their own rules; this rewords
    their errors with CLI option names (``episode_seconds`` -> ``--duration``)
    so they can be reported as usage errors.
    """
    try:
        resolve_task_definition(args)
    except ValueError as error:
        raise ValueError(option_message(str(error))) from error


def option_message(message: str) -> str:
    """Replace config and environment field names in ``message`` with CLI options."""
    names = {**CONFIG_FIELD_OPTIONS, **{name: name for name in ENVIRONMENT_FIELD_OPTIONS}}
    for field_name in sorted(names, key=len, reverse=True):
        option = "--" + names[field_name].replace("_", "-")
        message = re.sub(rf"(?<![\w-]){field_name}\b", option, message)
    return message
