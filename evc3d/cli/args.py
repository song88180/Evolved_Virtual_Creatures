"""Argument groups shared by ``evc3d`` subcommands.

Each ``add_*_arguments`` function adds one named group to a subcommand parser,
and the matching ``validate_*`` function checks it after parsing. Validators
raise ``ValueError``; ``evc3d.cli.parse_args`` reports those as usage errors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..evolution_tasks import task_definition, task_names
from ..evolution_tasks.shared import TaskDefinition
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


def validate_environment_arguments(args: argparse.Namespace) -> None:
    if args.max_body_count < 1:
        raise ValueError("--max-body-count must be at least 1")
    if args.fluid_density is not None and args.fluid_density < 0.0:
        raise ValueError("--fluid-density must be non-negative")
    if args.fluid_viscosity is not None and args.fluid_viscosity < 0.0:
        raise ValueError("--fluid-viscosity must be non-negative")


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


def validate_scoring_arguments(args: argparse.Namespace) -> None:
    if args.duration <= 0.0:
        raise ValueError("--duration must be greater than zero")
    if args.volume_penalty_cutoff is not None and args.volume_penalty_cutoff < 0.0:
        raise ValueError("--volume-penalty-cutoff must be non-negative")
    if (
        args.max_volume is not None
        and args.volume_penalty_cutoff is not None
        and args.max_volume <= args.volume_penalty_cutoff
    ):
        raise ValueError("--max-volume must be greater than --volume-penalty-cutoff")
    if args.min_body_volume is not None and args.min_body_volume < 0.0:
        raise ValueError("--min-body-volume must be non-negative")
    if args.min_total_volume is not None and args.min_total_volume < 0.0:
        raise ValueError("--min-total-volume must be non-negative")
    if args.volume_weight is not None and args.volume_weight < 0.0:
        raise ValueError("--volume-weight must be non-negative")
    if args.body_count_weight is not None and args.body_count_weight < 0.0:
        raise ValueError("--body-count-weight must be non-negative")
    if (
        args.fitness_gain_fraction is not None
        and not 0.0 <= args.fitness_gain_fraction <= 1.0
    ):
        raise ValueError("--fitness-gain-fraction must be between 0 and 1")


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
    """Return the task definition with CLI environment/scoring overrides applied."""
    definition = task_definition(args.task).args_override(args)
    config = definition.config
    if config.max_volume <= config.volume_penalty_cutoff:
        raise ValueError("--max-volume must be greater than --volume-penalty-cutoff")
    return definition
