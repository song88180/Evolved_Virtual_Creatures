"""``evc3d build``: write the MuJoCo XML scene for a genotype."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from ..genotype_io import load_genotype_from_json
from ..graph_analysis import PhenotypeBuildAbort
from ..phenotype import PhenotypeBuilder
from .args import (
    HelpFormatter,
    add_environment_arguments,
    add_genotype_argument,
    resolve_task_definition,
    validate_environment_arguments,
)

NAME = "build"
DEFAULT_OUTPUT_PATH = Path("generated_creature.xml")


def add_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        NAME,
        help="Write the MuJoCo XML for a genotype.",
        description=(
            "Build the MuJoCo MJCF scene for a genotype. The XML contains the "
            "creature and the task's physical environment (gravity, fluid, "
            "floor); it does not contain the controller."
        ),
        formatter_class=HelpFormatter,
    )
    add_genotype_argument(parser, "Source genotype JSON path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Generated MuJoCo XML output path.",
    )
    add_environment_arguments(
        parser, "Task whose physical environment is written into the XML."
    )
    return parser


def validate(args: argparse.Namespace) -> None:
    validate_environment_arguments(args)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse ``evc3d build`` options (``argv`` excludes the command name)."""
    from . import parse_args as parse_cli_args

    return parse_cli_args([NAME, *(sys.argv[1:] if argv is None else argv)])


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


def run(args: argparse.Namespace) -> None:
    genotype = load_genotype_from_json(args.genotype)
    definition = resolve_task_definition(args)
    builder = PhenotypeBuilder(genotype, definition)
    try:
        mjcf = builder.build()
    except PhenotypeBuildAbort as error:
        print(f"Aborting phenotype construction: {error}")
        raise SystemExit(1) from error
    args.output.write_text(mjcf)
    print(f"Generated MuJoCo model saved to {args.output}")
