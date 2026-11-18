"""``evc3d mutate``: write a randomly mutated copy of a genotype."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
from typing import Sequence

from ..genotype_io import load_genotype_from_json, save_genotype_to_json
from .args import (
    HelpFormatter,
    add_genotype_argument,
    add_mutation_arguments,
    mutation_options,
    validate_mutation_arguments,
)

NAME = "mutate"


def add_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        NAME,
        help="Write a randomly mutated copy of a genotype.",
        description=(
            "Apply random mutations to a genotype and save the result as a new "
            "genotype JSON file."
        ),
        formatter_class=HelpFormatter,
    )
    add_genotype_argument(parser, "Source genotype JSON path.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Mutated genotype JSON output path; must differ from --genotype.",
    )
    parser.add_argument(
        "--mutations",
        type=int,
        default=1,
        help="Number of random mutations to apply, one at a time.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible mutations. (default: random)",
    )
    add_mutation_arguments(parser)
    return parser


def validate(args: argparse.Namespace) -> None:
    if args.mutations < 0:
        raise ValueError("--mutations must be non-negative")
    if args.output.resolve() == args.genotype.resolve():
        raise ValueError("--output must differ from --genotype")
    validate_mutation_arguments(args)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse ``evc3d mutate`` options (``argv`` excludes the command name)."""
    from . import parse_args as parse_cli_args

    return parse_cli_args([NAME, *(sys.argv[1:] if argv is None else argv)])


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


def run(args: argparse.Namespace) -> None:
    genotype = load_genotype_from_json(args.genotype)
    rng = random.Random(args.seed)
    options = mutation_options(args)
    for _ in range(args.mutations):
        genotype.mutation(num_mutations=1, rng=rng, **options)
    save_genotype_to_json(genotype, args.output)
    print(f"Mutated genotype saved to {args.output}")
