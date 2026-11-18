"""The ``evc3d`` command-line interface.

Usage::

    evc3d evolve   [options]   # run mutation-based evolution
    evc3d evaluate [options]   # score one genotype on a task
    evc3d mutate   [options]   # write a randomly mutated copy of a genotype
    evc3d build    [options]   # write the MuJoCo XML for a genotype
    evc3d view     [options]   # watch a genotype live, or record it with --mp4

Each subcommand lives in its own module (``evc3d.cli.<command>``) and exposes
``add_parser``, ``validate`` and ``run``. Shared option groups are in
``evc3d.cli.args``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

# Keys set with ``set_defaults`` for dispatch; not user options.
INTERNAL_KEYS = ("command", "_run", "_validate", "_parser")


def _configure_rendering_backend(argv: Sequence[str]) -> None:
    """Use headless EGL rendering on Linux when an MP4 is requested.

    MuJoCo reads ``MUJOCO_GL`` when it is first imported, so this must run
    before any subcommand module is imported.
    """
    video_requested = any(
        argument == "--mp4" or argument.startswith("--mp4=")
        for argument in argv
    )
    if sys.platform.startswith("linux") and video_requested:
        os.environ.setdefault("MUJOCO_GL", "egl")


def build_parser() -> argparse.ArgumentParser:
    from . import build, evaluate, evolve, mutate, view

    parser = argparse.ArgumentParser(
        prog="evc3d",
        description="Evolve, evaluate, mutate, build, and view 3D virtual creatures.",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="COMMAND",
    )
    for module in (evolve, evaluate, mutate, build, view):
        subparser = module.add_parser(subparsers)
        subparser.set_defaults(
            _run=module.run,
            _validate=module.validate,
            _parser=subparser,
        )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate ``evc3d`` arguments; invalid values exit with usage."""
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    try:
        args._validate(args)
    except ValueError as error:
        args._parser.error(str(error))
    return args


def user_options(args: argparse.Namespace) -> dict:
    """Return parsed options without the internal dispatch entries."""
    return {
        key: value for key, value in vars(args).items() if key not in INTERNAL_KEYS
    }


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    _configure_rendering_backend(argv)
    args = parse_args(argv)
    args._run(args)
