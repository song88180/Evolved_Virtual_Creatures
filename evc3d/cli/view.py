"""``evc3d view``: watch a genotype perform its task, live or as an MP4."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from ..genotype_io import load_genotype_from_json
from .args import (
    HelpFormatter,
    add_environment_arguments,
    add_genotype_argument,
    resolve_task_definition,
    validate_task_options,
)

NAME = "view"
# RK4 with 4 substeps (dt=0.0025) was the cheapest verification setting that
# matched dt=0.0001 physics; view uses the same step count by default.
DEFAULT_VIEW_PHYSICS_SUBSTEPS = 4
_DEFAULT_MP4_SENTINEL = Path("__task_default_mp4__")

# MP4-only options and their defaults; given without --mp4 they are an error.
MP4_DEFAULTS = {
    "fps": 30,
    "width": 960,
    "height": 544,
    "episode_seconds": 10.0,
}


def add_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        NAME,
        help="Watch a genotype perform its task (window or MP4).",
        description=(
            "Simulate a genotype in its task environment, starting exactly like "
            "the scored rollout. By default an interactive MuJoCo window opens "
            "and runs until it is closed; with --mp4 the episode is recorded "
            "to a video file instead."
        ),
        formatter_class=HelpFormatter,
    )
    add_genotype_argument(parser, "Genotype JSON path.")
    # Watch creatures with the finer physics evolve --verify-top scores them
    # with, so integration-error motion is not mistaken for real swimming.
    add_environment_arguments(
        parser,
        "Task whose environment the creature is shown in.",
        physics_substeps_default=DEFAULT_VIEW_PHYSICS_SUBSTEPS,
    )

    group = parser.add_argument_group("display")
    group.add_argument(
        "--track-root",
        action="store_true",
        help="Keep the camera centered on the creature's root body.",
    )
    group.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help=(
            "Playback speed multiplier relative to simulated time "
            "(for example, 1.25 or 0.8)."
        ),
    )
    group.add_argument(
        "--shadowsize",
        type=int,
        default=4096,
        help="Shadow-map texture width and height in pixels.",
    )
    group.add_argument(
        "--spotlight",
        action="store_true",
        help="Use a shadow-casting spotlight instead of a directional light.",
    )

    group = parser.add_argument_group("mp4 (only with --mp4)")
    group.add_argument(
        "--mp4",
        nargs="?",
        const=_DEFAULT_MP4_SENTINEL,
        type=Path,
        help=(
            "Record an MP4 instead of opening a window; without a path uses "
            "example_<task>.mp4. (default: interactive window)"
        ),
    )
    group.add_argument(
        "--episode-seconds",
        type=float,
        default=None,
        help="Recorded episode length in simulated seconds. (default: 10.0)",
    )
    group.add_argument(
        "--fps", type=int, default=None, help="Video frames per second. (default: 30)"
    )
    group.add_argument(
        "--width", type=int, default=None, help="Video width in pixels. (default: 960)"
    )
    group.add_argument(
        "--height", type=int, default=None, help="Video height in pixels. (default: 544)"
    )
    group.add_argument(
        "--camera-circle-around",
        action="store_true",
        help="Circle the camera 360 degrees around the creature during the episode.",
    )
    return parser


def validate(args: argparse.Namespace) -> None:
    if args.speed <= 0.0:
        raise ValueError("--speed must be greater than zero")
    if args.shadowsize < 1:
        raise ValueError("--shadowsize must be greater than zero")

    if args.mp4 is None:
        given = [
            "--" + name.replace("_", "-")
            for name in MP4_DEFAULTS
            if getattr(args, name) is not None
        ]
        if args.camera_circle_around:
            given.append("--camera-circle-around")
        if given:
            verb = "applies" if len(given) == 1 else "apply"
            raise ValueError(f"{', '.join(given)} only {verb} with --mp4")
        validate_task_options(args)
        return

    for name, default in MP4_DEFAULTS.items():
        if getattr(args, name) is None:
            setattr(args, name, default)
    if args.fps < 1:
        raise ValueError("--fps must be at least 1")
    if args.width < 1 or args.height < 1:
        raise ValueError("--width and --height must be at least 1")
    validate_task_options(args)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse ``evc3d view`` options (``argv`` excludes the command name)."""
    from . import parse_args as parse_cli_args

    return parse_cli_args([NAME, *(sys.argv[1:] if argv is None else argv)])


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


def run(args: argparse.Namespace) -> None:
    from ..scene import SceneBuildError, SceneInitializationError

    genotype = load_genotype_from_json(args.genotype)
    definition = resolve_task_definition(args)

    if args.mp4 is None:
        from ..viewer import launch_viewer

        try:
            launch_viewer(
                genotype,
                definition,
                track_root=args.track_root,
                speed=args.speed,
                shadowsize=args.shadowsize,
                spotlight=args.spotlight,
            )
        except (SceneBuildError, SceneInitializationError) as error:
            print(f"Cannot view creature: {error}")
            raise SystemExit(1) from error
        return

    from ..video import save_x_axis_video

    output_path = args.mp4
    if output_path == _DEFAULT_MP4_SENTINEL:
        output_path = Path(f"example_{args.task}.mp4")
    try:
        save_x_axis_video(
            genotype=genotype,
            output_path=output_path,
            definition=definition,
            fps=args.fps,
            width=args.width,
            height=args.height,
            track_root=args.track_root,
            speed=args.speed,
            shadowsize=args.shadowsize,
            spotlight=args.spotlight,
            camera_circle_around=args.camera_circle_around,
        )
    except RuntimeError as error:
        print(error)
        raise SystemExit(1) from error
    print(f"Saved video: {output_path}")
