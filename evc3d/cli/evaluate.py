"""``evc3d evaluate``: score one genotype on a locomotion task."""

from __future__ import annotations

import argparse
from pathlib import Path
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
_DEFAULT_VIDEO_SENTINEL = Path("__task_default_video__")

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

    group = parser.add_argument_group("video")
    group.add_argument(
        "--video",
        nargs="?",
        const=_DEFAULT_VIDEO_SENTINEL,
        type=Path,
        help=(
            "Optional MP4 path; without a path uses example_<task>.mp4. "
            "(default: disabled)"
        ),
    )
    group.add_argument(
        "--fps", type=int, default=30, help="Video frames per second."
    )
    group.add_argument(
        "--width", type=int, default=960, help="Video width in pixels."
    )
    group.add_argument(
        "--height", type=int, default=544, help="Video height in pixels."
    )
    group.add_argument(
        "--track-root",
        action="store_true",
        help="Keep the video camera centered on the creature's root body.",
    )
    group.add_argument(
        "--camera-circle-around",
        action="store_true",
        help=(
            "Circle the video camera 360 degrees around the creature "
            "during the episode."
        ),
    )
    group.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Video playback speed multiplier (for example, 1.25 or 0.8).",
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
    return parser


def validate(args: argparse.Namespace) -> None:
    validate_environment_arguments(args)
    validate_scoring_arguments(args)
    if args.fps < 1:
        raise ValueError("--fps must be at least 1")
    if args.width < 1 or args.height < 1:
        raise ValueError("--width and --height must be at least 1")
    if args.shadowsize < 1:
        raise ValueError("--shadowsize must be greater than zero")
    if args.speed <= 0.0:
        raise ValueError("--speed must be greater than zero")


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

    if args.video:
        from ..video import save_x_axis_video

        video_path = args.video
        if video_path == _DEFAULT_VIDEO_SENTINEL:
            video_path = Path(f"example_{args.task}.mp4")
        try:
            save_x_axis_video(
                genotype=genotype,
                output_path=video_path,
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
            return
        print(f"Saved video: {video_path}")
