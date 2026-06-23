"""Evaluate a genotype on the x-axis swimming task."""

import argparse
import os
from pathlib import Path
import sys


def _configure_video_rendering_backend() -> None:
    """Use headless EGL rendering for Linux video output by default."""
    video_requested = any(
        argument == "--video" or argument.startswith("--video=")
        for argument in sys.argv[1:]
    )
    if sys.platform.startswith("linux") and video_requested:
        os.environ.setdefault("MUJOCO_GL", "egl")


# MuJoCo selects its OpenGL backend when it is imported, so this must run
# before importing project modules that import mujoco.
_configure_video_rendering_backend()

from evol_virtual_creature.evaluation import (
    SwimmingEvaluationConfig,
    evaluate_x_axis_swimming,
)
from evol_virtual_creature.genotype_io import load_genotype_from_json
from evol_virtual_creature.video import save_x_axis_swimming_video


DEFAULT_GENOTYPE_PATH = Path(__file__).with_name("examples") / "example_genotype.json"
DEFAULT_VIDEO_PATH = Path(__file__).with_name("example_swimming.mp4")


def main():
    args = parse_args()
    genotype = load_genotype_from_json(args.genotype)
    config = SwimmingEvaluationConfig(
        episode_seconds=args.duration,
        body_count_weight=args.body_count_weight,
    )
    result = evaluate_x_axis_swimming(
        genotype,
        config,
    )

    if result.build_failed:
        print(f"Build failed: {result.failure_reason}")
        print(f"Fitness: {result.fitness:.6f}")
        return

    print("X-axis swimming evaluation")
    print(f"Genotype: {args.genotype}")
    print(f"Fitness: {result.fitness:.6f}")
    print(f"Forward distance: {result.forward_distance:.6f}")
    print(f"Average forward speed: {result.average_forward_speed:.6f}")
    print(f"Sideways drift: {result.sideways_drift:.6f}")
    print(f"Vertical drift: {result.vertical_drift:.6f}")
    print(f"Control energy: {result.control_energy:.6f}")
    print(f"Mean angular speed: {result.mean_angular_speed:.6f}")
    print(f"Simulated seconds: {result.simulated_seconds:.2f}")
    print(f"Actuators: {result.actuator_count}")
    print(f"Bodies: {result.body_count}")

    if args.video:
        video_path = args.video
        try:
            save_x_axis_swimming_video(
                genotype=genotype,
                output_path=video_path,
                config=config,
                fps=args.fps,
                width=args.width,
                height=args.height,
                track_root=args.track_root,
                speed=args.speed,
            )
        except RuntimeError as error:
            print(error)
            return
        print(f"Saved video: {video_path}")


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show concrete defaults while allowing dynamic defaults in help text."""

    def _get_help_string(self, action):
        if action.default is None or action.default is argparse.SUPPRESS:
            return action.help
        return super()._get_help_string(action)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and optionally record a swimming creature genotype.",
        formatter_class=_HelpFormatter,
    )
    parser.add_argument(
        "--genotype",
        type=Path,
        default=DEFAULT_GENOTYPE_PATH,
        help="Genotype JSON path. (default: examples/example_genotype.json)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=6.0,
        help="Simulation duration in seconds.",
    )
    parser.add_argument(
        "--body-count-weight",
        type=float,
        default=SwimmingEvaluationConfig.body_count_weight,
        help="Fitness penalty per generated body.",
    )
    parser.add_argument(
        "--video",
        nargs="?",
        const=DEFAULT_VIDEO_PATH,
        type=Path,
        help=(
            "Optional MP4 output path (default: disabled); when this option is "
            "supplied without a path, example_swimming.mp4 is used."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frames per second.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Video width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=544,
        help="Video height in pixels.",
    )
    parser.add_argument(
        "--track-root",
        action="store_true",
        help="Keep the video camera centered on the creature's root body.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Video playback speed multiplier (for example, 1.25 or 0.8).",
    )
    args = parser.parse_args()
    if args.body_count_weight < 0.0:
        parser.error("--body-count-weight must be non-negative")
    if args.speed <= 0.0:
        parser.error("--speed must be greater than zero")
    return args


if __name__ == "__main__":
    main()
