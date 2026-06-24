"""Evaluate a genotype on a selected locomotion task."""

import argparse
import os
from pathlib import Path
import sys


def _configure_video_rendering_backend() -> None:
    video_requested = any(
        argument == "--video" or argument.startswith("--video=")
        for argument in sys.argv[1:]
    )
    if sys.platform.startswith("linux") and video_requested:
        os.environ.setdefault("MUJOCO_GL", "egl")


_configure_video_rendering_backend()

from evol_virtual_creature.evaluation import (
    SwimmingEvaluationConfig,
    WalkingEvaluationConfig,
    evaluate_for_task,
)
from evol_virtual_creature.genotype_io import load_genotype_from_json
from evol_virtual_creature.video import save_x_axis_video


DEFAULT_GENOTYPE_PATH = Path(__file__).with_name("examples") / "example_genotype.json"
_DEFAULT_VIDEO_SENTINEL = Path("__task_default_video__")


def main():
    args = parse_args()
    genotype = load_genotype_from_json(args.genotype)
    config_type = (
        WalkingEvaluationConfig if args.task == "walking" else SwimmingEvaluationConfig
    )
    config = config_type(
        episode_seconds=args.duration,
        body_count_weight=args.body_count_weight,
        volume_weight=args.volume_weight,
        volume_penalty_cutoff=args.volume_penalty_cutoff,
        max_volume=args.max_volume,
        self_collision=args.self_collision,
        disallow_collision=args.disallow_collision,
    )
    result = evaluate_for_task(genotype, config)

    if result.disqualified:
        print(f"Disqualified: {result.failure_reason}")
        print(f"Fitness: {result.fitness:.6f}")
        return

    if result.build_failed:
        print(f"Build failed: {result.failure_reason}")
        print(f"Fitness: {result.fitness:.6f}")
        return

    print(f"X-axis {args.task} evaluation")
    print(f"Genotype: {args.genotype}")
    print(f"Fitness: {result.fitness:.6f}")
    print(f"Forward distance: {result.forward_distance:.6f}")
    print(f"Average forward speed: {result.average_forward_speed:.6f}")
    print(f"Sideways drift: {result.sideways_drift:.6f}")
    if args.task == "swimming":
        print(f"Vertical drift: {result.vertical_drift:.6f}")
    else:
        print(f"Height loss: {result.height_loss:.6f}")
        print(f"Mean upright error: {result.mean_upright_error:.6f}")
    print(f"Control energy: {result.control_energy:.6f}")
    print(f"Mean angular speed: {result.mean_angular_speed:.6f}")
    print(f"Simulated seconds: {result.simulated_seconds:.2f}")
    print(f"Actuators: {result.actuator_count}")
    print(f"Total volume: {result.total_volume:.6f}")
    print(f"Bodies: {result.body_count}")

    if args.video:
        video_path = args.video
        if video_path == _DEFAULT_VIDEO_SENTINEL:
            video_path = Path(__file__).with_name(f"example_{args.task}.mp4")
        try:
            save_x_axis_video(
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
    def _get_help_string(self, action):
        if action.default is None or action.default is argparse.SUPPRESS:
            return action.help
        return super()._get_help_string(action)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a creature on a swimming or walking task.",
        formatter_class=_HelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=("swimming", "walking"),
        default="swimming",
        help="Locomotion task to evaluate.",
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
        help="Measured episode duration in seconds.",
    )
    parser.add_argument(
        "--body-count-weight",
        type=float,
        default=SwimmingEvaluationConfig.body_count_weight,
        help="Fitness penalty per generated body.",
    )
    parser.add_argument(
        "--volume-weight",
        type=float,
        default=SwimmingEvaluationConfig.volume_weight,
        help="Fitness penalty per cubic meter of generated creature volume.",
    )
    parser.add_argument(
        "--volume-penalty-cutoff",
        type=float,
        default=SwimmingEvaluationConfig.volume_penalty_cutoff,
        help="Creature volume in cubic meters allowed without a fitness penalty.",
    )
    parser.add_argument(
        "--max-volume",
        type=float,
        default=SwimmingEvaluationConfig.max_volume,
        help="Maximum allowed generated creature volume in cubic meters.",
    )
    parser.add_argument(
        "--self-collision",
        action="store_true",
        help=(
            "Enable collisions between non-parent creature bodies; direct "
            "parent-child collisions remain filtered."
        ),
    )
    parser.add_argument(
        "--disallow-collision",
        action="store_true",
        help=(
            "Assign low fitness if any non-parent creature bodies collide; "
            "self-collision detection is enabled automatically."
        ),
    )
    parser.add_argument(
        "--video",
        nargs="?",
        const=_DEFAULT_VIDEO_SENTINEL,
        type=Path,
        help=(
            "Optional MP4 path; without a path uses example_<task>.mp4. "
            "(default: disabled)"
        ),
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Video frames per second."
    )
    parser.add_argument(
        "--width", type=int, default=960, help="Video width in pixels."
    )
    parser.add_argument(
        "--height", type=int, default=544, help="Video height in pixels."
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
    if args.duration <= 0.0:
        parser.error("--duration must be greater than zero")
    if args.volume_penalty_cutoff < 0.0:
        parser.error("--volume-penalty-cutoff must be non-negative")
    if args.max_volume <= args.volume_penalty_cutoff:
        parser.error("--max-volume must be greater than --volume-penalty-cutoff")
    if args.volume_weight < 0.0:
        parser.error("--volume-weight must be non-negative")
    if args.body_count_weight < 0.0:
        parser.error("--body-count-weight must be non-negative")
    if args.fps < 1:
        parser.error("--fps must be at least 1")
    if args.width < 1 or args.height < 1:
        parser.error("--width and --height must be at least 1")
    if args.speed <= 0.0:
        parser.error("--speed must be greater than zero")
    return args


if __name__ == "__main__":
    main()
