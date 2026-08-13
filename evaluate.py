"""Evaluate a genotype on a selected locomotion task."""

import argparse
from dataclasses import fields, replace
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
    DEFAULT_UPRIGHT_ERROR_WEIGHT,
    evaluate_task,
    task_definition,
    task_names,
)
from evol_virtual_creature.evolution_tasks.flying_x import TASK_ENVIRONMENT as DEFAULT_FLYING_ENVIRONMENT
from evol_virtual_creature.genotype_io import load_genotype_from_json
from evol_virtual_creature.video import save_x_axis_video


DEFAULT_GENOTYPE_PATH = Path(__file__).with_name("examples") / "example_genotype.json"
_DEFAULT_VIDEO_SENTINEL = Path("__task_default_video__")


def main():
    args = parse_args()
    genotype = load_genotype_from_json(args.genotype)
    definition = task_definition(args.task)
    config_type = definition.config_type
    config_kwargs = {
        "episode_seconds": args.duration,
        "body_count_weight": args.body_count_weight,
        "volume_weight": args.volume_weight,
        "volume_penalty_cutoff": args.volume_penalty_cutoff,
        "min_body_volume": args.min_body_volume,
        "min_total_volume": args.min_total_volume,
        "max_volume": args.max_volume,
        "self_collision": args.self_collision,
        "disallow_collision": args.disallow_collision,
    }
    config_kwargs.update(
        upright_weight=(
            DEFAULT_UPRIGHT_ERROR_WEIGHT if args.upright_error else 0.0
        ),
        fitness_gain_fraction=args.fitness_gain_fraction,
    )
    environment = definition.environment
    if environment.supports_fluid_overrides:
        environment = replace(
            environment,
            fluid_density=args.fluid_density,
            fluid_viscosity=args.fluid_viscosity,
            fluid_shape=args.fluid_shape,
            fluid_coef=tuple(args.fluid_coef),
        )
    config_kwargs["environment"] = environment
    supported_fields = {field.name for field in fields(config_type)}
    config = config_type(
        **{
            name: value
            for name, value in config_kwargs.items()
            if name in supported_fields
        }
    )
    result = evaluate_task(genotype, config)

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
                shadowsize=args.shadowsize,
                spotlight=args.spotlight,
                camera_circle_around=args.camera_circle_around,
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
        description=(
            "Evaluate a creature on swimming, walking, or flying task variants."
        ),
        formatter_class=_HelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=task_names(),
        default="swimming_x",
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
        default=10.0,
        help="Measured episode duration in seconds.",
    )
    parser.add_argument(
        "--body-count-weight",
        type=float,
        default=None,
        help="Fitness penalty per generated body. (default: task default)",
    )
    parser.add_argument(
        "--volume-weight",
        type=float,
        default=None,
        help=(
            "Fitness penalty per cubic meter of generated creature volume. "
            "(default: task default)"
        ),
    )
    parser.add_argument(
        "--volume-penalty-cutoff",
        type=float,
        default=None,
        help=(
            "Creature volume in cubic meters allowed without a fitness penalty. "
            "(default: task default)"
        ),
    )
    parser.add_argument(
        "--min-body-volume",
        type=float,
        default=None,
        help=(
            "Minimum required volume for each generated body section in cubic meters. "
            "(default: task default)"
        ),
    )
    parser.add_argument(
        "--min-total-volume",
        type=float,
        default=None,
        help=(
            "Minimum required total generated creature volume in cubic meters. "
            "(default: task default)"
        ),
    )
    parser.add_argument(
        "--max-volume",
        type=float,
        default=None,
        help=(
            "Maximum allowed generated creature volume in cubic meters. "
            "(default: task default)"
        ),
    )
    parser.add_argument(
        "--fluid-density",
        type=float,
        default=None,
        help="Fluid density for flying tasks in kg/m^3. (default: task default)",
    )
    parser.add_argument(
        "--fluid-viscosity",
        type=float,
        default=None,
        help="Fluid viscosity for flying tasks in Pa*s. (default: task default)",
    )
    parser.add_argument(
        "--fluid-shape",
        choices=("none", "ellipsoid"),
        default=None,
        help=(
            "MuJoCo fluid shape approximation for flying creature geoms. "
            "(default: task default)"
        ),
    )
    parser.add_argument(
        "--fitness-gain-fraction",
        type=float,
        default=None,
        help=(
            "Blend fraction for controlled-minus-passive flying fitness gain. "
            "(default: task default)"
        ),
    )
    parser.add_argument(
        "--fluid-coef",
        type=float,
        nargs=5,
        default=None,
        metavar=("BLUNT", "SLENDER", "ANGULAR", "KUTTA", "MAGNUS"),
        help=(
            "Five MuJoCo fluid coefficients used by flying creature geoms. "
            "(default: task default)"
        ),
    )
    parser.add_argument(
        "--upright-error",
        action="store_true",
        help=(
            "Enable the root-upright fitness penalty for walking_x tasks."
        ),
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
        "--camera-circle-around",
        action="store_true",
        help=(
            "Circle the video camera 360 degrees around the creature "
            "during the episode."
        ),
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Video playback speed multiplier (for example, 1.25 or 0.8).",
    )
    parser.add_argument(
        "--shadowsize",
        type=int,
        default=4096,
        help=(
            "Shadow-map texture width and height in pixels."
        ),
    )
    parser.add_argument(
        "--spotlight",
        action="store_true",
        help="Use a shadow-casting spotlight instead of a directional light.",
    )
    args = parser.parse_args()
    _apply_task_defaults(args)
    if args.duration <= 0.0:
        parser.error("--duration must be greater than zero")
    if args.volume_penalty_cutoff < 0.0:
        parser.error("--volume-penalty-cutoff must be non-negative")
    if args.max_volume <= args.volume_penalty_cutoff:
        parser.error("--max-volume must be greater than --volume-penalty-cutoff")
    if args.min_body_volume < 0.0:
        parser.error("--min-body-volume must be non-negative")
    if args.min_total_volume < 0.0:
        parser.error("--min-total-volume must be non-negative")
    if args.volume_weight < 0.0:
        parser.error("--volume-weight must be non-negative")
    if args.body_count_weight < 0.0:
        parser.error("--body-count-weight must be non-negative")
    if args.fluid_density < 0.0:
        parser.error("--fluid-density must be non-negative")
    if not 0.0 <= args.fitness_gain_fraction <= 1.0:
        parser.error("--fitness-gain-fraction must be between 0 and 1")
    if args.fluid_viscosity < 0.0:
        parser.error("--fluid-viscosity must be non-negative")
    if args.fps < 1:
        parser.error("--fps must be at least 1")
    if args.width < 1 or args.height < 1:
        parser.error("--width and --height must be at least 1")
    if args.shadowsize < 1:
        parser.error("--shadowsize must be greater than zero")
    if args.speed <= 0.0:
        parser.error("--speed must be greater than zero")
    return args


def _apply_task_defaults(args: argparse.Namespace) -> None:
    """Fill task-dependent CLI defaults after ``--task`` has been parsed."""
    task_defaults = task_definition(args.task).config_type()
    for name in (
        "body_count_weight",
        "volume_weight",
        "volume_penalty_cutoff",
        "min_body_volume",
        "min_total_volume",
        "max_volume",
    ):
        if getattr(args, name) is None:
            setattr(args, name, getattr(task_defaults, name))

    definition = task_definition(args.task)
    environment_defaults = (
        definition.environment
        if definition.environment.supports_fluid_overrides
        else DEFAULT_FLYING_ENVIRONMENT
    )
    for name in ("fluid_density", "fluid_viscosity", "fluid_shape", "fluid_coef"):
        if getattr(args, name) is None:
            value = getattr(environment_defaults, name)
            if name == "fluid_coef":
                value = list(value)
            setattr(args, name, value)
    if args.fitness_gain_fraction is None:
        args.fitness_gain_fraction = getattr(task_defaults, "fitness_gain_fraction", 0.5)


if __name__ == "__main__":
    main()
