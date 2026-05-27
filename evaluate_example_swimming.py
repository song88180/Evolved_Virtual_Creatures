"""Evaluate the example genotype on the x-axis swimming task."""

import argparse
from pathlib import Path

from evol_virtual_creature.evaluation import (
    SwimmingEvaluationConfig,
    evaluate_x_axis_swimming,
)
from evol_virtual_creature.genotype_io import load_genotype_from_json
from evol_virtual_creature.video import save_x_axis_swimming_video


DEFAULT_GENOTYPE_PATH = Path(__file__).with_name("example_genotype.json")
DEFAULT_VIDEO_PATH = Path(__file__).with_name("example_swimming.mp4")


def main():
    args = parse_args()
    genotype = load_genotype_from_json(DEFAULT_GENOTYPE_PATH)
    config = SwimmingEvaluationConfig(episode_seconds=args.duration)
    result = evaluate_x_axis_swimming(
        genotype,
        config,
    )

    if result.build_failed:
        print(f"Build failed: {result.failure_reason}")
        print(f"Fitness: {result.fitness:.6f}")
        return

    print("X-axis swimming evaluation")
    print(f"Fitness: {result.fitness:.6f}")
    print(f"Forward distance: {result.forward_distance:.6f}")
    print(f"Average forward speed: {result.average_forward_speed:.6f}")
    print(f"Sideways drift: {result.sideways_drift:.6f}")
    print(f"Vertical drift: {result.vertical_drift:.6f}")
    print(f"Control energy: {result.control_energy:.6f}")
    print(f"Mean angular speed: {result.mean_angular_speed:.6f}")
    print(f"Simulated seconds: {result.simulated_seconds:.2f}")
    print(f"Actuators: {result.actuator_count}")

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
            )
        except RuntimeError as error:
            print(error)
            return
        print(f"Saved video: {video_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate and optionally record the example swimming creature.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Simulation duration in seconds.",
    )
    parser.add_argument(
        "--video",
        nargs="?",
        const=DEFAULT_VIDEO_PATH,
        type=Path,
        help="Optional MP4 output path. Defaults to example_swimming.mp4.",
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
        default=540,
        help="Video height in pixels.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
