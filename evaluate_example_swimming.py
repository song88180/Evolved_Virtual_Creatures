"""Evaluate the example genotype on the x-axis swimming task."""

from pathlib import Path

from evol_virtual_creature.evaluation import (
    SwimmingEvaluationConfig,
    evaluate_x_axis_swimming,
)
from evol_virtual_creature.genotype_io import load_genotype_from_json


DEFAULT_GENOTYPE_PATH = Path(__file__).with_name("example_genotype.json")


def main():
    genotype = load_genotype_from_json(DEFAULT_GENOTYPE_PATH)
    result = evaluate_x_axis_swimming(
        genotype,
        SwimmingEvaluationConfig(episode_seconds=10.0),
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


if __name__ == "__main__":
    main()
