"""Evolve virtual creatures for the x-axis swimming task."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime
from itertools import repeat
import json
from pathlib import Path
import random

from evol_virtual_creature.evaluation import (
    SwimmingEvaluationConfig,
    evaluate_x_axis_swimming,
)
from evol_virtual_creature.evolve import (
    EvaluatedCreature,
    best_of,
    default_thread_count,
    generation_summary,
    initial_population,
    next_population,
    write_json,
)
from evol_virtual_creature.genotype import Genotype
from evol_virtual_creature.genotype_io import (
    load_genotype_from_json,
    save_genotype_to_json,
)
from evol_virtual_creature.graph_analysis import PhenotypeBuildAbort
from evol_virtual_creature.phenotype import PhenotypeBuilder


DEFAULT_GENOTYPE_PATH = Path(__file__).with_name("examples") / "example_genotype.json"
DEFAULT_RUNS_DIR = Path(__file__).with_name("runs")



def main() -> None:
    """Run the full evolutionary loop and write results to the run directory."""
    args = parse_args()
    rng = random.Random(args.seed)
    run_dir = args.output_dir or _default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot CLI settings so each run can be reproduced from config.json.
    write_json(run_dir / "config.json", vars(args))

    seed_genotype = load_genotype_from_json(args.genotype)
    config = SwimmingEvaluationConfig(
        episode_seconds=args.duration,
        max_node=args.max_node,
        body_count_weight=args.body_count_weight,
    )

    population = initial_population(
        seed_genotype=seed_genotype,
        population_size=args.population_size,
        initial_mutations=args.initial_mutations,
        rng=rng,
    )

    best_so_far: EvaluatedCreature | None = None
    metrics_path = run_dir / "metrics.jsonl"

    print(f"Run directory: {run_dir}")
    print(
        "Evolving "
        f"{args.population_size} creatures for {args.generations} generation(s)."
    )

    print(f"Evaluation worker processes: {args.threads}")

    with (
        ProcessPoolExecutor(max_workers=args.threads) as executor,
        metrics_path.open("w") as metrics_file,
    ):
        for generation in range(args.generations + 1):
            evaluated = _evaluate_population(population, config, executor)
            evaluated.sort(key=lambda creature: creature.fitness, reverse=True)
            best = evaluated[0]
            best_so_far = best_of(best_so_far, best)

            summary = generation_summary(generation, evaluated, best_so_far)
            metrics_file.write(json.dumps(summary) + "\n")
            metrics_file.flush()
            _save_generation_best(run_dir, generation, best, best_so_far, args.max_node)

            print(
                f"gen={generation:04d} "
                f"best={best.fitness:.6f} "
                f"best_ever={best_so_far.fitness:.6f} "
                f"mean={summary['mean_fitness']:.6f} "
                f"bodies={summary['best_body_count']} "
                f"failures={summary['build_failures']}"
            )

            if generation == args.generations:
                break

            population = next_population(
                evaluated=evaluated,
                population_size=args.population_size,
                elite_count=args.elite_count,
                tournament_size=args.tournament_size,
                min_mutations=args.min_mutations,
                max_mutations=args.max_mutations,
                rng=rng,
            )

    print(f"Best genotype: {run_dir / 'best_genotype.json'}")
    print(f"Metrics: {metrics_path}")


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show concrete defaults while allowing dynamic defaults in help text."""

    def _get_help_string(self, action):
        if action.default is None or action.default is argparse.SUPPRESS:
            return action.help
        return super()._get_help_string(action)


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments for the evolution run."""
    parser = argparse.ArgumentParser(
        description="Run mutation-based evolution for x-axis swimming creatures.",
        formatter_class=_HelpFormatter,
    )
    parser.add_argument(
        "--genotype",
        type=Path,
        default=DEFAULT_GENOTYPE_PATH,
        help="Seed genotype JSON path. (default: examples/example_genotype.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Run output directory. (default: runs/swimming_<timestamp>)",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=100,
        help="Number of creatures per generation.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=default_thread_count(),
        help=(
            "Number of concurrent population-evaluation worker processes. "
            "Uses half of the available CPU cores by default."
        ),
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=100,
        help="Number of evolutionary generations after generation zero.",
    )
    parser.add_argument(
        "--elite-count",
        type=int,
        default=5,
        help="Number of top creatures copied unchanged into the next generation.",
    )
    parser.add_argument(
        "--tournament-size",
        type=int,
        default=4,
        help="Number of candidates sampled when selecting a parent.",
    )
    parser.add_argument(
        "--min-mutations",
        type=int,
        default=1,
        help="Minimum genotype mutations applied to each child.",
    )
    parser.add_argument(
        "--max-mutations",
        type=int,
        default=5,
        help="Maximum genotype mutations applied to each child.",
    )
    parser.add_argument(
        "--initial-mutations",
        type=int,
        default=3,
        help="Mutations used to seed initial variants around the input genotype.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Evaluation episode duration in seconds.",
    )
    parser.add_argument(
        "--max-node",
        type=int,
        default=500,
        help="Maximum phenotype nodes allowed during build/evaluation.",
    )
    parser.add_argument(
        "--body-count-weight",
        type=float,
        default=SwimmingEvaluationConfig.body_count_weight,
        help="Fitness penalty per generated body.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible evolution. (default: random)",
    )
    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    """Raise ValueError when CLI arguments are inconsistent."""
    if args.population_size < 1:
        raise ValueError("--population-size must be at least 1")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")
    if args.generations < 0:
        raise ValueError("--generations must be non-negative")
    if not 0 <= args.elite_count <= args.population_size:
        raise ValueError("--elite-count must be between 0 and population size")
    if args.tournament_size < 1:
        raise ValueError("--tournament-size must be at least 1")
    if args.min_mutations < 0 or args.max_mutations < args.min_mutations:
        raise ValueError("--max-mutations must be >= --min-mutations >= 0")
    if args.initial_mutations < 0:
        raise ValueError("--initial-mutations must be non-negative")
    if args.body_count_weight < 0.0:
        raise ValueError("--body-count-weight must be non-negative")


def _default_run_dir() -> Path:
    """Return a timestamped directory under ``runs/`` for this evolution run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_RUNS_DIR / f"swimming_{timestamp}"


def _evaluate_population(
    population: list[Genotype],
    config: SwimmingEvaluationConfig,
    executor: ProcessPoolExecutor,
) -> list[EvaluatedCreature]:
    """Simulate and score every genotype concurrently, preserving input order."""
    return list(executor.map(_evaluate_creature, population, repeat(config)))


def _evaluate_creature(
    genotype: Genotype,
    config: SwimmingEvaluationConfig,
) -> EvaluatedCreature:
    """Evaluate one genotype, returning a failure score if simulation raises."""
    try:
        result = evaluate_x_axis_swimming(genotype, config)
        metrics = asdict(result)
        fitness = result.fitness
    except Exception as error:
        metrics = {
            "fitness": config.build_failure_fitness,
            "forward_distance": 0.0,
            "average_forward_speed": 0.0,
            "sideways_drift": 0.0,
            "vertical_drift": 0.0,
            "control_energy": 0.0,
            "mean_angular_speed": 0.0,
            "simulated_seconds": 0.0,
            "actuator_count": 0,
            "body_count": 0,
            "build_failed": True,
            "failure_reason": str(error),
        }
        fitness = config.build_failure_fitness
    return EvaluatedCreature(genotype=genotype, fitness=fitness, metrics=metrics)


def _save_generation_best(
    run_dir: Path,
    generation: int,
    generation_best: EvaluatedCreature,
    best_so_far: EvaluatedCreature,
    max_node: int,
) -> None:
    """Persist the generation best and all-time best genotypes, metrics, and MJCF."""
    save_genotype_to_json(generation_best.genotype, run_dir / "latest_best_genotype.json")
    save_genotype_to_json(best_so_far.genotype, run_dir / "best_genotype.json")
    write_json(run_dir / "best_metrics.json", best_so_far.metrics)
    _write_best_xml(run_dir / "best_creature.xml", best_so_far.genotype, max_node)

    generation_dir = run_dir / "generation_bests"
    generation_dir.mkdir(exist_ok=True)
    save_genotype_to_json(
        generation_best.genotype,
        generation_dir / f"generation_{generation:04d}.json",
    )


def _write_best_xml(path: Path, genotype: Genotype, max_node: int) -> None:
    """Write the best creature's MJCF to disk, skipping output if the build aborts."""
    try:
        mjcf = PhenotypeBuilder(genotype, max_node=max_node).build()
    except PhenotypeBuildAbort:
        return
    path.write_text(mjcf)


if __name__ == "__main__":
    main()
