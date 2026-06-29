"""Evolve virtual creatures for a selected locomotion task."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import random

from evol_virtual_creature.evaluation import (
    FlyingAwayEvaluationConfig,
    FlyingEvaluationConfig,
    OriginDistanceEvaluationConfig,
    SwimmingEvaluationConfig,
    WalkingAwayEvaluationConfig,
    WalkingEvaluationConfig,
)
from evol_virtual_creature.evolve import (
    EvaluatedCreature,
    _evaluate_population,
    _save_generation_best,
    best_of,
    default_thread_count,
    generation_summary,
    initial_population,
    next_population,
    write_json,
)
from evol_virtual_creature.genotype_io import (
    load_genotype_from_json,
)


DEFAULT_GENOTYPE_PATH = Path(__file__).with_name("examples") / "example_genotype.json"
DEFAULT_RUNS_DIR = Path(__file__).with_name("runs")


def main() -> None:
    """Run the full evolutionary loop and write results to the run directory."""
    args = parse_args()
    rng = random.Random(args.seed)
    run_dir = args.output_dir or _default_run_dir(args.task)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot CLI settings so each run can be reproduced from config.json.
    write_json(run_dir / "config.json", vars(args))

    seed_genotype = load_genotype_from_json(args.genotype)
    config_type = _config_type_for_task(args.task)
    config_kwargs = {
        "episode_seconds": args.duration,
        "max_node": args.max_node,
        "body_count_weight": args.body_count_weight,
        "volume_weight": args.volume_weight,
        "volume_penalty_cutoff": args.volume_penalty_cutoff,
        "max_volume": args.max_volume,
        "self_collision": args.self_collision,
        "disallow_collision": args.disallow_collision,
    }
    if args.task.startswith("flying"):
        config_kwargs.update(
            fluid_density=args.fluid_density,
            fluid_viscosity=args.fluid_viscosity,
            fluid_shape=args.fluid_shape,
            fluid_coef=tuple(args.fluid_coef),
        )
    config = config_type(**config_kwargs)

    population = initial_population(
        seed_genotype=seed_genotype,
        population_size=args.population_size,
        initial_mutations=args.initial_mutations,
        rng=rng,
        allow_topology_mutations=not args.disallow_topology_mutations,
        allow_slide_joint=args.allow_slide_joint,
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
            _save_generation_best(
                run_dir,
                generation,
                best,
                best_so_far,
                config,
                save_generation_history=not args.latest_best_only,
            )

            print(_format_generation_progress(generation, best, summary))

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
                allow_topology_mutations=not args.disallow_topology_mutations,
                allow_slide_joint=args.allow_slide_joint,
            )

    print(f"Best genotype: {run_dir / 'best_genotype.json'}")
    print(f"Metrics: {metrics_path}")


def _format_generation_progress(
    generation: int,
    best: EvaluatedCreature,
    summary: dict,
) -> str:
    """Return the one-line progress report printed during evolution."""
    return (
        f"gen={generation:04d} "
        f"best={best.fitness:.6f} "
        f"bodies={summary['best_body_count']} "
        f"volume={best.metrics.get('total_volume', 0.0):.6f} "
        f"energy={best.metrics.get('control_energy', 0.0):.6f} "
        f"failures={summary['build_failures']} "
        f"disqualified={summary['disqualifications']}"
    )


def _config_type_for_task(task: str):
    if task == "flying_x":
        return FlyingEvaluationConfig
    if task == "flying_away":
        return FlyingAwayEvaluationConfig
    if task == "walking_x":
        return WalkingEvaluationConfig
    if task == "walking_away":
        return WalkingAwayEvaluationConfig
    if task == "swimming_away":
        return OriginDistanceEvaluationConfig
    return SwimmingEvaluationConfig


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show concrete defaults while allowing dynamic defaults in help text."""

    def _get_help_string(self, action):
        if action.default is None or action.default is argparse.SUPPRESS:
            return action.help
        return super()._get_help_string(action)


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments for the evolution run."""
    parser = argparse.ArgumentParser(
        description=(
            "Run mutation-based evolution for swimming, walking, "
            "or flying task variants."
        ),
        formatter_class=_HelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=(
            "swimming_x",
            "swimming_away",
            "walking_x",
            "walking_away",
            "flying_x",
            "flying_away",
        ),
        default="swimming_x",
        help="Locomotion task used for fitness evaluation.",
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
        help="Run output directory. (default: runs/<task>_<timestamp>)",
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
        "--disallow-topology-mutations",
        action="store_true",
        help=(
            "Only mutate properties of existing node and connection genes; "
            "do not add, remove, replace, or relink genes."
        ),
    )
    parser.add_argument(
        "--allow-slide-joint",
        action="store_true",
        help="Allow mutations to create slide-jointed body parts.",
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
        "--fluid-density",
        type=float,
        default=FlyingEvaluationConfig.fluid_density,
        help="Fluid density for flying tasks in kg/m^3.",
    )
    parser.add_argument(
        "--fluid-viscosity",
        type=float,
        default=FlyingEvaluationConfig.fluid_viscosity,
        help="Fluid viscosity for flying tasks in Pa*s.",
    )
    parser.add_argument(
        "--fluid-shape",
        choices=("none", "ellipsoid"),
        default=FlyingEvaluationConfig.fluid_shape,
        help="MuJoCo fluid shape approximation for flying creature geoms.",
    )
    parser.add_argument(
        "--fluid-coef",
        type=float,
        nargs=5,
        default=list(FlyingEvaluationConfig.fluid_coef),
        metavar=("BLUNT", "SLENDER", "ANGULAR", "KUTTA", "MAGNUS"),
        help="Five MuJoCo fluid coefficients used by flying creature geoms.",
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
        "--latest-best-only",
        action="store_true",
        help=(
            "Keep only latest_best_genotype.json instead of saving a separate "
            "best genotype for every generation."
        ),
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
    if args.duration <= 0.0:
        raise ValueError("--duration must be greater than zero")
    if args.max_node < 1:
        raise ValueError("--max-node must be at least 1")
    if args.volume_penalty_cutoff < 0.0:
        raise ValueError("--volume-penalty-cutoff must be non-negative")
    if args.max_volume <= args.volume_penalty_cutoff:
        raise ValueError("--max-volume must be greater than --volume-penalty-cutoff")
    if args.volume_weight < 0.0:
        raise ValueError("--volume-weight must be non-negative")
    if args.body_count_weight < 0.0:
        raise ValueError("--body-count-weight must be non-negative")
    if args.fluid_density < 0.0:
        raise ValueError("--fluid-density must be non-negative")
    if args.fluid_viscosity < 0.0:
        raise ValueError("--fluid-viscosity must be non-negative")


def _default_run_dir(task: str) -> Path:
    """Return a task-specific timestamped directory under ``runs/``."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_RUNS_DIR / f"{task}_{timestamp}"


if __name__ == "__main__":
    main()
