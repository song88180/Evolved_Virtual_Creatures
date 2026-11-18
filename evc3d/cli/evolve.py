"""``evc3d evolve``: run mutation-based evolution for a locomotion task."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import random
import sys
from typing import Sequence

from ..evolution_tasks import task_definition
from ..evolution_tasks.shared import EnvironmentFamily
from ..evolution import (
    EvaluatedCreature,
    best_of,
    default_thread_count,
    evaluate_population,
    generation_summary,
    initial_population,
    next_population_with_mutation_flags,
)
from ..genes import CONTROL_MODES
from ..genotype import Genotype
from ..genotype_io import convert_control_mode, load_genotype_from_json
from ..run_artifacts import save_generation_checkpoint, save_run_config
from .args import (
    DEFAULT_GENOTYPE_PATH,
    HelpFormatter,
    add_environment_arguments,
    add_genotype_argument,
    add_mutation_arguments,
    add_scoring_arguments,
    mutation_options,
    resolve_task_definition,
    validate_task_options,
    validate_mutation_arguments,
)

NAME = "evolve"
DEFAULT_RUNS_DIR = Path("runs")

__all__ = ["DEFAULT_GENOTYPE_PATH", "load_genotype_from_json", "parse_args", "run"]


def add_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        NAME,
        help="Run mutation-based evolution.",
        description=(
            "Run mutation-based evolution for swimming, walking, "
            "or flying task variants."
        ),
        formatter_class=HelpFormatter,
    )
    add_genotype_argument(parser, "Seed genotype JSON path.")
    parser.add_argument(
        "--control-mode",
        choices=CONTROL_MODES,
        default=None,
        help=(
            "Controller mode for the run. (default: the seed genotype's "
            "control_mode)"
        ),
    )
    parser.add_argument(
        "--override-control-mode",
        action="store_true",
        help=(
            "Convert the seed genotype to --control-mode when it differs, "
            "discarding the old controller parameters."
        ),
    )
    add_environment_arguments(parser, "Locomotion task used for fitness evaluation.")
    parser.add_argument(
        "--gradual-gravity-change",
        type=_parse_gravity_range,
        default=None,
        metavar="START_GRAVITY,END_GRAVITY",
        help=(
            "For flying evolution, linearly interpolate z gravity from "
            "start to end over generations."
        ),
    )
    add_scoring_arguments(parser, "Evaluation episode duration in seconds.")
    add_mutation_arguments(parser)

    group = parser.add_argument_group("evolution")
    group.add_argument(
        "--population-size",
        type=int,
        default=100,
        help="Number of creatures per generation.",
    )
    group.add_argument(
        "--generations",
        type=int,
        default=100,
        help="Number of evolutionary generations after generation zero.",
    )
    group.add_argument(
        "--elite-count",
        type=int,
        default=5,
        help="Number of top creatures copied unchanged into the next generation.",
    )
    group.add_argument(
        "--tournament-size",
        type=int,
        default=4,
        help="Number of candidates sampled when selecting a parent.",
    )
    group.add_argument(
        "--min-mutations",
        type=int,
        default=1,
        help="Minimum genotype mutations applied to each child.",
    )
    group.add_argument(
        "--max-mutations",
        type=int,
        default=5,
        help="Maximum genotype mutations applied to each child.",
    )
    group.add_argument(
        "--initial-mutations",
        type=int,
        default=3,
        help="Mutations used to seed initial variants around the input genotype.",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible evolution. (default: random)",
    )
    group.add_argument(
        "--threads",
        type=int,
        default=default_thread_count(),
        help=(
            "Number of concurrent population-evaluation worker processes. "
            "Uses half of the available CPU cores by default."
        ),
    )

    group = parser.add_argument_group("output")
    group.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Run output directory. (default: runs/<task>_<timestamp>)",
    )
    group.add_argument(
        "--latest-best-only",
        action="store_true",
        help=(
            "Keep only latest_best_genotype.json instead of saving a separate "
            "best genotype for every generation."
        ),
    )
    group.add_argument(
        "--save-genotype-every-N",
        dest="save_genotype_every_n",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Save a generation-best genotype checkpoint every N generations. "
            "Conflicts with --latest-best-only. (default: 1)"
        ),
    )
    group.add_argument(
        "--record-mutant-type",
        action="store_true",
        help=(
            "Record per-generation counts of fitter, neutral, and less-fit "
            "actual mutants."
        ),
    )
    return parser


def validate(args: argparse.Namespace) -> None:
    """Raise ValueError when evolution arguments are inconsistent."""
    if args.override_control_mode and args.control_mode is None:
        raise ValueError("--override-control-mode requires --control-mode")
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
    validate_mutation_arguments(args)
    validate_task_options(args)
    if (
        args.gradual_gravity_change is not None
        and not task_definition(args.task).environment.supports_scheduled_gravity
    ):
        raise ValueError("--gradual-gravity-change is only supported for flying tasks")
    if args.latest_best_only and args.save_genotype_every_n is not None:
        raise ValueError(
            "--save-genotype-every-N conflicts with --latest-best-only"
        )
    if args.save_genotype_every_n is not None and args.save_genotype_every_n < 1:
        raise ValueError("--save-genotype-every-N must be at least 1")
    if args.save_genotype_every_n is None:
        args.save_genotype_every_n = 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse ``evc3d evolve`` options (``argv`` excludes the command name)."""
    from . import parse_args as parse_cli_args

    return parse_cli_args([NAME, *(sys.argv[1:] if argv is None else argv)])


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


def run(args: argparse.Namespace) -> None:
    """Run the full evolutionary loop and write results to the run directory."""
    from . import user_options

    rng = random.Random(args.seed)
    run_dir = args.output_dir or _default_run_dir(args.task)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot CLI settings so each run can be reproduced from config.json.
    save_run_config(run_dir, user_options(args))

    seed_genotype = load_genotype_from_json(args.genotype)
    seed_genotype = _resolve_seed_control_mode(
        seed_genotype, args.control_mode, args.override_control_mode
    )
    definition = resolve_task_definition(args)
    base_environment = definition.environment
    mutation_kwargs = mutation_options(args)

    population = initial_population(
        seed_genotype=seed_genotype,
        population_size=args.population_size,
        initial_mutations=args.initial_mutations,
        rng=rng,
        **mutation_kwargs,
    )

    best_so_far: EvaluatedCreature | None = None
    changed_by_mutation = [False] * len(population)
    # Best fitness of the previous generation; baseline for --record-mutant-type.
    previous_best_fitness: float | None = None
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
            generation_environment = _environment_for_generation(
                base_environment, args, generation
            )
            definition.environment = generation_environment
            evaluated = evaluate_population(
                population,
                definition,
                executor,
                changed_by_mutation=changed_by_mutation,
            )
            evaluated.sort(key=lambda creature: creature.fitness, reverse=True)
            best = evaluated[0]
            best_so_far = best_of(best_so_far, best)

            summary = generation_summary(
                generation,
                evaluated,
                best_so_far,
                record_mutant_types=args.record_mutant_type,
                mutant_baseline_fitness=previous_best_fitness,
            )
            metrics_file.write(json.dumps(summary) + "\n")
            metrics_file.flush()
            save_generation_checkpoint(
                run_dir,
                generation,
                best,
                best_so_far,
                definition,
                save_generation_history=_should_save_generation_history(
                    generation,
                    args.save_genotype_every_n,
                    latest_best_only=args.latest_best_only,
                ),
            )

            print(_format_generation_progress(generation, best, summary))

            if generation == args.generations:
                break

            previous_best_fitness = best.fitness
            population, changed_by_mutation = next_population_with_mutation_flags(
                evaluated=evaluated,
                population_size=args.population_size,
                elite_count=args.elite_count,
                tournament_size=args.tournament_size,
                min_mutations=args.min_mutations,
                max_mutations=args.max_mutations,
                rng=rng,
                **mutation_kwargs,
            )

    print(f"Best genotype: {run_dir / 'best_genotype.json'}")
    print(f"Metrics: {metrics_path}")


def _format_generation_progress(
    generation: int,
    best: EvaluatedCreature,
    summary: dict,
) -> str:
    """Return the one-line progress report printed during evolution."""
    progress = (
        f"gen={generation:04d} "
        f"best={best.fitness:.6f} "
        f"genes={summary['best_gene_count']} "
        f"bodies={summary['best_body_count']} "
        f"volume={best.result.total_volume:.6f} "
        f"energy={best.result.control_energy:.6f} "
        f"failures={summary['build_failures']} "
        f"disqualified={summary['disqualifications']}"
    )
    if "fitter_mutants" in summary:
        progress += (
            f" fitter={summary['fitter_mutants']}"
            f" neutral={summary['neutral_mutants']}"
            f" less_fit={summary['less_fit_mutants']}"
        )
    return progress


def _parse_gravity_range(value: str) -> tuple[float, float]:
    """Parse START,END gravity values for flying evolution."""
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "--gradual-gravity-change must be START_GRAVITY,END_GRAVITY"
        )
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--gradual-gravity-change values must be numbers"
        ) from error


def _gravity_for_generation(
    gravity_range: tuple[float, float], generation: int, generations: int
) -> float:
    """Return linearly interpolated gravity for a generation."""
    start_gravity, end_gravity = gravity_range
    if generations == 0:
        return start_gravity
    fraction = generation / generations
    return start_gravity + (end_gravity - start_gravity) * fraction


def _environment_for_generation(
    environment: EnvironmentFamily,
    args: argparse.Namespace,
    generation: int,
) -> EnvironmentFamily:
    """Return the environment with any generation-dependent settings."""
    if args.gradual_gravity_change is None:
        return environment
    gravity = _gravity_for_generation(
        args.gradual_gravity_change, generation, args.generations
    )
    return replace(environment, gravity=gravity)


def _resolve_seed_control_mode(
    genotype: Genotype,
    control_mode: str | None,
    override: bool,
) -> Genotype:
    """Return the seed genotype in the requested controller mode.

    A missing or matching ``control_mode`` keeps the genotype as loaded. A
    mismatch is an error unless ``override`` asks for the seed to be converted.
    """
    if control_mode is None or genotype.control_mode == control_mode:
        return genotype
    if not override:
        raise ValueError(
            f"genotype control_mode {genotype.control_mode!r} does not match "
            f"--control-mode {control_mode!r}; add --override-control-mode to "
            f"convert the seed to {control_mode!r} (its "
            f"{genotype.control_mode} controller parameters are discarded and "
            f"the {control_mode} controller starts from zero)"
        )
    print(
        f"Converting seed genotype control_mode from {genotype.control_mode!r} "
        f"to {control_mode!r}."
    )
    return convert_control_mode(genotype, control_mode)


def _should_save_generation_history(
    generation: int,
    save_genotype_every_n: int,
    *,
    latest_best_only: bool,
) -> bool:
    """Return whether the generation-best genotype checkpoint should be written."""
    return not latest_best_only and generation % save_genotype_every_n == 0


def _default_run_dir(task: str) -> Path:
    """Return a task-specific timestamped directory under ``runs/``."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_RUNS_DIR / f"{task}_{timestamp}"
