"""Evolve virtual creatures for a selected locomotion task."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import random

from evol_virtual_creature.evaluation import (
    DEFAULT_UPRIGHT_ERROR_WEIGHT,
    FlyingAwayEvaluationConfig,
    FlyingEvaluationConfig,
    SwimmingAwayEvaluationConfig,
    SwimmingEvaluationConfig,
    WalkingAwayEvaluationConfig,
    WalkingEvaluationConfig,
)
from evol_virtual_creature.genes import CONTROL_MODES
from evol_virtual_creature.genotype import Genotype
from evol_virtual_creature.evolve import (
    EvaluatedCreature,
    _evaluate_population,
    _save_generation_best,
    best_of,
    default_thread_count,
    generation_summary,
    initial_population,
    next_population_with_mutant_records,
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
    _set_genotype_control_mode(seed_genotype, args.control_mode)
    config_type = _config_type_for_task(args.task)
    config_kwargs = {
        "episode_seconds": args.duration,
        "max_node": args.max_node,
        "body_count_weight": args.body_count_weight,
        "volume_weight": args.volume_weight,
        "volume_penalty_cutoff": args.volume_penalty_cutoff,
        "min_body_volume": args.min_body_volume,
        "min_total_volume": args.min_total_volume,
        "max_volume": args.max_volume,
        "self_collision": args.self_collision,
        "disallow_collision": args.disallow_collision,
    }
    if args.task == "walking_x":
        config_kwargs["upright_weight"] = (
            DEFAULT_UPRIGHT_ERROR_WEIGHT if args.upright_error else 0.0
        )
    if args.task.startswith("flying"):
        config_kwargs.update(
            fluid_density=args.fluid_density,
            fluid_viscosity=args.fluid_viscosity,
            fluid_shape=args.fluid_shape,
            fluid_coef=tuple(args.fluid_coef),
            fitness_gain_fraction=args.fitness_gain_fraction,
        )
    config = config_type(**config_kwargs)

    population = initial_population(
        seed_genotype=seed_genotype,
        population_size=args.population_size,
        initial_mutations=args.initial_mutations,
        rng=rng,
        allow_topology_mutations=not args.disallow_topology_mutations,
        allow_slide_joint=args.allow_slide_joint,
        allow_root_mutation=not args.disallow_root_mutation,
        topology_mutation_rate_min=args.topology_mutation_rate_min,
    )
    _set_population_control_mode(population, args.control_mode)

    best_so_far: EvaluatedCreature | None = None
    mutant_records = [None] * len(population) if args.record_mutant_type else None
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
            generation_config = _config_for_generation(config, args, generation)
            evaluated = _evaluate_population(population, generation_config, executor)
            if args.record_mutant_type:
                evaluated_with_records = list(zip(evaluated, mutant_records))
                evaluated_with_records.sort(
                    key=lambda item: item[0].fitness, reverse=True
                )
                evaluated = [item[0] for item in evaluated_with_records]
                sorted_mutant_records = [item[1] for item in evaluated_with_records]
            else:
                evaluated.sort(key=lambda creature: creature.fitness, reverse=True)
                sorted_mutant_records = None
            best = evaluated[0]
            best_so_far = best_of(best_so_far, best)

            summary = generation_summary(
                generation,
                evaluated,
                best_so_far,
                mutant_records=sorted_mutant_records,
            )
            metrics_file.write(json.dumps(summary) + "\n")
            metrics_file.flush()
            _save_generation_best(
                run_dir,
                generation,
                best,
                best_so_far,
                generation_config,
                save_generation_history=_should_save_generation_history(
                    generation,
                    args.save_genotype_every_n,
                    latest_best_only=args.latest_best_only,
                ),
            )

            print(_format_generation_progress(generation, best, summary))

            if generation == args.generations:
                break

            population, mutant_records = next_population_with_mutant_records(
                evaluated=evaluated,
                population_size=args.population_size,
                elite_count=args.elite_count,
                tournament_size=args.tournament_size,
                min_mutations=args.min_mutations,
                max_mutations=args.max_mutations,
                rng=rng,
                allow_topology_mutations=not args.disallow_topology_mutations,
                allow_slide_joint=args.allow_slide_joint,
                allow_root_mutation=not args.disallow_root_mutation,
                topology_mutation_rate_min=args.topology_mutation_rate_min,
                previous_best_fitness=best.fitness if args.record_mutant_type else None,
            )
            _set_population_control_mode(population, args.control_mode)

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
        f"volume={best.metrics.get('total_volume', 0.0):.6f} "
        f"energy={best.metrics.get('control_energy', 0.0):.6f} "
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
        return SwimmingAwayEvaluationConfig
    return SwimmingEvaluationConfig


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


def _config_for_generation(config, args: argparse.Namespace, generation: int):
    """Return evaluation config with any generation-dependent settings."""
    if args.gradual_gravity_change is None:
        return config
    gravity = _gravity_for_generation(
        args.gradual_gravity_change, generation, args.generations
    )
    return replace(config, gravity=gravity)


def _set_genotype_control_mode(genotype: Genotype, control_mode: str) -> None:
    """Force all encoded connection controllers to the selected mode."""
    for node in genotype.nodes.values():
        for connection in node.children:
            connection.control_mode = control_mode
    for connection in genotype.archived_connections:
        connection.control_mode = control_mode
    for node in genotype.archived_nodes:
        for connection in node.children:
            connection.control_mode = control_mode


def _set_population_control_mode(
    population: list[Genotype],
    control_mode: str,
) -> None:
    """Force every genotype in a population to the selected controller mode."""
    for genotype in population:
        _set_genotype_control_mode(genotype, control_mode)


def _should_save_generation_history(
    generation: int,
    save_genotype_every_n: int,
    *,
    latest_best_only: bool,
) -> bool:
    """Return whether the generation-best genotype checkpoint should be written."""
    return not latest_best_only and generation % save_genotype_every_n == 0


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
        "--control-mode",
        choices=CONTROL_MODES,
        default="neural",
        help="Controller mode used for evolved connection genes.",
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
        "--topology-mutation-rate-min",
        type=float,
        default=Genotype.DEFAULT_TOPOLOGY_MUTATION_RATE_MIN,
        help=(
            "Minimum independent mutation rate for topology-changing mutation "
            "operators."
        ),
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
        "--disallow-root-mutation",
        action="store_true",
        help="Do not mutate the root node gene during evolution.",
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
        "--gradual-gravity-change",
        type=_parse_gravity_range,
        default=None,
        metavar="START_GRAVITY,END_GRAVITY",
        help=(
            "For flying evolution, linearly interpolate z gravity from "
            "start to end over generations."
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
        "--latest-best-only",
        action="store_true",
        help=(
            "Keep only latest_best_genotype.json instead of saving a separate "
            "best genotype for every generation."
        ),
    )
    parser.add_argument(
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
    parser.add_argument(
        "--record-mutant-type",
        action="store_true",
        help=(
            "Record per-generation counts of fitter, neutral, and less-fit "
            "actual mutants."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible evolution. (default: random)",
    )
    args = parser.parse_args()
    _apply_task_defaults(args)
    _validate_args(args)
    return args


def _apply_task_defaults(args: argparse.Namespace) -> None:
    """Fill task-dependent CLI defaults after ``--task`` has been parsed."""
    task_defaults = _config_type_for_task(args.task)()
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

    flying_defaults = (
        task_defaults if args.task.startswith("flying") else FlyingEvaluationConfig()
    )
    for name in (
        "fluid_density",
        "fluid_viscosity",
        "fluid_shape",
        "fluid_coef",
        "fitness_gain_fraction",
    ):
        if getattr(args, name) is None:
            value = getattr(flying_defaults, name)
            if name == "fluid_coef":
                value = list(value)
            setattr(args, name, value)


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
    if not 0.0 <= args.topology_mutation_rate_min <= 1.0:
        raise ValueError(
            "--topology-mutation-rate-min must be between 0 and 1"
        )
    if args.duration <= 0.0:
        raise ValueError("--duration must be greater than zero")
    if args.max_node < 1:
        raise ValueError("--max-node must be at least 1")
    if args.volume_penalty_cutoff < 0.0:
        raise ValueError("--volume-penalty-cutoff must be non-negative")
    if args.max_volume <= args.volume_penalty_cutoff:
        raise ValueError("--max-volume must be greater than --volume-penalty-cutoff")
    if args.min_body_volume < 0.0:
        raise ValueError("--min-body-volume must be non-negative")
    if args.min_total_volume < 0.0:
        raise ValueError("--min-total-volume must be non-negative")
    if args.volume_weight < 0.0:
        raise ValueError("--volume-weight must be non-negative")
    if args.body_count_weight < 0.0:
        raise ValueError("--body-count-weight must be non-negative")
    if args.fluid_density < 0.0:
        raise ValueError("--fluid-density must be non-negative")
    if not 0.0 <= args.fitness_gain_fraction <= 1.0:
        raise ValueError("--fitness-gain-fraction must be between 0 and 1")
    if args.gradual_gravity_change is not None and not args.task.startswith("flying"):
        raise ValueError("--gradual-gravity-change is only supported for flying tasks")
    if args.fluid_viscosity < 0.0:
        raise ValueError("--fluid-viscosity must be non-negative")
    if args.latest_best_only and args.save_genotype_every_n is not None:
        raise ValueError(
            "--save-genotype-every-N conflicts with --latest-best-only"
        )
    if args.save_genotype_every_n is not None and args.save_genotype_every_n < 1:
        raise ValueError("--save-genotype-every-N must be at least 1")
    if args.save_genotype_every_n is None:
        args.save_genotype_every_n = 1


def _default_run_dir(task: str) -> Path:
    """Return a task-specific timestamped directory under ``runs/``."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_RUNS_DIR / f"{task}_{timestamp}"

if __name__ == "__main__":
    main()
