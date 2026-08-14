"""Reusable helpers for mutation-based virtual-creature evolution."""

from __future__ import annotations

import contextlib
import copy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import io
from itertools import repeat
import json
import math
import os
from pathlib import Path
import random
from statistics import mean
from typing import Any

from .evaluation import (
    environment_for_config,
    evaluate_task,
    task_definition_for_config,
)
from .evolution_tasks.shared import EvaluationConfig
from .genotype import Genotype
from .genotype_io import genotype_to_dict, save_genotype_to_json
from .graph_analysis import PhenotypeBuildAbort
from .phenotype import PhenotypeBuilder


@dataclass
class EvaluatedCreature:
    """A genotype paired with its fitness score and evaluation metrics."""

    genotype: Genotype
    fitness: float
    metrics: dict[str, Any]


@dataclass(frozen=True)
class MutantRecord:
    """Comparison baseline for a child that actually changed during mutation."""

    previous_best_fitness: float


def _evaluate_population(
    population: list[Genotype],
    config: EvaluationConfig,
    executor: ProcessPoolExecutor,
) -> list[EvaluatedCreature]:
    """Simulate and score every genotype concurrently, preserving input order."""
    return list(executor.map(_evaluate_creature, population, repeat(config)))


def _evaluate_creature(
    genotype: Genotype,
    config: EvaluationConfig,
) -> EvaluatedCreature:
    """Evaluate one genotype, returning a failure score if simulation raises."""
    try:
        result = evaluate_task(genotype, config)
        metrics = asdict(result)
        fitness = result.fitness
    except Exception as error:
        metrics = {
            "fitness": config.build_failure_fitness,
            "forward_distance": 0.0,
            "average_forward_speed": 0.0,
            "sideways_drift_speed": 0.0,
            "vertical_drift_speed": 0.0,
            "control_energy": 0.0,
            "mean_angular_speed": 0.0,
            "simulated_seconds": 0.0,
            "actuator_count": 0,
            "body_count": 0,
            "total_volume": 0.0,
            "build_failed": True,
            "disqualified": False,
            "failure_reason": str(error),
        }
        fitness = config.build_failure_fitness
    return EvaluatedCreature(genotype=genotype, fitness=fitness, metrics=metrics)


def _save_generation_best(
    run_dir: Path,
    generation: int,
    generation_best: EvaluatedCreature,
    best_so_far: EvaluatedCreature,
    config: EvaluationConfig,
    save_generation_history: bool = True,
) -> None:
    """Persist the generation best and all-time best genotypes, metrics, and MJCF."""
    save_genotype_to_json(generation_best.genotype, run_dir / "latest_best_genotype.json")
    save_genotype_to_json(best_so_far.genotype, run_dir / "best_genotype.json")
    write_json(run_dir / "best_metrics.json", best_so_far.metrics)
    _write_best_xml(run_dir / "best_creature.xml", best_so_far.genotype, config)

    if save_generation_history:
        generation_dir = run_dir / "generation_bests"
        generation_dir.mkdir(exist_ok=True)
        save_genotype_to_json(
            generation_best.genotype,
            generation_dir / f"generation_{generation:04d}.json",
        )


def _write_best_xml(
    path: Path, genotype: Genotype, config: EvaluationConfig
) -> None:
    """Write the best creature's MJCF to disk, skipping output if the build aborts."""
    try:
        mjcf = PhenotypeBuilder(
            genotype,
            max_node=config.max_node,
            environment=environment_for_config(config),
            self_collision=(
                config.self_collision or config.disallow_collision
            ),
        ).build()
    except PhenotypeBuildAbort:
        return
    path.write_text(mjcf)


def default_thread_count() -> int:
    """Use half of the CPU cores available to this process, with a minimum of one."""
    if hasattr(os, "sched_getaffinity"):
        available_cores = len(os.sched_getaffinity(0))
    else:
        available_cores = os.cpu_count() or 1
    return max(1, available_cores // 2)


def initial_population(
    seed_genotype: Genotype,
    population_size: int,
    initial_mutations: int,
    rng: random.Random,
    allow_topology_mutations: bool = True,
    allow_slide_joint: bool = False,
    allow_root_mutation: bool = True,
    topology_mutation_rate_min: float = Genotype.DEFAULT_TOPOLOGY_MUTATION_RATE_MIN,
) -> list[Genotype]:
    """Build the starting population from the seed genotype and random mutations."""
    population = [copy.deepcopy(seed_genotype)]
    while len(population) < population_size:
        genotype = copy.deepcopy(seed_genotype)
        mutate_quietly(
            genotype,
            initial_mutations,
            rng,
            allow_topology_mutations=allow_topology_mutations,
            allow_slide_joint=allow_slide_joint,
            allow_root_mutation=allow_root_mutation,
            topology_mutation_rate_min=topology_mutation_rate_min,
        )
        population.append(genotype)
    return population


def next_population(
    evaluated: list[EvaluatedCreature],
    population_size: int,
    elite_count: int,
    tournament_size: int,
    min_mutations: int,
    max_mutations: int,
    rng: random.Random,
    allow_topology_mutations: bool = True,
    allow_slide_joint: bool = False,
    allow_root_mutation: bool = True,
    topology_mutation_rate_min: float = Genotype.DEFAULT_TOPOLOGY_MUTATION_RATE_MIN,
) -> list[Genotype]:
    """Select elites and tournament winners, then mutate them into the next generation."""
    population, _mutant_records = next_population_with_mutant_records(
        evaluated=evaluated,
        population_size=population_size,
        elite_count=elite_count,
        tournament_size=tournament_size,
        min_mutations=min_mutations,
        max_mutations=max_mutations,
        rng=rng,
        allow_topology_mutations=allow_topology_mutations,
        allow_slide_joint=allow_slide_joint,
        allow_root_mutation=allow_root_mutation,
        topology_mutation_rate_min=topology_mutation_rate_min,
        previous_best_fitness=None,
    )
    return population


def next_population_with_mutant_records(
    evaluated: list[EvaluatedCreature],
    population_size: int,
    elite_count: int,
    tournament_size: int,
    min_mutations: int,
    max_mutations: int,
    rng: random.Random,
    allow_topology_mutations: bool = True,
    allow_slide_joint: bool = False,
    allow_root_mutation: bool = True,
    topology_mutation_rate_min: float = Genotype.DEFAULT_TOPOLOGY_MUTATION_RATE_MIN,
    previous_best_fitness: float | None = None,
) -> tuple[list[Genotype], list[MutantRecord | None]]:
    """Select the next generation and mark children that changed by mutation."""
    next_generation = [
        copy.deepcopy(creature.genotype)
        for creature in evaluated[:elite_count]
    ]
    mutant_records: list[MutantRecord | None] = [None] * len(next_generation)

    while len(next_generation) < population_size:
        parent = tournament_select(evaluated, tournament_size, rng)
        child = copy.deepcopy(parent.genotype)
        parent_snapshot = genotype_to_dict(parent.genotype)
        mutation_count = rng.randint(min_mutations, max_mutations)
        mutate_quietly(
            child,
            mutation_count,
            rng,
            allow_topology_mutations=allow_topology_mutations,
            allow_slide_joint=allow_slide_joint,
            allow_root_mutation=allow_root_mutation,
            topology_mutation_rate_min=topology_mutation_rate_min,
        )
        next_generation.append(child)
        if (
            previous_best_fitness is not None
            and genotype_to_dict(child) != parent_snapshot
        ):
            mutant_records.append(
                MutantRecord(previous_best_fitness=previous_best_fitness)
            )
        else:
            mutant_records.append(None)

    return next_generation, mutant_records


def tournament_select(
    evaluated: list[EvaluatedCreature],
    tournament_size: int,
    rng: random.Random,
) -> EvaluatedCreature:
    """Pick the fittest creature from a random subset of the population."""
    sample_size = min(tournament_size, len(evaluated))
    candidates = rng.sample(evaluated, k=sample_size)
    return max(candidates, key=lambda creature: creature.fitness)


def mutate_quietly(
    genotype: Genotype,
    mutation_count: int,
    rng: random.Random,
    allow_topology_mutations: bool = True,
    allow_slide_joint: bool = False,
    allow_root_mutation: bool = True,
    topology_mutation_rate_min: float = Genotype.DEFAULT_TOPOLOGY_MUTATION_RATE_MIN,
) -> None:
    """Apply genotype mutations while suppressing mutation log output."""
    if mutation_count == 0:
        return
    with contextlib.redirect_stdout(io.StringIO()):
        genotype.mutation(
            num_mutations=mutation_count,
            rng=rng,
            allow_topology_mutations=allow_topology_mutations,
            allow_slide_joint=allow_slide_joint,
            allow_root_mutation=allow_root_mutation,
            topology_mutation_rate_min=topology_mutation_rate_min,
        )


def best_of(
    current_best: EvaluatedCreature | None,
    candidate: EvaluatedCreature,
) -> EvaluatedCreature:
    """Return the higher-fitness creature, keeping the incumbent when tied or better."""
    if current_best is None or candidate.fitness > current_best.fitness:
        return copy.deepcopy(candidate)
    return current_best


def generation_summary(
    generation: int,
    evaluated: list[EvaluatedCreature],
    best_so_far: EvaluatedCreature,
    mutant_records: list[MutantRecord | None] | None = None,
) -> dict[str, Any]:
    """Summarize generation fitness statistics for logging and metrics output."""
    fitnesses = [creature.fitness for creature in evaluated]
    build_failures = sum(
        1
        for creature in evaluated
        if creature.metrics.get("build_failed", False)
    )
    summary = {
        "generation": generation,
        "best_fitness": evaluated[0].fitness,
        "best_ever_fitness": best_so_far.fitness,
        "mean_fitness": mean(fitnesses),
        "worst_fitness": fitnesses[-1],
        "build_failures": build_failures,
        "disqualifications": sum(
            1
            for creature in evaluated
            if creature.metrics.get("disqualified", False)
        ),
        "best_gene_count": _active_gene_count(evaluated[0].genotype),
        "best_ever_gene_count": _active_gene_count(best_so_far.genotype),
        "best_body_count": evaluated[0].metrics.get("body_count", 0),
        "best_ever_body_count": best_so_far.metrics.get("body_count", 0),
        "best_metrics": evaluated[0].metrics,
        "best_ever_metrics": best_so_far.metrics,
    }
    if mutant_records is not None:
        summary.update(_mutant_type_counts(evaluated, mutant_records))
    return summary


def _active_gene_count(genotype: Genotype) -> int:
    """Count active node and connection genes in a genotype."""
    return len(genotype.nodes) + len(genotype.connections)


def _mutant_type_counts(
    evaluated: list[EvaluatedCreature],
    mutant_records: list[MutantRecord | None],
) -> dict[str, int]:
    """Count actual mutants by fitness relative to the previous generation best."""
    if len(mutant_records) != len(evaluated):
        raise ValueError("mutant records must match evaluated population length")

    counts = {
        "fitter_mutants": 0,
        "less_fit_mutants": 0,
        "neutral_mutants": 0,
    }
    for creature, record in zip(evaluated, mutant_records):
        if record is None:
            continue
        if (
            creature.metrics.get("build_failed", False)
            or creature.metrics.get("disqualified", False)
        ):
            counts["less_fit_mutants"] += 1
        elif creature.fitness > record.previous_best_fitness:
            counts["fitter_mutants"] += 1
        elif math.isclose(
            creature.fitness,
            record.previous_best_fitness,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            counts["neutral_mutants"] += 1
        else:
            counts["less_fit_mutants"] += 1
    return counts


def write_json(path: Path, data: Any) -> None:
    """Write a JSON-serializable object to ``path``, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, default=str)
        f.write("\n")
