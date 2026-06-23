"""Reusable helpers for mutation-based virtual-creature evolution."""

from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import random
from statistics import mean
from typing import Any

from .genotype import Genotype


@dataclass
class EvaluatedCreature:
    """A genotype paired with its fitness score and evaluation metrics."""

    genotype: Genotype
    fitness: float
    metrics: dict[str, Any]


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
) -> list[Genotype]:
    """Build the starting population from the seed genotype and random mutations."""
    population = [copy.deepcopy(seed_genotype)]
    while len(population) < population_size:
        genotype = copy.deepcopy(seed_genotype)
        mutate_quietly(genotype, initial_mutations, rng)
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
) -> list[Genotype]:
    """Select elites and tournament winners, then mutate them into the next generation."""
    next_generation = [
        copy.deepcopy(creature.genotype)
        for creature in evaluated[:elite_count]
    ]

    while len(next_generation) < population_size:
        parent = tournament_select(evaluated, tournament_size, rng)
        child = copy.deepcopy(parent.genotype)
        mutation_count = rng.randint(min_mutations, max_mutations)
        mutate_quietly(child, mutation_count, rng)
        next_generation.append(child)

    return next_generation


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
) -> None:
    """Apply genotype mutations while suppressing mutation log output."""
    if mutation_count == 0:
        return
    with contextlib.redirect_stdout(io.StringIO()):
        genotype.mutation(num_mutations=mutation_count, rng=rng)


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
) -> dict[str, Any]:
    """Summarize generation fitness statistics for logging and metrics output."""
    fitnesses = [creature.fitness for creature in evaluated]
    build_failures = sum(
        1
        for creature in evaluated
        if creature.metrics.get("build_failed", False)
    )
    return {
        "generation": generation,
        "best_fitness": evaluated[0].fitness,
        "best_ever_fitness": best_so_far.fitness,
        "mean_fitness": mean(fitnesses),
        "worst_fitness": fitnesses[-1],
        "build_failures": build_failures,
        "best_body_count": evaluated[0].metrics.get("body_count", 0),
        "best_ever_body_count": best_so_far.metrics.get("body_count", 0),
        "best_metrics": evaluated[0].metrics,
        "best_ever_metrics": best_so_far.metrics,
    }


def write_json(path: Path, data: Any) -> None:
    """Write a JSON-serializable object to ``path``, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, default=str)
        f.write("\n")
