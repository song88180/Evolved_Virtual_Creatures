"""Reusable helpers for mutation-based virtual-creature evolution."""

from __future__ import annotations

import contextlib
import copy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import io
from itertools import repeat
import json
import os
from pathlib import Path
import random
from statistics import mean
from typing import Any

from .evaluation import EvaluationConfig, evaluate_for_task, task_for_config
from .genotype import Genotype
from .genotype_io import save_genotype_to_json
from .graph_analysis import PhenotypeBuildAbort
from .phenotype import PhenotypeBuilder


@dataclass
class EvaluatedCreature:
    """A genotype paired with its fitness score and evaluation metrics."""

    genotype: Genotype
    fitness: float
    metrics: dict[str, Any]


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
        result = evaluate_for_task(genotype, config)
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
            task=task_for_config(config),
            self_collision=(
                config.self_collision or config.disallow_collision
            ),
            fluid_density=getattr(config, "fluid_density", None),
            fluid_viscosity=getattr(config, "fluid_viscosity", None),
            fluid_shape=getattr(config, "fluid_shape", None),
            fluid_coef=getattr(config, "fluid_coef", None),
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
        mutate_quietly(
            child,
            mutation_count,
            rng,
            allow_topology_mutations=allow_topology_mutations,
            allow_slide_joint=allow_slide_joint,
            allow_root_mutation=allow_root_mutation,
        )
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
    allow_topology_mutations: bool = True,
    allow_slide_joint: bool = False,
    allow_root_mutation: bool = True,
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
        "disqualifications": sum(
            1
            for creature in evaluated
            if creature.metrics.get("disqualified", False)
        ),
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
