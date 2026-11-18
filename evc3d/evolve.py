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

from .evaluation import evaluate_task
from .evolution_tasks.shared import (
    EnvironmentFamily,
    EvaluationConfig,
    EvaluationResult,
    TaskDefinition,
)
from .genotype import Genotype
from .genotype_io import genotype_to_dict, save_genotype_to_json
from .graph_analysis import PhenotypeBuildAbort
from .phenotype import PhenotypeBuilder


@dataclass
class EvaluatedCreature:
    """A genotype paired with its typed task evaluation result.

    ``result`` is the single source of truth for fitness and metrics. Use
    ``metrics_dict()`` only when the result has to be written as JSON.
    ``changed_by_mutation`` is true when this creature's genotype differs from
    the parent it was copied from in the previous generation.
    """

    genotype: Genotype
    result: EvaluationResult
    changed_by_mutation: bool = False

    @property
    def fitness(self) -> float:
        return self.result.fitness

    @property
    def build_failed(self) -> bool:
        return self.result.build_failed

    @property
    def disqualified(self) -> bool:
        return self.result.disqualified

    def metrics_dict(self) -> dict[str, Any]:
        """Return the evaluation result as a JSON-ready dictionary."""
        return asdict(self.result)


def _evaluate_population(
    population: list[Genotype],
    definition: TaskDefinition,
    config_or_executor: EvaluationConfig | ProcessPoolExecutor,
    executor: ProcessPoolExecutor | None = None,
    changed_by_mutation: list[bool] | None = None,
) -> list[EvaluatedCreature]:
    """Simulate and score every genotype concurrently, preserving input order.

    ``changed_by_mutation`` optionally gives one flag per genotype, as returned
    by ``next_population_with_mutation_flags``; it is attached to each result.
    """
    if executor is None:
        executor = config_or_executor
    else:
        definition.config = config_or_executor
    if changed_by_mutation is not None and len(changed_by_mutation) != len(population):
        raise ValueError("changed_by_mutation must match population length")
    evaluated = list(
        executor.map(
            _evaluate_creature,
            population,
            repeat(definition),
        )
    )
    if changed_by_mutation is not None:
        for creature, changed in zip(evaluated, changed_by_mutation):
            creature.changed_by_mutation = changed
    return evaluated


def _evaluate_creature(
    genotype: Genotype,
    definition: TaskDefinition,
) -> EvaluatedCreature:
    """Evaluate one genotype, returning the task's failure result if simulation raises."""
    try:
        result = evaluate_task(genotype, definition)
    except Exception as error:
        result = definition.failed_task_callback(definition.config, str(error))
    return EvaluatedCreature(genotype=genotype, result=result)


def _save_generation_best(
    run_dir: Path,
    generation: int,
    generation_best: EvaluatedCreature,
    best_so_far: EvaluatedCreature,
    definition: TaskDefinition,
    config: EvaluationConfig | None = None,
    save_generation_history: bool = True,
) -> None:
    """Persist the generation best and all-time best genotypes, metrics, and MJCF."""
    if config is not None:
        definition.config = config
    save_genotype_to_json(generation_best.genotype, run_dir / "latest_best_genotype.json")
    save_genotype_to_json(best_so_far.genotype, run_dir / "best_genotype.json")
    write_json(run_dir / "best_metrics.json", best_so_far.metrics_dict())
    _write_best_xml(
        run_dir / "best_creature.xml",
        best_so_far.genotype,
        definition,
    )

    if save_generation_history:
        generation_dir = run_dir / "generation_bests"
        generation_dir.mkdir(exist_ok=True)
        save_genotype_to_json(
            generation_best.genotype,
            generation_dir / f"generation_{generation:04d}.json",
        )


def _write_best_xml(
    path: Path,
    genotype: Genotype,
    definition: TaskDefinition,
    config: EvaluationConfig | None = None,
    environment: EnvironmentFamily | None = None,
) -> None:
    """Write the best creature's MJCF to disk, skipping output if the build aborts."""
    if config is not None:
        definition.config = config
    if environment is not None:
        definition.environment = environment
    try:
        mjcf = PhenotypeBuilder(genotype, definition).build()
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
    population, _changed_by_mutation = next_population_with_mutation_flags(
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
    )
    return population


def next_population_with_mutation_flags(
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
) -> tuple[list[Genotype], list[bool]]:
    """Select the next generation and flag children that changed by mutation.

    Returns the new genotypes and a parallel list that is ``True`` only for
    offspring whose genotype differs from their parent's. Elites and offspring
    whose mutations had no net effect are ``False``.
    """
    next_generation = [
        copy.deepcopy(creature.genotype)
        for creature in evaluated[:elite_count]
    ]
    changed_by_mutation = [False] * len(next_generation)

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
        changed_by_mutation.append(genotype_to_dict(child) != parent_snapshot)

    return next_generation, changed_by_mutation


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
    record_mutant_types: bool = False,
    mutant_baseline_fitness: float | None = None,
) -> dict[str, Any]:
    """Summarize generation fitness statistics for logging and metrics output.

    With ``record_mutant_types``, also count creatures changed by mutation as
    fitter, neutral, or less fit than ``mutant_baseline_fitness``, the previous
    generation's best fitness (``None`` for generation 0, which has no mutants).
    """
    fitnesses = [creature.fitness for creature in evaluated]
    build_failures = sum(
        1
        for creature in evaluated
        if creature.build_failed
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
            if creature.disqualified
        ),
        "best_gene_count": _active_gene_count(evaluated[0].genotype),
        "best_ever_gene_count": _active_gene_count(best_so_far.genotype),
        "best_body_count": evaluated[0].result.body_count,
        "best_ever_body_count": best_so_far.result.body_count,
        "best_metrics": evaluated[0].metrics_dict(),
        "best_ever_metrics": best_so_far.metrics_dict(),
    }
    if record_mutant_types:
        summary.update(_mutant_type_counts(evaluated, mutant_baseline_fitness))
    return summary


def _active_gene_count(genotype: Genotype) -> int:
    """Count active node and connection genes in a genotype."""
    return len(genotype.nodes) + len(genotype.connections)


def _mutant_type_counts(
    evaluated: list[EvaluatedCreature],
    baseline_fitness: float | None,
) -> dict[str, int]:
    """Count creatures changed by mutation by fitness relative to ``baseline_fitness``."""
    if baseline_fitness is None and any(
        creature.changed_by_mutation for creature in evaluated
    ):
        raise ValueError("mutated creatures need a baseline fitness to compare against")

    counts = {
        "fitter_mutants": 0,
        "less_fit_mutants": 0,
        "neutral_mutants": 0,
    }
    for creature in evaluated:
        if not creature.changed_by_mutation:
            continue
        if creature.build_failed or creature.disqualified:
            counts["less_fit_mutants"] += 1
        elif creature.fitness > baseline_fitness:
            counts["fitter_mutants"] += 1
        elif math.isclose(
            creature.fitness,
            baseline_fitness,
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
