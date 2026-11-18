"""Tests for evc3d.evolution: population creation, selection, mutation flags, evaluation, and generation summaries."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import FrozenInstanceError, replace
import random

import pytest

from evc3d import evolution
from evc3d.cli import evolve as evolve_cli
from evc3d.evolution_tasks import task_definition
from evc3d.evolution_tasks.flying_away import FlyingAwayEvaluationConfig
from evc3d.evolution_tasks.flying_x import FlyingEvaluationConfig
from evc3d.evolution_tasks.swimming_away import SwimmingAwayEvaluationConfig
from evc3d.evolution_tasks.swimming_x import SwimmingEvaluationConfig
from evc3d.evolution_tasks.walking_away import WalkingAwayEvaluationConfig
from evc3d.evolution_tasks.walking_x import WalkingEvaluationConfig
from helpers import make_creature


def test_default_thread_count_uses_half_available_affinity(monkeypatch):
    monkeypatch.setattr(
        evolution.os, "sched_getaffinity", lambda _pid: set(range(10))
    )
    assert evolution.default_thread_count() == 5


def test_default_thread_count_has_minimum_of_one(monkeypatch):
    monkeypatch.setattr(evolution.os, "sched_getaffinity", lambda _pid: {0})
    assert evolution.default_thread_count() == 1


@pytest.mark.parametrize(
    "task, config",
    [
        ("swimming_x", SwimmingEvaluationConfig(episode_seconds=0.02)),
        ("walking_x", WalkingEvaluationConfig(episode_seconds=0.02, settle_seconds=0.0)),
        ("swimming_away", SwimmingAwayEvaluationConfig(episode_seconds=0.02)),
        ("walking_away", WalkingAwayEvaluationConfig(episode_seconds=0.02, settle_seconds=0.0)),
        ("flying_x", FlyingEvaluationConfig(episode_seconds=0.02)),
        ("flying_away", FlyingAwayEvaluationConfig(episode_seconds=0.02)),
    ],
)


def test_evaluate_population_runs_in_processes_and_preserves_order(task, config):
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    population = [genotype, genotype]
    with ProcessPoolExecutor(max_workers=2) as executor:
        evaluated = evolution.evaluate_population(
            population, replace(task_definition(task), config=config), executor
        )
    assert len(evaluated) == len(population)
    assert evaluated[0].fitness == evaluated[1].fitness


def test_generation_summary_counts_disqualifications_and_active_genes():
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    creature = make_creature(genotype, -1_000.0, disqualified=True, body_count=1)
    summary = evolution.generation_summary(0, [creature], creature)
    expected_genes = len(genotype.nodes) + sum(
        len(node.child_connections)
        for node in genotype.nodes.values()
    )
    assert summary["disqualifications"] == 1
    assert summary["build_failures"] == 0
    assert summary["best_gene_count"] == expected_genes
    assert summary["best_ever_gene_count"] == expected_genes


def test_generation_summary_omits_mutant_type_counts_by_default():
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    creature = make_creature(genotype, 1.0)

    summary = evolution.generation_summary(0, [creature], creature)

    assert "fitter_mutants" not in summary
    assert "neutral_mutants" not in summary
    assert "less_fit_mutants" not in summary


def test_generation_summary_counts_mutant_types_against_previous_best():
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    evaluated = [
        make_creature(genotype, 2.0, changed_by_mutation=True),
        make_creature(genotype, 1.5, changed_by_mutation=True),
        make_creature(genotype, 1.0, changed_by_mutation=True),
        make_creature(genotype, 3.0, changed_by_mutation=True, build_failed=True),
        make_creature(genotype, 3.0, changed_by_mutation=True, disqualified=True),
        make_creature(genotype, 5.0),
    ]

    summary = evolution.generation_summary(
        1,
        evaluated,
        evaluated[0],
        record_mutant_types=True,
        mutant_baseline_fitness=1.5,
    )

    assert summary["fitter_mutants"] == 1
    assert summary["neutral_mutants"] == 1
    assert summary["less_fit_mutants"] == 3


def test_generation_summary_reports_zero_mutant_counts_for_initial_generation():
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    evaluated = [make_creature(genotype, 1.0), make_creature(genotype, 0.5)]

    summary = evolution.generation_summary(
        0, evaluated, evaluated[0], record_mutant_types=True
    )

    assert summary["fitter_mutants"] == 0
    assert summary["neutral_mutants"] == 0
    assert summary["less_fit_mutants"] == 0


def test_generation_summary_requires_baseline_when_creatures_were_mutated():
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    evaluated = [make_creature(genotype, 1.0, changed_by_mutation=True)]

    with pytest.raises(ValueError, match="baseline fitness"):
        evolution.generation_summary(
            1, evaluated, evaluated[0], record_mutant_types=True
        )


def test_evaluated_creature_fitness_is_read_from_result():
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    creature = make_creature(genotype, 2.0, body_count=3)

    assert creature.fitness == 2.0
    assert creature.metrics_dict()["fitness"] == 2.0
    assert creature.metrics_dict()["body_count"] == 3
    with pytest.raises(AttributeError):
        creature.fitness = 5.0
    with pytest.raises(FrozenInstanceError):
        creature.result.fitness = 5.0


def test_evaluate_creature_uses_task_failure_result_when_evaluation_raises(monkeypatch):
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    definition = task_definition("walking_x")
    definition.config = WalkingEvaluationConfig(episode_seconds=0.02)

    def explode(_genotype, _definition):
        raise RuntimeError("boom")

    monkeypatch.setattr(evolution, "evaluate_task", explode)

    creature = evolution._evaluate_creature(genotype, definition)

    assert isinstance(creature.result, definition.result_type)
    assert creature.fitness == definition.config.build_failure_fitness
    assert creature.build_failed
    assert not creature.disqualified
    assert creature.result.failure_reason == "boom"


def test_next_population_with_mutation_flags_marks_only_changed_children(monkeypatch):
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    evaluated = [
        make_creature(genotype, 1.0),
    ]
    calls = 0

    def mutate_second_child(
        genotype,
        _mutation_count,
        _rng,
        allow_topology_mutations=True,
        allow_slide_joint=False,
        allow_root_mutation=True,
        topology_mutation_rate_min=0.05,
    ):
        nonlocal calls
        calls += 1
        if calls == 2:
            root = genotype.nodes[genotype.root]
            root.size = (root.size[0] + 0.1, root.size[1], root.size[2])

    monkeypatch.setattr(evolution, "mutate_quietly", mutate_second_child)

    population, changed = evolution.next_population_with_mutation_flags(
        evaluated=evaluated,
        population_size=3,
        elite_count=1,
        tournament_size=1,
        min_mutations=1,
        max_mutations=1,
        rng=random.Random(1),
    )

    assert len(population) == 3
    assert changed == [False, False, True]


def test_initial_population_forwards_topology_mutation_setting(monkeypatch):
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    settings = []

    def record_mutation(
        _genotype,
        _mutation_count,
        _rng,
        allow_topology_mutations=True,
        allow_slide_joint=False,
        allow_root_mutation=True,
        topology_mutation_rate_min=0.05,
    ):
        settings.append(
            (allow_topology_mutations, allow_slide_joint, topology_mutation_rate_min)
        )

    monkeypatch.setattr(evolution, "mutate_quietly", record_mutation)
    evolution.initial_population(
        seed_genotype=genotype,
        population_size=3,
        initial_mutations=2,
        rng=random.Random(1),
        allow_topology_mutations=False,
        topology_mutation_rate_min=0.2,
    )

    assert settings == [(False, False, 0.2), (False, False, 0.2)]


def test_next_population_forwards_topology_mutation_setting(monkeypatch):
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    evaluated = [
        make_creature(genotype, 1.0),
    ]
    settings = []

    def record_mutation(
        _genotype,
        _mutation_count,
        _rng,
        allow_topology_mutations=True,
        allow_slide_joint=False,
        allow_root_mutation=True,
        topology_mutation_rate_min=0.05,
    ):
        settings.append(
            (allow_topology_mutations, allow_slide_joint, topology_mutation_rate_min)
        )

    monkeypatch.setattr(evolution, "mutate_quietly", record_mutation)
    evolution.next_population(
        evaluated=evaluated,
        population_size=3,
        elite_count=1,
        tournament_size=1,
        min_mutations=1,
        max_mutations=1,
        rng=random.Random(1),
        allow_topology_mutations=False,
        topology_mutation_rate_min=0.2,
    )

    assert settings == [(False, False, 0.2), (False, False, 0.2)]


def test_population_helpers_forward_allow_slide_joint(monkeypatch):
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    evaluated = [
        make_creature(genotype, 1.0),
    ]
    settings = []

    def record_mutation(
        _genotype,
        _mutation_count,
        _rng,
        allow_topology_mutations=True,
        allow_slide_joint=False,
        allow_root_mutation=True,
        topology_mutation_rate_min=0.05,
    ):
        settings.append(allow_slide_joint)

    monkeypatch.setattr(evolution, "mutate_quietly", record_mutation)
    evolution.initial_population(
        seed_genotype=genotype,
        population_size=2,
        initial_mutations=1,
        rng=random.Random(1),
        allow_slide_joint=True,
    )
    evolution.next_population(
        evaluated=evaluated,
        population_size=2,
        elite_count=1,
        tournament_size=1,
        min_mutations=1,
        max_mutations=1,
        rng=random.Random(1),
        allow_slide_joint=True,
    )

    assert settings == [True, True]


def test_population_helpers_forward_root_mutation_setting(monkeypatch):
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    evaluated = [
        make_creature(genotype, 1.0),
    ]
    settings = []

    def record_mutation(
        _genotype,
        _mutation_count,
        _rng,
        allow_topology_mutations=True,
        allow_slide_joint=False,
        allow_root_mutation=True,
        topology_mutation_rate_min=0.05,
    ):
        settings.append(allow_root_mutation)

    monkeypatch.setattr(evolution, "mutate_quietly", record_mutation)
    evolution.initial_population(
        seed_genotype=genotype,
        population_size=2,
        initial_mutations=1,
        rng=random.Random(1),
        allow_root_mutation=False,
    )
    evolution.next_population(
        evaluated=evaluated,
        population_size=2,
        elite_count=1,
        tournament_size=1,
        min_mutations=1,
        max_mutations=1,
        rng=random.Random(1),
        allow_root_mutation=False,
    )

    assert settings == [False, False]
