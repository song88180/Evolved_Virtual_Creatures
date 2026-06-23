from concurrent.futures import ProcessPoolExecutor
import sys

import pytest

import evolve_swimming
import evaluate_swimming
import generate_model


def test_default_thread_count_uses_half_available_affinity(monkeypatch):
    monkeypatch.setattr(
        evolve_swimming.os, "sched_getaffinity", lambda _pid: set(range(10))
    )

    assert evolve_swimming._default_thread_count() == 5


def test_default_thread_count_has_minimum_of_one(monkeypatch):
    monkeypatch.setattr(
        evolve_swimming.os, "sched_getaffinity", lambda _pid: {0}
    )

    assert evolve_swimming._default_thread_count() == 1


def test_parse_args_accepts_thread_override(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evolve_swimming.py", "--threads", "3"])

    assert evolve_swimming.parse_args().threads == 3


def test_parse_args_rejects_nonpositive_threads(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evolve_swimming.py", "--threads", "0"])

    with pytest.raises(ValueError, match="--threads must be at least 1"):
        evolve_swimming.parse_args()


def test_evaluate_population_runs_in_processes_and_preserves_order():
    genotype = evolve_swimming.load_genotype_from_json(
        evolve_swimming.DEFAULT_GENOTYPE_PATH
    )
    population = [genotype, genotype]
    config = evolve_swimming.SwimmingEvaluationConfig(episode_seconds=0.02)

    with ProcessPoolExecutor(max_workers=2) as executor:
        evaluated = evolve_swimming._evaluate_population(
            population, config, executor
        )

    assert len(evaluated) == len(population)
    assert [creature.fitness for creature in evaluated] == [
        evaluated[0].fitness,
        evaluated[0].fitness,
    ]

@pytest.mark.parametrize(
    ("parse_args", "expected_defaults"),
    [
        (
            evaluate_swimming.parse_args,
            (
                "default: examples/example_genotype.json",
                "default: 6.0",
                "default: 0.001",
                "default: disabled",
                "default: 30",
                "default: 960",
                "default: 544",
            ),
        ),
        (
            evolve_swimming.parse_args,
            (
                "default: examples/example_genotype.json",
                "default: runs/swimming_<timestamp>",
                "default: 100",
                f"default: {evolve_swimming._default_thread_count()}",
                "default: 5",
                "default: 4",
                "default: 1",
                "default: 3",
                "default: 10.0",
                "default: 500",
                "default: 0.001",
                "default: random",
            ),
        ),
        (
            generate_model.parse_args,
            (
                "default: examples/example_genotype.json",
                "default: generated_creature.xml",
                "default: 50",
                "default: 500",
            ),
        ),
    ],
)
def test_help_describes_parameter_defaults(
    monkeypatch, capsys, parse_args, expected_defaults
):
    monkeypatch.setattr(sys, "argv", ["program", "--help"])

    with pytest.raises(SystemExit, match="0"):
        parse_args()

    normalized_help = " ".join(capsys.readouterr().out.split())
    for expected_default in expected_defaults:
        assert expected_default in normalized_help

