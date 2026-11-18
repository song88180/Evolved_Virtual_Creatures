"""Tests for the evc3d command: argument parsing, validation, help text, and subcommand behaviour."""

import argparse
import random
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from evc3d import evolution, cli
from evc3d.cli.args import resolve_task_definition
from evc3d.cli import (
    build as build_cli,
    evaluate,
    evolve as evolve_cli,
    mutate as mutate_cli,
    view as view_cli,
)
from evc3d.evolution_tasks import task_definition
from evc3d.evolution_tasks.swimming_x import SwimmingEvaluationConfig
from evc3d.genotype import Genotype
from evc3d.genotype_io import build_genotype
from evc3d.phenotype import PhenotypeBuilder
from helpers import make_definition, make_creature


def _sine_seed_genotype():
    return build_genotype(
        root="body",
        nodes={
            "body": {
                "size": (0.2, 0.2, 0.2),
                "joint_type": "free",
                "child_connections": ["body_to_limb"],
            },
            "limb": {"size": (0.1, 0.1, 0.1)},
        },
        connections={
            "body_to_limb": {
                "child": "limb",
                "axis": (0.0, 1.0, 0.0),
                "control_amp": 0.4,
                "control_freq": 7,
                "control_phase": 1.2,
            }
        },
        global_control_freq=2.5,
        control_mode="sine",
    )


def test_evaluate_defaults_to_swimming_without_self_collision(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate.py"])
    args = evaluate.parse_args()
    assert args.task == "swimming_x"
    assert not args.self_collision
    assert not args.disallow_collision


def test_evaluate_accepts_disallow_collision(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--disallow-collision"])
    args = evaluate.parse_args()
    assert args.disallow_collision
    assert not args.self_collision


def test_evaluate_accepts_volume_weight(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["evaluate.py", "--volume-weight", "0.25"]
    )
    assert evaluate.parse_args().volume_weight == pytest.approx(0.25)


def test_evaluate_accepts_volume_cutoff_and_maximum(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--volume-penalty-cutoff",
            "0.2",
            "--max-volume",
            "2.0",
        ],
    )
    args = evaluate.parse_args()
    assert args.volume_penalty_cutoff == pytest.approx(0.2)
    assert args.max_volume == pytest.approx(2.0)


def test_evaluate_accepts_self_collision(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--self-collision"])
    assert evaluate.parse_args().self_collision


def test_evaluate_accepts_upright_error(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["evaluate.py", "--task", "walking_x", "--upright-error"]
    )
    assert evaluate.parse_args().upright_error


def test_evolve_control_mode_defaults_to_genotype(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evolve.py"])
    args = evolve_cli.parse_args()
    assert args.control_mode is None
    assert args.override_control_mode is False


def test_evolve_accepts_sine_control_mode(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evolve.py", "--control-mode", "sine"])
    assert evolve_cli.parse_args().control_mode == "sine"


def test_evolve_override_control_mode_requires_control_mode(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["evolve.py", "--override-control-mode"])
    with pytest.raises(SystemExit, match="2"):
        evolve_cli.parse_args()
    assert re.search("requires --control-mode", capsys.readouterr().err)


def test_resolve_seed_control_mode_keeps_genotype_without_cli_mode():
    genotype = _sine_seed_genotype()
    assert evolve_cli._resolve_seed_control_mode(genotype, None, False) is genotype


def test_resolve_seed_control_mode_accepts_match():
    genotype = _sine_seed_genotype()
    assert evolve_cli._resolve_seed_control_mode(genotype, "sine", False) is genotype


def test_resolve_seed_control_mode_rejects_mismatch_without_override():
    with pytest.raises(ValueError, match="add --override-control-mode"):
        evolve_cli._resolve_seed_control_mode(_sine_seed_genotype(), "neural", False)


def test_resolve_seed_control_mode_converts_with_override():
    genotype = _sine_seed_genotype()
    converted = evolve_cli._resolve_seed_control_mode(genotype, "neural", True)

    assert converted.control_mode == "neural"
    assert genotype.control_mode == "sine"
    connection = converted.connections["body_to_limb"]
    assert connection.control_amp == 0.0
    assert connection.neural_b2 == (0.0,)


def test_evolve_accepts_upright_error(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["evolve.py", "--task", "walking_x", "--upright-error"]
    )
    assert evolve_cli.parse_args().upright_error


def test_evaluate_disables_upright_error_by_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--task", "walking_x"])
    assert not evaluate.parse_args().upright_error


def test_evaluate_accepts_flying_fluid_options(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--task",
            "flying_x",
            "--fluid-density",
            "0.9",
            "--fluid-viscosity",
            "0.00002",
            "--fluid-shape",
            "none",
            "--fitness-gain-fraction",
            "0.8",
            "--fluid-coef",
            "0.1",
            "0.2",
            "0.3",
            "0.4",
            "0.5",
        ],
    )
    args = evaluate.parse_args()
    assert args.fluid_density == pytest.approx(0.9)
    assert args.fluid_viscosity == pytest.approx(0.00002)
    assert args.fluid_shape == "none"
    assert args.fluid_coef == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
    assert args.fitness_gain_fraction == pytest.approx(0.8)


def test_evaluate_no_longer_records_video():
    with pytest.raises(SystemExit, match="2"):
        evaluate.parse_args(["--video"])


def test_evolve_accepts_walking_and_thread_override(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["evolve.py", "--task", "walking_x", "--threads", "3"]
    )
    args = evolve_cli.parse_args()
    assert args.task == "walking_x"
    assert args.threads == 3
    assert not args.self_collision


def test_evaluate_forwards_upright_weight_only_when_enabled(monkeypatch):
    observed = []

    def record_config(_genotype, definition):
        observed.append(definition.config.upright_weight)
        return type(
            "Result",
            (),
            {
                "disqualified": True,
                "build_failed": False,
                "failure_reason": "stop",
                "fitness": 0.0,
            },
        )()

    monkeypatch.setattr(evaluate, "load_genotype_from_json", lambda _path: object())
    monkeypatch.setattr(evaluate, "evaluate_task", record_config)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--task", "walking_x"])
    evaluate.main()
    monkeypatch.setattr(
        sys, "argv", ["evaluate.py", "--task", "walking_x", "--upright-error"]
    )
    evaluate.main()

    assert observed == [0.0, pytest.approx(evaluate.DEFAULT_UPRIGHT_ERROR_WEIGHT)]


def test_evaluate_and_evolve_accept_away_tasks(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--task", "swimming_away"])
    assert evaluate.parse_args().task == "swimming_away"
    assert isinstance(evaluate.parse_args().task, str)

    monkeypatch.setattr(sys, "argv", ["evolve.py", "--task", "walking_away"])
    assert evolve_cli.parse_args().task == "walking_away"
    assert isinstance(evolve_cli.parse_args().task, str)


def test_evaluate_and_evolve_accept_flying_tasks(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--task", "flying_x"])
    assert evaluate.parse_args().task == "flying_x"

    monkeypatch.setattr(sys, "argv", ["evolve.py", "--task", "flying_away"])
    assert evolve_cli.parse_args().task == "flying_away"


def test_evolve_accepts_gradual_gravity_change_for_flying(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evolve.py",
            "--task",
            "flying_x",
            "--gradual-gravity-change=-1.0,-9.81",
        ],
    )
    args = evolve_cli.parse_args()
    assert args.gradual_gravity_change == pytest.approx((-1.0, -9.81))


def test_evolve_rejects_gradual_gravity_change_for_non_flying(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["evolve.py", "--gradual-gravity-change=-1.0,-9.81"],
    )
    with pytest.raises(SystemExit, match="2"):
        evolve_cli.parse_args()
    assert re.search("--gradual-gravity-change is only supported for flying tasks", capsys.readouterr().err)


def test_gradual_gravity_change_interpolates_across_generations():
    assert evolve_cli._gravity_for_generation((-1.0, -9.0), 0, 4) == pytest.approx(-1.0)
    assert evolve_cli._gravity_for_generation((-1.0, -9.0), 2, 4) == pytest.approx(-5.0)
    assert evolve_cli._gravity_for_generation((-1.0, -9.0), 4, 4) == pytest.approx(-9.0)


def test_generation_environment_applies_gradual_flying_gravity():
    args = argparse.Namespace(
        gradual_gravity_change=(-1.0, -9.0),
        generations=4,
    )
    definition = task_definition("flying_x")
    generation_environment = evolve_cli._environment_for_generation(
        definition.environment, args, 2
    )
    assert generation_environment.gravity == pytest.approx(-5.0)
    assert definition.environment.gravity == pytest.approx(-9.81)


def test_evolve_accepts_flying_fluid_options(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evolve.py",
            "--task",
            "flying_away",
            "--fluid-density",
            "0.9",
            "--fluid-viscosity",
            "0.00002",
            "--fluid-shape",
            "none",
            "--fitness-gain-fraction",
            "0.8",
            "--fluid-coef",
            "0.1",
            "0.2",
            "0.3",
            "0.4",
            "0.5",
        ],
    )
    args = evolve_cli.parse_args()
    assert args.fluid_density == pytest.approx(0.9)
    assert args.fluid_viscosity == pytest.approx(0.00002)
    assert args.fluid_shape == "none"
    assert args.fluid_coef == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
    assert args.fitness_gain_fraction == pytest.approx(0.8)


@pytest.mark.parametrize("module", [evaluate, evolve_cli])


def test_cli_leaves_task_dependent_options_unset(monkeypatch, module):
    monkeypatch.setattr(sys, "argv", [module.__file__, "--task", "flying_x"])

    args = module.parse_args()

    assert args.body_count_weight is None
    assert args.volume_weight is None
    assert args.volume_penalty_cutoff is None
    assert args.min_body_volume is None
    assert args.min_total_volume is None
    assert args.max_volume is None
    assert args.fluid_density is None
    assert args.fluid_viscosity is None
    assert args.fluid_shape is None
    assert args.fluid_coef is None
    assert args.fitness_gain_fraction is None


@pytest.mark.parametrize(
    "task", ["swimming_x", "walking_x", "flying_x"]
)
def test_task_definition_allows_explicit_fluid_overrides_for_every_environment(task):
    definition = make_definition(task)
    defaults = definition._default_environment
    args = argparse.Namespace(
        fluid_density=0.8,
        fluid_viscosity=0.00003,
        fluid_shape="none",
        fluid_coef=(0.1, 0.2, 0.3, 0.4, 0.5),
    )

    returned = definition.args_override(args)

    assert returned is definition
    assert definition.environment.fluid_density == pytest.approx(0.8)
    assert definition.environment.fluid_viscosity == pytest.approx(0.00003)
    assert definition.environment.fluid_shape == "none"
    assert definition.environment.fluid_coef == pytest.approx(
        (0.1, 0.2, 0.3, 0.4, 0.5)
    )
    definition.args_override(argparse.Namespace())
    assert definition.environment is not defaults
    assert definition.environment == defaults


def test_task_definition_config_overrides_reset_to_task_defaults():
    definition = make_definition("swimming_x")
    defaults = SwimmingEvaluationConfig()

    definition.args_override(
        argparse.Namespace(duration=3.0, body_count_weight=0.25)
    )
    assert definition.config.episode_seconds == pytest.approx(3.0)
    assert definition.config.body_count_weight == pytest.approx(0.25)

    definition.args_override(argparse.Namespace())
    assert definition.config == defaults


@pytest.mark.parametrize("module", [evaluate, evolve_cli])


def test_cli_preserves_explicit_task_parameter_overrides(monkeypatch, module):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            module.__file__,
            "--task",
            "flying_x",
            "--body-count-weight",
            "0.2",
            "--volume-weight",
            "0.3",
            "--volume-penalty-cutoff",
            "0.4",
            "--min-body-volume",
            "0.0005",
            "--min-total-volume",
            "0.0006",
            "--max-volume",
            "1.6",
            "--fluid-density",
            "1.7",
            "--fluid-viscosity",
            "0.00008",
            "--fluid-shape",
            "ellipsoid",
            "--fitness-gain-fraction",
            "0.8",
            "--fluid-coef",
            "0.6",
            "0.7",
            "0.8",
            "0.9",
            "1.0",
        ],
    )

    args = module.parse_args()

    assert args.body_count_weight == pytest.approx(0.2)
    assert args.volume_weight == pytest.approx(0.3)
    assert args.volume_penalty_cutoff == pytest.approx(0.4)
    assert args.min_body_volume == pytest.approx(0.0005)
    assert args.min_total_volume == pytest.approx(0.0006)
    assert args.max_volume == pytest.approx(1.6)
    assert args.fluid_density == pytest.approx(1.7)
    assert args.fluid_viscosity == pytest.approx(0.00008)
    assert args.fluid_shape == "ellipsoid"
    assert args.fitness_gain_fraction == pytest.approx(0.8)
    assert args.fluid_coef == pytest.approx([0.6, 0.7, 0.8, 0.9, 1.0])


def test_evolve_accepts_volume_weight(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["evolve.py", "--volume-weight", "0.25"]
    )
    assert evolve_cli.parse_args().volume_weight == pytest.approx(0.25)


def test_evolve_accepts_volume_cutoff_and_maximum(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evolve.py",
            "--volume-penalty-cutoff",
            "0.2",
            "--max-volume",
            "2.0",
        ],
    )
    args = evolve_cli.parse_args()
    assert args.volume_penalty_cutoff == pytest.approx(0.2)
    assert args.max_volume == pytest.approx(2.0)


def test_evolve_accepts_self_collision(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evolve.py", "--self-collision"])
    assert evolve_cli.parse_args().self_collision


def test_evolve_accepts_disallow_collision(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evolve.py", "--disallow-collision"])
    args = evolve_cli.parse_args()
    assert args.disallow_collision
    assert not args.self_collision


def test_evolve_accepts_latest_best_only(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evolve.py", "--latest-best-only"])
    assert evolve_cli.parse_args().latest_best_only


def test_evolve_accepts_save_genotype_every_n(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["evolve.py", "--save-genotype-every-N", "5"]
    )
    assert evolve_cli.parse_args().save_genotype_every_n == 5


def test_evolve_rejects_nonpositive_save_genotype_every_n(monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["evolve.py", "--save-genotype-every-N", "0"]
    )
    with pytest.raises(SystemExit, match="2"):
        evolve_cli.parse_args()
    assert re.search("--save-genotype-every-N must be at least 1", capsys.readouterr().err)


def test_evolve_rejects_save_genotype_interval_with_latest_best_only(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["evolve.py", "--latest-best-only", "--save-genotype-every-N", "5"],
    )
    with pytest.raises(SystemExit, match="2"):
        evolve_cli.parse_args()
    assert re.search("--save-genotype-every-N conflicts with --latest-best-only", capsys.readouterr().err)


def test_evolve_save_genotype_interval_controls_generation_history():
    assert evolve_cli._should_save_generation_history(0, 3, latest_best_only=False)
    assert not evolve_cli._should_save_generation_history(1, 3, latest_best_only=False)
    assert evolve_cli._should_save_generation_history(3, 3, latest_best_only=False)
    assert not evolve_cli._should_save_generation_history(3, 3, latest_best_only=True)


def test_evolve_accepts_record_mutant_type(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evolve.py", "--record-mutant-type"])
    assert evolve_cli.parse_args().record_mutant_type


def test_evolve_disables_record_mutant_type_by_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evolve.py"])
    assert not evolve_cli.parse_args().record_mutant_type


def test_evolve_defaults_topology_mutation_rate_floor(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evolve.py"])
    assert evolve_cli.parse_args().topology_mutation_rate_min == pytest.approx(0.05)


def test_evolve_accepts_topology_mutation_rate_floor(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["evolve.py", "--topology-mutation-rate-min", "0.12"],
    )
    assert evolve_cli.parse_args().topology_mutation_rate_min == pytest.approx(0.12)


def test_evolve_rejects_invalid_topology_mutation_rate_floor(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["evolve.py", "--topology-mutation-rate-min", "1.1"],
    )
    with pytest.raises(SystemExit, match="2"):
        evolve_cli.parse_args()
    assert re.search("--topology-mutation-rate-min must be between 0 and 1", capsys.readouterr().err)


def test_evolve_accepts_disallow_topology_mutations(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["evolve.py", "--disallow-topology-mutations"],
    )
    assert evolve_cli.parse_args().disallow_topology_mutations


def test_evolve_allows_topology_mutations_by_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evolve.py"])
    assert not evolve_cli.parse_args().disallow_topology_mutations


def test_evolve_accepts_disallow_root_mutation(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evolve.py", "--disallow-root-mutation"])
    assert evolve_cli.parse_args().disallow_root_mutation


def test_evolve_allows_root_mutation_by_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evolve.py"])
    assert not evolve_cli.parse_args().disallow_root_mutation


def test_evolve_accepts_allow_slide_joint(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evolve.py", "--allow-slide-joint"])
    assert evolve_cli.parse_args().allow_slide_joint


def test_evolve_disallows_slide_joint_by_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evolve.py"])
    assert not evolve_cli.parse_args().allow_slide_joint


def test_mutate_accepts_allow_slide_joint():
    args = mutate_cli.parse_args(["--output", "out.json", "--allow-slide-joint"])
    assert args.allow_slide_joint


def test_evolve_records_generation_history_by_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evolve.py"])
    assert not evolve_cli.parse_args().latest_best_only


def test_evolve_rejects_nonpositive_threads(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["evolve.py", "--threads", "0"])
    with pytest.raises(SystemExit, match="2"):
        evolve_cli.parse_args()
    assert re.search("--threads must be at least 1", capsys.readouterr().err)


def test_generation_progress_reports_volume_and_energy_without_best_ever_or_mean():
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    creature = make_creature(
        genotype, 2.5, body_count=7, total_volume=0.125, control_energy=3.75
    )
    summary = {
        "best_gene_count": 11,
        "best_body_count": 7,
        "build_failures": 1,
        "disqualifications": 2,
    }

    line = evolve_cli._format_generation_progress(4, creature, summary)

    assert line == (
        "gen=0004 best=2.500000 genes=11 bodies=7 volume=0.125000 "
        "energy=3.750000 failures=1 disqualified=2"
    )
    assert "best_ever=" not in line
    assert "mean=" not in line


def test_generation_progress_reports_mutant_type_counts_when_present():
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    creature = make_creature(
        genotype, 2.5, body_count=7, total_volume=0.125, control_energy=3.75
    )
    summary = {
        "best_gene_count": 11,
        "best_body_count": 7,
        "build_failures": 1,
        "disqualifications": 2,
        "fitter_mutants": 3,
        "neutral_mutants": 1,
        "less_fit_mutants": 4,
    }

    line = evolve_cli._format_generation_progress(4, creature, summary)

    assert line.endswith(" fitter=3 neutral=1 less_fit=4")


def test_default_run_directory_uses_task_name(monkeypatch):
    class FixedDateTime:
        @classmethod
        def now(cls):
            return cls()

        def strftime(self, _format):
            return "20260102_030405"

    monkeypatch.setattr(evolve_cli, "datetime", FixedDateTime)
    assert str(evolve_cli._default_run_dir("walking_x")).endswith(
        "runs/walking_x_20260102_030405"
    )


@pytest.mark.parametrize(
    ("parse_args", "expected_defaults"),
    [
        (
            evaluate.parse_args,
            (
                "default: swimming_x",
                "default: examples/example_genotype.json",
                "default: 10.0",
                "default: 500",
            ),
        ),
        (
            view_cli.parse_args,
            (
                "default: swimming_x",
                "default: examples/example_genotype.json",
                "default: interactive window",
                "default: 10.0",
                "default: 30",
                "default: 960",
                "default: 4096",
            ),
        ),
        (
            evolve_cli.parse_args,
            (
                "default: swimming_x",
                "default: examples/example_genotype.json",
                "default: runs/<task>_<timestamp>",
                "default: 100",
                f"default: {evolution.default_thread_count()}",
                "default: 10.0",
                "default: 500",
                "default: 0.05",
            ),
        ),
        (
            mutate_cli.parse_args,
            (
                "default: examples/example_genotype.json",
                "default: 1",
                "default: random",
                "default: 0.05",
            ),
        ),
        (
            build_cli.parse_args,
            (
                "default: examples/example_genotype.json",
                "default: generated_creature.xml",
                "default: swimming_x",
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


def test_cli_lists_subcommands(capsys):
    with pytest.raises(SystemExit, match="0"):
        cli.main(["--help"])
    help_text = capsys.readouterr().out
    for command in ("evolve", "evaluate", "mutate", "build"):
        assert command in help_text


def test_cli_requires_a_subcommand(capsys):
    with pytest.raises(SystemExit, match="2"):
        cli.main([])


def test_python_dash_m_runs_cli():
    completed = subprocess.run(
        [sys.executable, "-m", "evc3d", "build", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--output" in completed.stdout


def test_mutate_is_reproducible_with_seed(tmp_path):
    outputs = []
    for name, seed in (("a.json", "42"), ("b.json", "42"), ("c.json", "43")):
        output = tmp_path / name
        cli.main(["mutate", "--output", str(output), "--mutations", "3", "--seed", seed])
        outputs.append(output.read_text())

    assert outputs[0] == outputs[1]
    assert outputs[0] != outputs[2]
    assert outputs[0] != evolve_cli.DEFAULT_GENOTYPE_PATH.read_text()


def test_mutate_refuses_to_overwrite_source_genotype(tmp_path, capsys):
    source = tmp_path / "seed.json"
    source.write_text(evolve_cli.DEFAULT_GENOTYPE_PATH.read_text())

    with pytest.raises(SystemExit, match="2"):
        cli.main(["mutate", "--genotype", str(source), "--output", str(source)])
    assert "--output must differ from --genotype" in capsys.readouterr().err


def test_mutate_forwards_mutation_options(monkeypatch, tmp_path):
    calls = []

    def record_mutation(self, num_mutations=1, rng=None, **options):
        calls.append((num_mutations, options))
        return self

    monkeypatch.setattr(Genotype, "mutation", record_mutation)
    cli.main([
        "mutate", "--output", str(tmp_path / "out.json"), "--mutations", "2",
        "--allow-slide-joint", "--disallow-topology-mutations",
        "--disallow-root-mutation", "--topology-mutation-rate-min", "0.2",
    ])

    assert calls == [
        (1, {
            "allow_topology_mutations": False,
            "allow_slide_joint": True,
            "allow_root_mutation": False,
            "topology_mutation_rate_min": 0.2,
        }),
    ] * 2


def test_build_default_matches_plain_phenotype_builder(tmp_path):
    output = tmp_path / "creature.xml"
    cli.main(["build", "--output", str(output)])

    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    expected = PhenotypeBuilder(genotype, make_definition(max_body_count=500)).build()
    assert output.read_text() == expected


def test_build_writes_task_environment(tmp_path):
    output = tmp_path / "walker.xml"
    cli.main(["build", "--task", "walking_x", "--output", str(output)])

    option = ET.parse(output).getroot().find("option")
    assert option.get("gravity") == "0 0 -9.81"
    assert option.get("density") == "0"


def test_resolving_cli_options_does_not_modify_the_task_registry():
    shared = task_definition("flying_x")
    environment, config = shared.environment, shared.config

    args = evolve_cli.parse_args(
        ["--task", "flying_x", "--fluid-density", "0.5", "--duration", "2"]
    )
    resolved = resolve_task_definition(args)

    assert resolved is not shared
    assert resolved.environment.fluid_density == pytest.approx(0.5)
    assert resolved.config.episode_seconds == pytest.approx(2.0)
    assert shared.environment is environment
    assert shared.config is config



@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["evaluate", "--duration", "0"], "--duration must be greater than zero"),
        (["evolve", "--max-body-count", "0"], "--max-body-count must be at least 1"),
        (["build", "--max-body-count", "0"], "--max-body-count must be at least 1"),
        (["evaluate", "--volume-weight", "-1"], "--volume-weight must be non-negative"),
        (
            ["evolve", "--max-volume", "0.1", "--volume-penalty-cutoff", "0.5"],
            "--max-volume must be greater than --volume-penalty-cutoff",
        ),
        (["evaluate", "--max-volume", "0.05"], "--max-volume must be greater than --volume-penalty-cutoff"),
        (["evaluate", "--task", "flying_x", "--fitness-gain-fraction", "2"], "--fitness-gain-fraction must be between 0 and 1"),
        (["view", "--fluid-density", "-1"], "--fluid-density must be non-negative"),
        (["view", "--mp4", "x.mp4", "--duration", "-2"], "--duration must be greater than zero"),
        (["view", "--duration", "-2"], "--duration only applies with --mp4"),
    ],
)
def test_cli_reports_config_errors_with_option_names(argv, message, capsys):
    with pytest.raises(SystemExit, match="2"):
        cli.parse_args(argv)
    assert message in capsys.readouterr().err
