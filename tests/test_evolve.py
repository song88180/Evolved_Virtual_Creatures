from concurrent.futures import ProcessPoolExecutor
import sys
import xml.etree.ElementTree as ET

import pytest

import evaluate
import evolve as evolve_cli
from evol_virtual_creature import evolve as evolve_lib
from evol_virtual_creature.evaluation import (
    SwimmingEvaluationConfig,
    WalkingEvaluationConfig,
)
import generate_model


def test_default_thread_count_uses_half_available_affinity(monkeypatch):
    monkeypatch.setattr(
        evolve_lib.os, "sched_getaffinity", lambda _pid: set(range(10))
    )
    assert evolve_lib.default_thread_count() == 5


def test_default_thread_count_has_minimum_of_one(monkeypatch):
    monkeypatch.setattr(evolve_lib.os, "sched_getaffinity", lambda _pid: {0})
    assert evolve_lib.default_thread_count() == 1


def test_evaluate_defaults_to_swimming_without_self_collision(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate.py"])
    args = evaluate.parse_args()
    assert args.task == "swimming"
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


def test_evaluate_accepts_shadow_rendering_options(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--shadowclip", "0.75", "--spotlight"],
    )
    args = evaluate.parse_args()
    assert args.shadowclip == pytest.approx(0.75)
    assert args.spotlight


def test_evaluate_rejects_nonpositive_shadowclip(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--shadowclip", "0"])
    with pytest.raises(SystemExit, match="2"):
        evaluate.parse_args()

def test_evolve_accepts_walking_and_thread_override(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["evolve.py", "--task", "walking", "--threads", "3"]
    )
    args = evolve_cli.parse_args()
    assert args.task == "walking"
    assert args.threads == 3
    assert not args.self_collision


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


def test_evolve_records_generation_history_by_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evolve.py"])
    assert not evolve_cli.parse_args().latest_best_only


def test_evolve_rejects_nonpositive_threads(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evolve.py", "--threads", "0"])
    with pytest.raises(ValueError, match="--threads must be at least 1"):
        evolve_cli.parse_args()


@pytest.mark.parametrize(
    "config",
    [
        SwimmingEvaluationConfig(episode_seconds=0.02),
        WalkingEvaluationConfig(episode_seconds=0.02, settle_seconds=0.0),
    ],
)
def test_evaluate_population_runs_in_processes_and_preserves_order(config):
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    population = [genotype, genotype]
    with ProcessPoolExecutor(max_workers=2) as executor:
        evaluated = evolve_lib._evaluate_population(population, config, executor)
    assert len(evaluated) == len(population)
    assert evaluated[0].fitness == evaluated[1].fitness


def test_saved_best_xml_preserves_walking_self_collision(tmp_path):
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    path = tmp_path / "best.xml"
    evolve_lib._write_best_xml(
        path,
        genotype,
        WalkingEvaluationConfig(episode_seconds=0.02, self_collision=True),
    )
    option = ET.fromstring(path.read_text()).find("option")
    assert option is not None
    assert option.get("gravity") == "0 0 -9.81"
    default_geom = ET.fromstring(path.read_text()).find("./default/geom")
    assert default_geom is not None
    assert default_geom.get("contype") == "2"
    assert default_geom.get("conaffinity") == "3"


def test_save_generation_best_can_skip_generation_history(tmp_path):
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    creature = evolve_lib.EvaluatedCreature(
        genotype=genotype,
        fitness=1.0,
        metrics={"fitness": 1.0},
    )

    evolve_lib._save_generation_best(
        tmp_path,
        generation=3,
        generation_best=creature,
        best_so_far=creature,
        config=SwimmingEvaluationConfig(episode_seconds=0.02),
        save_generation_history=False,
    )

    assert (tmp_path / "latest_best_genotype.json").is_file()
    assert (tmp_path / "best_genotype.json").is_file()
    assert not (tmp_path / "generation_bests").exists()


def test_save_generation_best_records_history_by_default(tmp_path):
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    creature = evolve_lib.EvaluatedCreature(
        genotype=genotype,
        fitness=1.0,
        metrics={"fitness": 1.0},
    )

    evolve_lib._save_generation_best(
        tmp_path,
        generation=3,
        generation_best=creature,
        best_so_far=creature,
        config=SwimmingEvaluationConfig(episode_seconds=0.02),
    )

    assert (
        tmp_path / "generation_bests" / "generation_0003.json"
    ).is_file()


def test_generation_summary_counts_disqualifications():
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    creature = evolve_lib.EvaluatedCreature(
        genotype=genotype,
        fitness=-1_000.0,
        metrics={"disqualified": True, "body_count": 1},
    )
    summary = evolve_lib.generation_summary(0, [creature], creature)
    assert summary["disqualifications"] == 1
    assert summary["build_failures"] == 0


def test_default_run_directory_uses_task_name(monkeypatch):
    class FixedDateTime:
        @classmethod
        def now(cls):
            return cls()

        def strftime(self, _format):
            return "20260102_030405"

    monkeypatch.setattr(evolve_cli, "datetime", FixedDateTime)
    assert str(evolve_cli._default_run_dir("walking")).endswith(
        "runs/walking_20260102_030405"
    )


@pytest.mark.parametrize(
    ("parse_args", "expected_defaults"),
    [
        (
            evaluate.parse_args,
            (
                "default: swimming",
                "default: examples/example_genotype.json",
                "default: 6.0",
                "default: disabled",
                "default: 30",
            ),
        ),
        (
            evolve_cli.parse_args,
            (
                "default: swimming",
                "default: examples/example_genotype.json",
                "default: runs/<task>_<timestamp>",
                "default: 100",
                f"default: {evolve_lib.default_thread_count()}",
                "default: 10.0",
                "default: 500",
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


def test_evaluate_accepts_disallow_collision(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--disallow-collision"])
    args = evaluate.parse_args()
    assert args.disallow_collision
    assert not args.self_collision


def test_evolve_accepts_disallow_collision(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evolve.py", "--disallow-collision"])
    args = evolve_cli.parse_args()
    assert args.disallow_collision
    assert not args.self_collision


def test_disallow_collision_enables_saved_self_contact_masks(tmp_path):
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    path = tmp_path / "best.xml"
    evolve_lib._write_best_xml(
        path,
        genotype,
        SwimmingEvaluationConfig(
            episode_seconds=0.02,
            self_collision=False,
            disallow_collision=True,
        ),
    )
    default_geom = ET.fromstring(path.read_text()).find("./default/geom")
    assert default_geom is not None
    assert default_geom.get("contype") == "2"
    assert default_geom.get("conaffinity") == "2"
