import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import random
import sys
import xml.etree.ElementTree as ET

import pytest

import evaluate
import evolve as evolve_cli
from evol_virtual_creature import evolve as evolve_lib
from evol_virtual_creature.evaluation import task_definition
from evol_virtual_creature.genes import ConnectionGene, NodeGene
from evol_virtual_creature.genotype import Genotype
from evol_virtual_creature.evolution_tasks.flying_away import FlyingAwayEvaluationConfig
from evol_virtual_creature.evolution_tasks.flying_x import FlyingEvaluationConfig
from evol_virtual_creature.evolution_tasks.flying_x import TASK_ENVIRONMENT as FLYING_ENVIRONMENT
from evol_virtual_creature.evolution_tasks.swimming_away import SwimmingAwayEvaluationConfig
from evol_virtual_creature.evolution_tasks.swimming_x import SwimmingEvaluationConfig
from evol_virtual_creature.evolution_tasks.walking_away import WalkingAwayEvaluationConfig
from evol_virtual_creature.evolution_tasks.walking_x import WalkingEvaluationConfig
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


def test_evolve_defaults_to_neural_control_mode(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evolve.py"])
    assert evolve_cli.parse_args().control_mode == "neural"


def test_evolve_accepts_sine_control_mode(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evolve.py", "--control-mode", "sine"])
    assert evolve_cli.parse_args().control_mode == "sine"


def test_validate_genotype_control_mode_accepts_match():
    genotype = Genotype(
        root="body",
        nodes={"body": NodeGene(name="body", size=(0.2, 0.2, 0.2))},
        control_mode="sine",
    )

    evolve_cli._validate_genotype_control_mode(genotype, "sine")


def test_validate_genotype_control_mode_rejects_mismatch():
    genotype = Genotype(
        root="body",
        nodes={"body": NodeGene(name="body", size=(0.2, 0.2, 0.2))},
        control_mode="sine",
    )

    with pytest.raises(ValueError, match="does not match --control-mode"):
        evolve_cli._validate_genotype_control_mode(genotype, "neural")


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


def test_evaluate_accepts_shadow_rendering_options(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--shadowsize",
            "8192",
            "--spotlight",
            "--camera-circle-around",
        ],
    )
    args = evaluate.parse_args()
    assert args.shadowsize == 8192
    assert args.spotlight
    assert args.camera_circle_around


def test_evaluate_rejects_nonpositive_shadowsize(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--shadowsize", "0"])
    with pytest.raises(SystemExit, match="2"):
        evaluate.parse_args()


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

    def record_config(_genotype, _definition, config, _environment):
        observed.append(config.upright_weight)
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


def test_evolve_rejects_gradual_gravity_change_for_non_flying(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["evolve.py", "--gradual-gravity-change=-1.0,-9.81"],
    )
    with pytest.raises(
        ValueError,
        match="--gradual-gravity-change is only supported for flying tasks",
    ):
        evolve_cli.parse_args()


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


def test_cli_uses_task_defaults_when_options_are_omitted(monkeypatch, module):
    class TaskDefaults:
        body_count_weight = 0.012
        volume_weight = 0.034
        volume_penalty_cutoff = 0.056
        min_body_volume = 0.000078
        min_total_volume = 0.00012
        max_volume = 0.9
        fluid_density = 0.8
        fluid_viscosity = 0.00003
        fluid_shape = "none"
        fluid_coef = (0.1, 0.2, 0.3, 0.4, 0.5)
        fitness_gain_fraction = 0.7
        environment = FLYING_ENVIRONMENT

    monkeypatch.setattr(
        module,
        "task_definition",
        lambda _task: argparse.Namespace(
            config_type=TaskDefaults,
            environment=replace(
                FLYING_ENVIRONMENT, fluid_density=0.8,
                fluid_viscosity=0.00003, fluid_shape="none",
                fluid_coef=(0.1, 0.2, 0.3, 0.4, 0.5),
            ),
        ),
    )
    monkeypatch.setattr(sys, "argv", [module.__file__, "--task", "flying_x"])

    args = module.parse_args()

    assert args.body_count_weight == pytest.approx(0.012)
    assert args.volume_weight == pytest.approx(0.034)
    assert args.volume_penalty_cutoff == pytest.approx(0.056)
    assert args.min_body_volume == pytest.approx(0.000078)
    assert args.min_total_volume == pytest.approx(0.00012)
    assert args.max_volume == pytest.approx(0.9)
    assert args.fluid_density == pytest.approx(0.8)
    assert args.fluid_viscosity == pytest.approx(0.00003)
    assert args.fluid_shape == "none"
    assert args.fluid_coef == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
    assert args.fitness_gain_fraction == pytest.approx(0.7)


@pytest.mark.parametrize("module", [evaluate, evolve_cli])


def test_cli_preserves_explicit_task_parameter_overrides(monkeypatch, module):
    class TaskDefaults:
        body_count_weight = 0.012
        volume_weight = 0.034
        volume_penalty_cutoff = 0.056
        min_body_volume = 0.000078
        min_total_volume = 0.00012
        max_volume = 0.9
        fluid_density = 0.8
        fluid_viscosity = 0.00003
        fluid_shape = "none"
        fluid_coef = (0.1, 0.2, 0.3, 0.4, 0.5)
        fitness_gain_fraction = 0.7
        environment = FLYING_ENVIRONMENT

    monkeypatch.setattr(
        module,
        "task_definition",
        lambda _task: argparse.Namespace(
            config_type=TaskDefaults,
            environment=replace(
                FLYING_ENVIRONMENT, fluid_density=0.8,
                fluid_viscosity=0.00003, fluid_shape="none",
                fluid_coef=(0.1, 0.2, 0.3, 0.4, 0.5),
            ),
        ),
    )
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


def test_evolve_rejects_nonpositive_save_genotype_every_n(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["evolve.py", "--save-genotype-every-N", "0"]
    )
    with pytest.raises(
        ValueError, match="--save-genotype-every-N must be at least 1"
    ):
        evolve_cli.parse_args()


def test_evolve_rejects_save_genotype_interval_with_latest_best_only(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["evolve.py", "--latest-best-only", "--save-genotype-every-N", "5"],
    )
    with pytest.raises(
        ValueError,
        match="--save-genotype-every-N conflicts with --latest-best-only",
    ):
        evolve_cli.parse_args()


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


def test_evolve_rejects_invalid_topology_mutation_rate_floor(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["evolve.py", "--topology-mutation-rate-min", "1.1"],
    )
    with pytest.raises(
        ValueError, match="--topology-mutation-rate-min must be between 0 and 1"
    ):
        evolve_cli.parse_args()


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


def test_generate_model_accepts_allow_slide_joint(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["generate_model.py", "--allow-slide-joint"])
    assert generate_model.parse_args().allow_slide_joint


def test_evolve_records_generation_history_by_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evolve.py"])
    assert not evolve_cli.parse_args().latest_best_only


def test_evolve_rejects_nonpositive_threads(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evolve.py", "--threads", "0"])
    with pytest.raises(ValueError, match="--threads must be at least 1"):
        evolve_cli.parse_args()


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
        evaluated = evolve_lib._evaluate_population(
            population, task_definition(task), config, executor
        )
    assert len(evaluated) == len(population)
    assert evaluated[0].fitness == evaluated[1].fitness


def test_saved_best_xml_preserves_walking_self_collision(tmp_path):
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    path = tmp_path / "best.xml"
    evolve_lib._write_best_xml(
        path,
        genotype,
        task_definition("walking_x"),
        WalkingEvaluationConfig(episode_seconds=0.02, self_collision=True),
    )
    option = ET.fromstring(path.read_text()).find("option")
    assert option is not None
    assert option.get("gravity") == "0 0 -9.81"
    default_geom = ET.fromstring(path.read_text()).find("./default/geom")
    assert default_geom is not None
    assert default_geom.get("contype") == "2"
    assert default_geom.get("conaffinity") == "3"


def test_saved_best_xml_preserves_swimming_away_physics(tmp_path):
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    path = tmp_path / "best.xml"
    evolve_lib._write_best_xml(
        path,
        genotype,
        task_definition("swimming_away"),
        SwimmingAwayEvaluationConfig(episode_seconds=0.02, self_collision=True),
    )
    root = ET.fromstring(path.read_text())
    option = root.find("option")
    default_geom = root.find("./default/geom")
    floor = root.find("./worldbody/geom[@name='floor']")

    assert option is not None
    assert option.get("gravity") == "0 0 0"
    assert option.get("density") == "1000"
    assert default_geom is not None
    assert default_geom.get("contype") == "2"
    assert floor is not None
    assert floor.get("contype") == "0"


def test_saved_best_xml_preserves_flying_physics(tmp_path):
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    path = tmp_path / "best.xml"
    evolve_lib._write_best_xml(
        path,
        genotype,
        task_definition("flying_x"),
        FlyingEvaluationConfig(episode_seconds=0.02, self_collision=True),
    )
    root = ET.fromstring(path.read_text())
    option = root.find("option")
    default_geom = root.find("./default/geom")
    floor = root.find("./worldbody/geom[@name='floor']")

    assert option is not None
    assert option.get("gravity") == "0 0 -9.81"
    assert option.get("density") == "1.225"
    assert option.get("viscosity") == "1.8e-05"
    assert default_geom is not None
    assert default_geom.get("fluidshape") == "ellipsoid"
    assert default_geom.get("fluidcoef") == "0.5 0.25 1.5 1.0 1.0"
    assert default_geom.get("contype") == "2"
    assert default_geom.get("conaffinity") == "3"
    assert floor is not None
    assert floor.get("contype") == "1"


def test_saved_best_xml_preserves_custom_flying_gravity(tmp_path):
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    path = tmp_path / "best.xml"
    evolve_lib._write_best_xml(
        path,
        genotype,
        task_definition("flying_x"),
        FlyingEvaluationConfig(episode_seconds=0.02),
        replace(FLYING_ENVIRONMENT, gravity=-3.5),
    )
    option = ET.fromstring(path.read_text()).find("option")
    assert option is not None
    assert option.get("gravity") == "0 0 -3.5"


def test_saved_best_xml_preserves_walking_away_physics(tmp_path):
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    path = tmp_path / "best.xml"
    evolve_lib._write_best_xml(
        path,
        genotype,
        task_definition("walking_away"),
        WalkingAwayEvaluationConfig(episode_seconds=0.02, self_collision=True),
    )
    root = ET.fromstring(path.read_text())
    option = root.find("option")
    default_geom = root.find("./default/geom")
    floor = root.find("./worldbody/geom[@name='floor']")

    assert option is not None
    assert option.get("gravity") == "0 0 -9.81"
    assert option.get("density") == "0"
    assert default_geom is not None
    assert default_geom.get("contype") == "2"
    assert default_geom.get("conaffinity") == "3"
    assert floor is not None
    assert floor.get("contype") == "1"


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
        definition=task_definition("swimming_x"),
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
        definition=task_definition("swimming_x"),
        config=SwimmingEvaluationConfig(episode_seconds=0.02),
    )

    assert (
        tmp_path / "generation_bests" / "generation_0003.json"
    ).is_file()


def test_generation_progress_reports_volume_and_energy_without_best_ever_or_mean():
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    creature = evolve_lib.EvaluatedCreature(
        genotype=genotype,
        fitness=2.5,
        metrics={
            "body_count": 7,
            "total_volume": 0.125,
            "control_energy": 3.75,
        },
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
    creature = evolve_lib.EvaluatedCreature(
        genotype=genotype,
        fitness=2.5,
        metrics={
            "body_count": 7,
            "total_volume": 0.125,
            "control_energy": 3.75,
        },
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


def test_generation_summary_counts_disqualifications_and_active_genes():
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    creature = evolve_lib.EvaluatedCreature(
        genotype=genotype,
        fitness=-1_000.0,
        metrics={"disqualified": True, "body_count": 1},
    )
    summary = evolve_lib.generation_summary(0, [creature], creature)
    expected_genes = len(genotype.nodes) + sum(
        len(node.children)
        for node in genotype.nodes.values()
    )
    assert summary["disqualifications"] == 1
    assert summary["build_failures"] == 0
    assert summary["best_gene_count"] == expected_genes
    assert summary["best_ever_gene_count"] == expected_genes


def test_generation_summary_omits_mutant_type_counts_by_default():
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    creature = evolve_lib.EvaluatedCreature(genotype, fitness=1.0, metrics={})

    summary = evolve_lib.generation_summary(0, [creature], creature)

    assert "fitter_mutants" not in summary
    assert "neutral_mutants" not in summary
    assert "less_fit_mutants" not in summary


def test_generation_summary_counts_mutant_types_against_previous_best():
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    evaluated = [
        evolve_lib.EvaluatedCreature(genotype, fitness=2.0, metrics={}),
        evolve_lib.EvaluatedCreature(genotype, fitness=1.5, metrics={}),
        evolve_lib.EvaluatedCreature(genotype, fitness=1.0, metrics={}),
        evolve_lib.EvaluatedCreature(
            genotype, fitness=3.0, metrics={"build_failed": True}
        ),
        evolve_lib.EvaluatedCreature(
            genotype, fitness=3.0, metrics={"disqualified": True}
        ),
        evolve_lib.EvaluatedCreature(genotype, fitness=5.0, metrics={}),
    ]
    records = [
        evolve_lib.MutantRecord(previous_best_fitness=1.5),
        evolve_lib.MutantRecord(previous_best_fitness=1.5),
        evolve_lib.MutantRecord(previous_best_fitness=1.5),
        evolve_lib.MutantRecord(previous_best_fitness=1.5),
        evolve_lib.MutantRecord(previous_best_fitness=1.5),
        None,
    ]

    summary = evolve_lib.generation_summary(1, evaluated, evaluated[0], records)

    assert summary["fitter_mutants"] == 1
    assert summary["neutral_mutants"] == 1
    assert summary["less_fit_mutants"] == 3


def test_next_population_with_mutant_records_marks_only_changed_children(monkeypatch):
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    evaluated = [
        evolve_lib.EvaluatedCreature(genotype, fitness=1.0, metrics={}),
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

    monkeypatch.setattr(evolve_lib, "mutate_quietly", mutate_second_child)

    population, records = evolve_lib.next_population_with_mutant_records(
        evaluated=evaluated,
        population_size=3,
        elite_count=1,
        tournament_size=1,
        min_mutations=1,
        max_mutations=1,
        rng=random.Random(1),
        previous_best_fitness=1.0,
    )

    assert len(population) == 3
    assert records[0] is None
    assert records[1] is None
    assert records[2] == evolve_lib.MutantRecord(previous_best_fitness=1.0)


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
                "default: disabled",
                "default: 30",
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
                f"default: {evolve_lib.default_thread_count()}",
                "default: 10.0",
                "default: 500",
                "default: 0.05",
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
        task_definition("swimming_x"),
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

    monkeypatch.setattr(evolve_lib, "mutate_quietly", record_mutation)
    evolve_lib.initial_population(
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
        evolve_lib.EvaluatedCreature(genotype, fitness=1.0, metrics={}),
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

    monkeypatch.setattr(evolve_lib, "mutate_quietly", record_mutation)
    evolve_lib.next_population(
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
        evolve_lib.EvaluatedCreature(genotype, fitness=1.0, metrics={}),
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

    monkeypatch.setattr(evolve_lib, "mutate_quietly", record_mutation)
    evolve_lib.initial_population(
        seed_genotype=genotype,
        population_size=2,
        initial_mutations=1,
        rng=random.Random(1),
        allow_slide_joint=True,
    )
    evolve_lib.next_population(
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
        evolve_lib.EvaluatedCreature(genotype, fitness=1.0, metrics={}),
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

    monkeypatch.setattr(evolve_lib, "mutate_quietly", record_mutation)
    evolve_lib.initial_population(
        seed_genotype=genotype,
        population_size=2,
        initial_mutations=1,
        rng=random.Random(1),
        allow_root_mutation=False,
    )
    evolve_lib.next_population(
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
