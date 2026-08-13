import importlib
from dataclasses import replace
import math
import subprocess
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

import evol_virtual_creature.evaluation as evaluation
from evol_virtual_creature.control import apply_controller
from evol_virtual_creature.evaluation import (
    NUMERICAL_INSTABILITY_REASON,
    INITIAL_FLOOR_OVERLAP_REASON,
    MAXIMUM_CREATURE_HEIGHT_REASON,
    WALKING_CENTER_HEIGHT_DROP_REASON,
    load_task_definition,
    task_definition,
    task_definition_for_config,
    task_names,
    _has_nonparent_self_collision,
    _run_flying_episode,
    _run_controlled_episode,
    _creature_volume,
    evaluate_task,
    initialize_model,
    simulation_failure_reason,
)
from evol_virtual_creature.evolution_tasks.shared import (
    DEFAULT_ENVIRONMENT,
    EnvironmentFamily,
    EvaluationConfig,
    ResultField,
    TASK_REGISTRY,
    TaskDefinition,
)
from evol_virtual_creature.evolution_tasks.flying_away import (
    FlyingAwayEvaluationConfig,
    FlyingAwayEvaluationResult,
    _episode_fitness as flying_away_episode_fitness,
    fitness_callback as flying_away_fitness,
)
from evol_virtual_creature.evolution_tasks.flying_x import (
    FlyingEvaluationConfig,
    FlyingEvaluationResult,
    _episode_fitness as flying_x_episode_fitness,
    fitness_callback as flying_x_fitness,
)
from evol_virtual_creature.evolution_tasks.swimming_away import (
    OriginDistanceEvaluationResult,
    SwimmingAwayEvaluationConfig,
    fitness_callback as swimming_away_fitness,
)
from evol_virtual_creature.evolution_tasks.swimming_x import (
    SwimmingEvaluationConfig,
    SwimmingEvaluationResult,
    fitness_callback as swimming_x_fitness,
)
from evol_virtual_creature.evolution_tasks.walking_away import (
    WalkingAwayEvaluationConfig,
    WalkingAwayEvaluationResult,
    fitness_callback as walking_away_fitness,
)
from evol_virtual_creature.evolution_tasks.walking_x import (
    WalkingEvaluationConfig,
    WalkingEvaluationResult,
    fitness_callback as walking_x_fitness,
)
from evol_virtual_creature.evolution_tasks.flying_away import TASK_ENVIRONMENT as FLYING_AWAY_ENVIRONMENT
from evol_virtual_creature.evolution_tasks.flying_x import TASK_ENVIRONMENT as FLYING_X_ENVIRONMENT
from evol_virtual_creature.evolution_tasks.swimming_away import TASK_ENVIRONMENT as SWIMMING_AWAY_ENVIRONMENT
from evol_virtual_creature.evolution_tasks.swimming_x import TASK_ENVIRONMENT as SWIMMING_X_ENVIRONMENT
from evol_virtual_creature.evolution_tasks.walking_away import TASK_ENVIRONMENT as WALKING_AWAY_ENVIRONMENT
from evol_virtual_creature.evolution_tasks.walking_x import TASK_ENVIRONMENT as WALKING_X_ENVIRONMENT
from evol_virtual_creature.genotype_io import build_genotype, load_genotype_from_json
from evol_virtual_creature.phenotype import PhenotypeBuilder


GENOTYPE_PATH = "examples/example_genotype.json"


def _reject_initialization(_model, _data, _config):
    return "custom initialization rejected model"


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"timestep": 0.0}, "timestep"),
        ({"fluid_density": -1.0}, "fluid_density"),
        ({"fluid_viscosity": -1.0}, "fluid_viscosity"),
        ({"fluid_coef": (1.0, 2.0)}, "fluid_coef"),
        ({"body_friction": (1.0, 2.0)}, "body_friction"),
        ({"initial_floor_clearance": -1.0}, "initial_floor_clearance"),
    ),
)
def test_environment_rejects_invalid_physics_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        replace(DEFAULT_ENVIRONMENT, **kwargs)


def test_task_registry_covers_every_task_and_config():
    expected = {
        "swimming_x": (
            SwimmingEvaluationConfig,
            SwimmingEvaluationResult,
            swimming_x_fitness,
            SWIMMING_X_ENVIRONMENT,
        ),
        "swimming_away": (
            SwimmingAwayEvaluationConfig,
            OriginDistanceEvaluationResult,
            swimming_away_fitness,
            SWIMMING_AWAY_ENVIRONMENT,
        ),
        "walking_x": (
            WalkingEvaluationConfig,
            WalkingEvaluationResult,
            walking_x_fitness,
            WALKING_X_ENVIRONMENT,
        ),
        "walking_away": (
            WalkingAwayEvaluationConfig,
            WalkingAwayEvaluationResult,
            walking_away_fitness,
            WALKING_AWAY_ENVIRONMENT,
        ),
        "flying_x": (
            FlyingEvaluationConfig,
            FlyingEvaluationResult,
            flying_x_fitness,
            FLYING_X_ENVIRONMENT,
        ),
        "flying_away": (
            FlyingAwayEvaluationConfig,
            FlyingAwayEvaluationResult,
            flying_away_fitness,
            FLYING_AWAY_ENVIRONMENT,
        ),
    }

    assert set(task_names()) == set(expected)
    for task, (config_type, result_type, fitness_callback, environment) in expected.items():
        definition = TASK_REGISTRY[task]
        assert definition.config_type is config_type
        assert definition.result_type is result_type
        assert definition.fitness_callback is fitness_callback
        assert callable(definition.failed_task_callback)
        assert definition.rollout_policy.episode in {"controlled", "flying"}
        assert definition.environment is environment
        assert definition.name == task
        assert task_definition_for_config(config_type()) is definition

    with pytest.raises(KeyError):
        task_definition("unknown_task")


def test_evaluation_module_lazily_loads_builtin_tasks():
    script = (
        "import evol_virtual_creature.evaluation as evaluation; "
        "from evol_virtual_creature.evolution_tasks import shared; "
        "assert not shared.TASK_REGISTRY; "
        "assert evaluation.task_names() == "
        "('swimming_x', 'swimming_away', 'walking_x', 'walking_away', "
        "'flying_x', 'flying_away')"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_evolution_tasks_package_import_has_no_registration_side_effect():
    script = (
        "import importlib; "
        "import evol_virtual_creature.evaluation; "
        "from evol_virtual_creature.evolution_tasks import shared; "
        "importlib.import_module('evol_virtual_creature.evolution_tasks'); "
        "assert not shared.TASK_REGISTRY"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize(
    "imports",
    (
        "import evol_virtual_creature.evaluation; "
        "import evol_virtual_creature.evolution_tasks.shared",
        "import evol_virtual_creature.evolution_tasks.shared; "
        "import evol_virtual_creature.evaluation",
    ),
)
def test_evaluation_and_shared_import_cleanly_in_either_order(imports):
    script = (
        f"{imports}; "
        "import evol_virtual_creature.evaluation as evaluation; "
        "from evol_virtual_creature.evolution_tasks import shared; "
        "assert not hasattr(evaluation, 'EvaluationConfig'); "
        "assert not hasattr(evaluation, 'ResultField'); "
        "assert not hasattr(evaluation, 'TaskDefinition'); "
        "assert not hasattr(evaluation, 'TASK_REGISTRY'); "
        "assert shared.EvaluationConfig; assert shared.ResultField; "
        "assert shared.TaskDefinition; assert shared.TASK_REGISTRY == {}"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_load_task_definition_matches_task_to_module_filename():
    definition = load_task_definition("swimming_x")
    module = importlib.import_module(
        "evol_virtual_creature.evolution_tasks.swimming_x"
    )
    assert definition is module.TASK_DEFINITION
    assert isinstance(definition, TaskDefinition)
    assert isinstance(definition.result_fields[0], ResultField)
    assert EvaluationConfig.__module__.endswith("evolution_tasks.shared")
    assert load_task_definition("swimming_x") is definition

    for invalid_task in ("turn-around", "tasks.walking_x", "../walking_x", "_shared"):
        with pytest.raises(KeyError, match="Unknown task"):
            load_task_definition(invalid_task)


def test_away_tasks_have_task_specific_result_types():
    assert WalkingAwayEvaluationResult is not WalkingEvaluationResult
    assert FlyingAwayEvaluationResult is not FlyingEvaluationResult


def test_task_physics_settings_are_distinct():
    genotype = load_genotype_from_json(GENOTYPE_PATH)
    swimming = ET.fromstring(
        PhenotypeBuilder(
            genotype, max_node=500, environment=SWIMMING_X_ENVIRONMENT
        ).build()
    )
    walking = ET.fromstring(
        PhenotypeBuilder(genotype, max_node=500, environment=WALKING_X_ENVIRONMENT).build()
    )
    swimming_away = ET.fromstring(
        PhenotypeBuilder(
            genotype, max_node=500, environment=SWIMMING_AWAY_ENVIRONMENT
        ).build()
    )
    walking_away = ET.fromstring(
        PhenotypeBuilder(genotype, max_node=500, environment=WALKING_AWAY_ENVIRONMENT).build()
    )
    flying = ET.fromstring(
        PhenotypeBuilder(genotype, max_node=500, environment=FLYING_X_ENVIRONMENT).build()
    )
    flying_away = ET.fromstring(
        PhenotypeBuilder(genotype, max_node=500, environment=FLYING_AWAY_ENVIRONMENT).build()
    )

    swimming_option = swimming.find("option")
    walking_option = walking.find("option")
    swimming_away_option = swimming_away.find("option")
    walking_away_option = walking_away.find("option")
    flying_option = flying.find("option")
    flying_away_option = flying_away.find("option")
    assert swimming_option is not None and walking_option is not None
    assert swimming_away_option is not None and walking_away_option is not None
    assert flying_option is not None and flying_away_option is not None
    assert swimming_option.get("gravity") == "0 0 0"
    assert swimming_option.get("density") == "1000"
    assert swimming_away_option.get("gravity") == "0 0 0"
    assert swimming_away_option.get("density") == "1000"
    assert walking_away_option.get("gravity") == "0 0 -9.81"
    assert walking_away_option.get("density") == "0"
    assert walking_option.get("gravity") == "0 0 -9.81"
    assert walking_option.get("density") == "0"
    assert walking_option.get("viscosity") == "0"
    assert flying_option.get("gravity") == "0 0 -9.81"
    assert flying_option.get("density") == "1.225"
    assert flying_option.get("viscosity") == "1.8e-05"
    assert flying_away_option.get("gravity") == "0 0 -9.81"
    assert flying_away_option.get("density") == "1.225"
    assert flying_away_option.get("viscosity") == "1.8e-05"

    swimming_geom = swimming.find("./default/geom")
    walking_geom = walking.find("./default/geom")
    flying_geom = flying.find("./default/geom")
    assert swimming_geom is not None and walking_geom is not None
    assert flying_geom is not None
    assert flying_geom.get("fluidshape") == "ellipsoid"
    assert flying_geom.get("fluidcoef") == "0.5 0.25 1.5 1.0 1.0"
    assert swimming_geom.get("contype") == "0"
    assert walking_geom.get("contype") == "2"
    assert walking_geom.get("conaffinity") == "1"
    walking_floor = walking.find("./worldbody/geom[@name=\"floor\"]")
    assert walking_floor is not None
    checker = walking.find("./asset/texture[@name=\"floor_checker\"]")
    floor_material = walking.find("./asset/material[@name=\"floor_material\"]")
    assert checker is not None
    assert checker.get("builtin") == "checker"
    assert checker.get("rgb1") == "0.172549 0.286275 0.521569"
    assert checker.get("rgb2") == "0.250980 0.458824 0.784314"
    assert floor_material is not None
    assert floor_material.get("texture") == "floor_checker"
    assert floor_material.get("reflectance") == "0"
    assert walking_floor.get("material") == "floor_material"
    creature_geom = walking.find("./worldbody/body/geom")
    assert creature_geom is not None
    assert creature_geom.get("rgba") == "0.933333 0.603922 0.301961 1"
    assert walking_floor.get("contype") == "1"
    assert walking_floor.get("conaffinity") == "2"
    flying_floor = flying.find('./worldbody/geom[@name="floor"]')
    assert flying_floor is not None
    assert flying_floor.get("contype") == "1"
    assert flying_floor.get("conaffinity") == "2"


def test_flying_fluid_settings_compile_in_mujoco():
    genotype = load_genotype_from_json(GENOTYPE_PATH)
    mjcf = PhenotypeBuilder(
        genotype, max_node=500, environment=FLYING_X_ENVIRONMENT
    ).build()
    model = mujoco.MjModel.from_xml_string(mjcf)

    assert model.opt.density == pytest.approx(1.225)
    assert model.opt.viscosity == pytest.approx(0.000018)


def test_flying_fluid_settings_can_be_overridden():
    genotype = load_genotype_from_json(GENOTYPE_PATH)
    root = ET.fromstring(
        PhenotypeBuilder(
            genotype,
            max_node=500,
            environment=replace(
                FLYING_X_ENVIRONMENT, fluid_density=0.9,
                fluid_viscosity=0.00002, fluid_shape="none",
                fluid_coef=(0.1, 0.2, 0.3, 0.4, 0.5),
            ),
        ).build()
    )
    option = root.find("option")
    default_geom = root.find("./default/geom")

    assert option is not None
    assert option.get("density") == "0.9"
    assert option.get("viscosity") == "2e-05"
    assert default_geom is not None
    assert default_geom.get("fluidshape") == "none"
    assert default_geom.get("fluidcoef") == "0.1 0.2 0.3 0.4 0.5"


def _collision_enabled(model, first_geom, second_geom):
    return bool(
        model.geom_contype[first_geom] & model.geom_conaffinity[second_geom]
        or model.geom_contype[second_geom] & model.geom_conaffinity[first_geom]
    )


def _creature_geom_ids(model):
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    return [geom_id for geom_id in range(model.ngeom) if geom_id != floor_id]


def _direct_parent_and_nonparent_pairs(model):
    geom_ids = _creature_geom_ids(model)
    parent_pair = None
    nonparent_pair = None
    for index, first_geom in enumerate(geom_ids):
        first_body = model.geom_bodyid[first_geom]
        for second_geom in geom_ids[index + 1:]:
            second_body = model.geom_bodyid[second_geom]
            directly_related = (
                model.body_parentid[first_body] == second_body
                or model.body_parentid[second_body] == first_body
            )
            if directly_related and parent_pair is None:
                parent_pair = (first_geom, second_geom)
            elif not directly_related and nonparent_pair is None:
                nonparent_pair = (first_geom, second_geom)
    assert parent_pair is not None
    assert nonparent_pair is not None
    return parent_pair, nonparent_pair


def test_self_collision_masks_and_parent_filtering():
    genotype = load_genotype_from_json(GENOTYPE_PATH)

    walking_without_self_collision = mujoco.MjModel.from_xml_string(
        PhenotypeBuilder(
            genotype,
            max_node=500,
            environment=WALKING_X_ENVIRONMENT,
            self_collision=False,
        ).build()
    )
    floor_id = mujoco.mj_name2id(
        walking_without_self_collision, mujoco.mjtObj.mjOBJ_GEOM, "floor"
    )
    creature_geom = _creature_geom_ids(walking_without_self_collision)[0]
    _, nonparent_pair = _direct_parent_and_nonparent_pairs(
        walking_without_self_collision
    )
    assert _collision_enabled(
        walking_without_self_collision, floor_id, creature_geom
    )
    assert not _collision_enabled(
        walking_without_self_collision, *nonparent_pair
    )

    walking_with_self_collision = mujoco.MjModel.from_xml_string(
        PhenotypeBuilder(
            genotype,
            max_node=500,
            environment=WALKING_X_ENVIRONMENT,
            self_collision=True,
        ).build()
    )
    parent_pair, nonparent_pair = _direct_parent_and_nonparent_pairs(
        walking_with_self_collision
    )
    assert _collision_enabled(walking_with_self_collision, *parent_pair)
    assert _collision_enabled(walking_with_self_collision, *nonparent_pair)
    assert not (
        walking_with_self_collision.opt.disableflags
        & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    )

    swimming_with_self_collision = mujoco.MjModel.from_xml_string(
        PhenotypeBuilder(
            genotype,
            max_node=500,
            environment=SWIMMING_X_ENVIRONMENT,
            self_collision=True,
        ).build()
    )
    floor_id = mujoco.mj_name2id(
        swimming_with_self_collision, mujoco.mjtObj.mjOBJ_GEOM, "floor"
    )
    creature_geom = _creature_geom_ids(swimming_with_self_collision)[0]
    _, nonparent_pair = _direct_parent_and_nonparent_pairs(
        swimming_with_self_collision
    )
    assert not _collision_enabled(
        swimming_with_self_collision, floor_id, creature_geom
    )
    assert _collision_enabled(swimming_with_self_collision, *nonparent_pair)


def test_both_tasks_return_finite_results():
    genotype = load_genotype_from_json(GENOTYPE_PATH)
    swimming = evaluate_task(
        genotype, SwimmingEvaluationConfig(episode_seconds=0.02)
    )
    walking = evaluate_task(
        genotype,
        WalkingEvaluationConfig(episode_seconds=0.02, settle_seconds=0.02),
    )
    swimming_away = evaluate_task(
        genotype, SwimmingAwayEvaluationConfig(episode_seconds=0.02)
    )
    walking_away = evaluate_task(
        genotype,
        WalkingAwayEvaluationConfig(episode_seconds=0.02, settle_seconds=0.02),
    )
    flying = evaluate_task(
        genotype, FlyingEvaluationConfig(episode_seconds=0.02)
    )
    flying_away = evaluate_task(
        genotype, FlyingAwayEvaluationConfig(episode_seconds=0.02)
    )
    assert not swimming.build_failed
    assert not walking.build_failed
    assert not swimming_away.build_failed
    assert not walking_away.build_failed
    assert not flying.build_failed
    assert not flying_away.build_failed
    assert swimming.simulated_seconds == 0.02
    assert walking.simulated_seconds == 0.02
    assert swimming_away.simulated_seconds == 0.02
    assert walking_away.simulated_seconds == 0.02
    assert flying.simulated_seconds == 0.02
    assert flying_away.simulated_seconds == 0.02
    assert swimming_away.origin_distance >= 0.0
    assert swimming_away.average_origin_speed >= 0.0
    assert walking_away.origin_distance >= 0.0
    assert walking_away.average_origin_speed >= 0.0
    assert walking.mean_upright_error >= 0.0
    assert walking.height_loss >= 0.0
    assert walking_away.mean_upright_error >= 0.0
    assert walking_away.height_loss >= 0.0
    assert flying.height_loss >= 0.0
    assert flying_away.height_loss >= 0.0
    assert flying_away.origin_distance >= 0.0


def test_controlled_episode_can_measure_horizontal_origin_distance():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.5" gravity="0 0 0"/>
          <worldbody>
            <body name="root">
              <freejoint/>
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    config = WalkingAwayEvaluationConfig(episode_seconds=1.0)

    class Builder:
        actuator_controllers = []

    data = mujoco.MjData(model)
    data.qvel[:3] = (3.0, 4.0, 12.0)
    horizontal_metrics = _run_controlled_episode(
        model,
        data,
        Builder(),
        config,
        horizontal_origin_distance=True,
    )

    data = mujoco.MjData(model)
    data.qvel[:3] = (3.0, 4.0, 12.0)
    spatial_metrics = _run_controlled_episode(model, data, Builder(), config)

    assert not isinstance(horizontal_metrics, str)
    assert not isinstance(spatial_metrics, str)
    assert horizontal_metrics["origin_distance"] == pytest.approx(2.5)
    assert horizontal_metrics["average_origin_speed"] == pytest.approx(2.5)
    assert spatial_metrics["origin_distance"] == pytest.approx(6.5)


def test_controlled_episode_reports_drift_speeds(monkeypatch):
    from evol_virtual_creature import evaluation

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.5" gravity="0 0 0"/>
          <worldbody>
            <body name="root">
              <freejoint/>
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )

    class Builder:
        actuator_controllers = []

    centers = iter([np.array((0.0, 0.0, 0.0)), np.array((0.0, 8.0, 12.0))])
    monkeypatch.setattr(
        evaluation,
        "_creature_center_of_mass",
        lambda *_args: next(centers),
    )
    metrics = _run_controlled_episode(
        model,
        mujoco.MjData(model),
        Builder(),
        SwimmingEvaluationConfig(episode_seconds=2.0),
    )

    assert not isinstance(metrics, str)
    assert metrics["sideways_drift_speed"] == pytest.approx(4.0)
    assert metrics["vertical_drift_speed"] == pytest.approx(6.0)


def test_controlled_episode_initializes_center_of_mass_before_scoring():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.5" gravity="0 0 0"/>
          <worldbody>
            <body name="root">
              <freejoint/>
              <geom type="sphere" size="0.1"/>
              <body name="offset" pos="100 0 0">
                <geom type="sphere" size="0.1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )

    class Builder:
        actuator_controllers = []

    metrics = _run_controlled_episode(
        model,
        mujoco.MjData(model),
        Builder(),
        SwimmingEvaluationConfig(episode_seconds=1.0),
    )

    assert not isinstance(metrics, str)
    assert metrics["origin_distance"] == pytest.approx(0.0)
    assert metrics["forward_distance"] == pytest.approx(0.0)
    assert metrics["vertical_drift_speed"] == pytest.approx(0.0)


def test_controlled_episode_uses_center_of_mass_for_distance(monkeypatch):
    from evol_virtual_creature import evaluation

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.5" gravity="0 0 0"/>
          <worldbody>
            <body name="root">
              <freejoint/>
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )

    class Builder:
        actuator_controllers = []

    centers = iter([
        np.array((1.0, 2.0, 3.0)),
        np.array((4.0, 6.0, 15.0)),
        np.array((1.0, 2.0, 3.0)),
        np.array((4.0, 6.0, 15.0)),
    ])
    monkeypatch.setattr(
        evaluation,
        "_creature_center_of_mass",
        lambda *_args: next(centers),
    )

    horizontal_metrics = _run_controlled_episode(
        model,
        mujoco.MjData(model),
        Builder(),
        WalkingAwayEvaluationConfig(episode_seconds=1.0),
        horizontal_origin_distance=True,
    )
    spatial_metrics = _run_controlled_episode(
        model,
        mujoco.MjData(model),
        Builder(),
        SwimmingEvaluationConfig(episode_seconds=1.0),
    )

    assert not isinstance(horizontal_metrics, str)
    assert not isinstance(spatial_metrics, str)
    assert horizontal_metrics["forward_distance"] == pytest.approx(3.0)
    assert horizontal_metrics["origin_distance"] == pytest.approx(5.0)
    assert horizontal_metrics["average_origin_speed"] == pytest.approx(5.0)
    assert spatial_metrics["origin_distance"] == pytest.approx(13.0)
    assert spatial_metrics["vertical_drift_speed"] == pytest.approx(12.0)


def test_controlled_episode_penalizes_center_of_mass_height_loss(monkeypatch):
    from evol_virtual_creature import evaluation

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.5" gravity="0 0 0"/>
          <worldbody>
            <body name="root">
              <freejoint/>
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )

    class Builder:
        actuator_controllers = []

    centers = iter([np.array((0.0, 0.0, 2.0)), np.array((0.0, 0.0, 1.25))])
    monkeypatch.setattr(
        evaluation,
        "_creature_center_of_mass",
        lambda *_args: next(centers),
    )

    metrics = _run_controlled_episode(
        model,
        mujoco.MjData(model),
        Builder(),
        WalkingEvaluationConfig(
            episode_seconds=0.5,
            min_center_height_fraction=0.5,
        ),
        root_body_name="root",
    )

    assert not isinstance(metrics, str)
    assert metrics["height_loss"] == pytest.approx(0.75)


def test_controlled_episode_rejects_large_center_of_mass_drop(monkeypatch):
    from evol_virtual_creature import evaluation

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.5" gravity="0 0 0"/>
          <worldbody>
            <body name="root">
              <freejoint/>
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )

    class Builder:
        actuator_controllers = []

    centers = iter([np.array((0.0, 0.0, 2.0)), np.array((0.0, 0.0, 0.9))])
    monkeypatch.setattr(
        evaluation,
        "_creature_center_of_mass",
        lambda *_args: next(centers),
    )

    result = _run_controlled_episode(
        model,
        mujoco.MjData(model),
        Builder(),
        WalkingEvaluationConfig(
            episode_seconds=0.5,
            min_center_height_fraction=0.5,
        ),
        root_body_name="root",
    )

    assert result == WALKING_CENTER_HEIGHT_DROP_REASON


def test_walking_rejects_creature_above_maximum_height():
    genotype = build_genotype(
        root="body",
        spec={
            "body": {
                "size": (0.1, 0.1, 6.0),
                "joint_type": "free",
            }
        },
    )

    result = evaluate_task(
        genotype,
        WalkingEvaluationConfig(
            episode_seconds=0.02,
            settle_seconds=0.0,
            max_creature_height=10.0,
            max_volume=10.0,
        ),
    )

    assert result.fitness == -1_000.0
    assert result.build_failed
    assert result.failure_reason is not None
    assert MAXIMUM_CREATURE_HEIGHT_REASON in result.failure_reason


def test_walking_upright_error_penalty_is_optional(monkeypatch):
    from evol_virtual_creature import evaluation

    genotype = build_genotype(
        root="body",
        spec={"body": {"size": (0.1, 0.1, 0.1), "joint_type": "free"}},
    )
    metrics = {
        "origin_distance": 0.0,
        "average_origin_speed": 0.0,
        "forward_distance": 0.0,
        "average_forward_speed": 0.0,
        "sideways_drift_speed": 0.0,
        "vertical_drift_speed": 0.0,
        "height_loss": 0.0,
        "control_energy": 0.0,
        "mean_angular_speed": 0.0,
        "mean_upright_error": 2.0,
        "simulated_seconds": 1.0,
        "actuator_count": 0,
        "body_count": 0,
        "total_volume": 0.0,
    }

    monkeypatch.setattr(evaluation, "_build_model", lambda *_args: (object(), type("Data", (), {})(), object()))
    monkeypatch.setattr(evaluation, "initialize_model", lambda *_args: None)
    monkeypatch.setattr(evaluation, "_walking_height_failure_reason", lambda *_args: None)
    monkeypatch.setattr(evaluation, "_run_controlled_episode", lambda *_args, **_kwargs: metrics)

    default_result = evaluate_task(
        genotype, WalkingEvaluationConfig(volume_penalty_cutoff=0.0)
    )
    enabled_result = evaluate_task(
        genotype,
        WalkingEvaluationConfig(
            upright_weight=evaluation.DEFAULT_UPRIGHT_ERROR_WEIGHT,
            volume_penalty_cutoff=0.0,
        ),
    )

    assert default_result.fitness == pytest.approx(0.0)
    assert enabled_result.fitness == pytest.approx(
        -evaluation.DEFAULT_UPRIGHT_ERROR_WEIGHT * metrics["mean_upright_error"]
    )


def test_walking_away_fitness_uses_average_origin_speed(monkeypatch):
    from evol_virtual_creature import evaluation

    genotype = build_genotype(
        root="body",
        spec={"body": {"size": (0.1, 0.1, 0.1), "joint_type": "free"}},
    )
    metrics = {
        "origin_distance": 20.0,
        "average_origin_speed": 2.0,
        "forward_distance": 0.0,
        "average_forward_speed": 0.0,
        "sideways_drift_speed": 0.0,
        "vertical_drift_speed": 0.0,
        "height_loss": 0.0,
        "control_energy": 0.0,
        "mean_angular_speed": 0.0,
        "mean_upright_error": 0.0,
        "simulated_seconds": 10.0,
        "actuator_count": 0,
        "body_count": 0,
        "total_volume": 0.0,
    }

    monkeypatch.setattr(evaluation, "_build_model", lambda *_args: (object(), type("Data", (), {})(), object()))
    monkeypatch.setattr(evaluation, "initialize_model", lambda *_args: None)
    monkeypatch.setattr(evaluation, "_walking_height_failure_reason", lambda *_args: None)
    monkeypatch.setattr(evaluation, "_run_controlled_episode", lambda *_args, **_kwargs: metrics)

    result = evaluate_task(
        genotype, WalkingAwayEvaluationConfig(volume_penalty_cutoff=0.0)
    )

    assert result.fitness == pytest.approx(2.0)


def test_flying_speed_is_measured_before_first_ground_contact():
    genotype = build_genotype(
        root="body",
        spec={
            "body": {
                "size": (0.1, 0.1, 0.1),
                "joint_type": "free",
            }
        },
    )
    config = FlyingAwayEvaluationConfig(
        episode_seconds=3.0,
        environment=replace(FLYING_AWAY_ENVIRONMENT, fluid_density=0.0, fluid_viscosity=0.0),
    )
    builder = PhenotypeBuilder(
        genotype,
        max_node=config.max_node,
        environment=config.environment,
    )
    model = mujoco.MjModel.from_xml_string(builder.build())
    data = mujoco.MjData(model)

    assert initialize_model(model, data, config) is None
    data.qvel[0] = 10.0

    metrics = _run_flying_episode(model, data, builder, config)

    assert not isinstance(metrics, str)
    assert metrics["first_ground_contact_time"] is not None
    assert metrics["simulated_seconds"] == pytest.approx(config.episode_seconds)
    assert metrics["origin_distance"] > 0.0
    assert metrics["average_origin_speed"] == pytest.approx(
        metrics["origin_distance"] / metrics["first_ground_contact_time"]
    )
    assert metrics["simulated_seconds"] > metrics["first_ground_contact_time"]
    assert metrics["average_origin_speed"] > (
        metrics["origin_distance"] / metrics["simulated_seconds"]
    )


def test_flying_precontact_speed_uses_center_of_mass(monkeypatch):
    from evol_virtual_creature import evaluation

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.5" gravity="0 0 0"/>
          <worldbody>
            <geom name="floor" type="plane" size="5 5 0.1"/>
            <body name="root">
              <freejoint/>
              <geom type="sphere" size="0.1" pos="0 0 1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )

    class Builder:
        actuator_controllers = []

    centers = iter([
        np.array((0.0, 0.0, 10.0)),
        np.array((3.0, 4.0, 8.0)),
        np.array((100.0, 100.0, 0.0)),
    ])
    floor_contacts = iter([False, True])
    monkeypatch.setattr(
        evaluation,
        "_creature_center_of_mass",
        lambda *_args: next(centers),
    )
    monkeypatch.setattr(
        evaluation,
        "_has_floor_contact",
        lambda *_args: next(floor_contacts),
    )

    metrics = _run_flying_episode(
        model,
        mujoco.MjData(model),
        Builder(),
        FlyingAwayEvaluationConfig(episode_seconds=1.0),
    )

    assert not isinstance(metrics, str)
    assert metrics["first_ground_contact_time"] == pytest.approx(0.5)
    assert metrics["origin_distance"] == pytest.approx(5.0)
    assert metrics["average_origin_speed"] == pytest.approx(10.0)
    assert metrics["forward_distance"] == pytest.approx(3.0)
    assert metrics["height_loss"] == pytest.approx(10.0)


def test_flying_initialization_raises_low_creature_above_floor():
    genotype = build_genotype(
        root="body",
        spec={
            "body": {
                "size": (0.2, 0.2, 0.8),
                "joint_type": "free",
            }
        },
    )
    model = mujoco.MjModel.from_xml_string(
        PhenotypeBuilder(
            genotype, max_node=500, environment=FLYING_X_ENVIRONMENT
        ).build()
    )
    data = mujoco.MjData(model)

    assert initialize_model(model, data, FlyingEvaluationConfig()) is None

    floor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
    )
    body_geom_id = _creature_geom_ids(model)[0]
    lowest_z = data.geom_xpos[body_geom_id, 2] - model.geom_size[body_geom_id, 2]
    assert lowest_z == pytest.approx(data.geom_xpos[floor_id, 2] + 5.0)


def test_walking_initialization_raises_low_creature_above_floor():
    genotype = build_genotype(
        root="body",
        spec={
            "body": {
                "size": (0.2, 0.2, 0.8),
                "joint_type": "free",
            }
        },
    )
    model = mujoco.MjModel.from_xml_string(
        PhenotypeBuilder(
            genotype, max_node=500, environment=WALKING_X_ENVIRONMENT
        ).build()
    )
    data = mujoco.MjData(model)

    assert initialize_model(model, data, WalkingEvaluationConfig()) is None

    floor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
    )
    body_geom_id = _creature_geom_ids(model)[0]
    lowest_z = data.geom_xpos[body_geom_id, 2] - model.geom_size[body_geom_id, 2]
    # Generic initialization applies clearance and then the configured settling pass.
    assert lowest_z == pytest.approx(data.geom_xpos[floor_id, 2], abs=2e-4)
    assert all(float(contact.dist) >= -2e-4 for contact in data.contact)


def test_generic_initialization_runs_environment_callback():
    genotype = build_genotype(
        root="body",
        spec={"body": {"size": (0.1, 0.1, 0.1), "joint_type": "free"}},
    )
    environment = replace(
        SWIMMING_X_ENVIRONMENT,
        initialization_callback=_reject_initialization,
    )
    config = SwimmingEvaluationConfig(environment=environment)
    builder = PhenotypeBuilder(genotype, max_node=500, environment=environment)
    model = mujoco.MjModel.from_xml_string(builder.build())

    assert initialize_model(model, mujoco.MjData(model), config) == (
        "custom initialization rejected model"
    )


def test_shared_evaluator_initializes_swimming_tasks():
    environment = replace(
        SWIMMING_X_ENVIRONMENT,
        initialization_callback=_reject_initialization,
    )
    result = evaluate_task(
        load_genotype_from_json("examples/single_root.json"),
        SwimmingEvaluationConfig(environment=environment),
    )

    assert result.fitness == -1_000.0
    assert result.build_failed
    assert result.failure_reason == "custom initialization rejected model"


def test_initial_floor_overlap_assigns_low_fitness(monkeypatch):
    from evol_virtual_creature import evaluation

    monkeypatch.setattr(evaluation, "_has_floor_penetration", lambda *_args: True)
    result = evaluate_task(
        load_genotype_from_json("examples/single_root.json"),
        WalkingEvaluationConfig(episode_seconds=0.02, settle_seconds=0.0),
    )

    assert result.fitness == -1_000.0
    assert result.build_failed
    assert result.failure_reason == INITIAL_FLOOR_OVERLAP_REASON


def test_detects_nonparent_contact_but_parent_filter_remains_enabled():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option gravity="0 0 0"/>
          <worldbody>
            <body name="root">
              <freejoint/>
              <geom name="root_geom" type="sphere" size="0.1" pos="0 0 1"/>
              <body name="first_child">
                <joint type="hinge" axis="1 0 0"/>
                <geom name="first_geom" type="sphere" size="0.2"/>
              </body>
              <body name="second_child">
                <joint type="hinge" axis="0 1 0"/>
                <geom name="second_geom" type="sphere" size="0.2"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    assert data.ncon == 1
    assert _has_nonparent_self_collision(model, data)


def test_disallowed_collision_assigns_low_fitness(monkeypatch):
    from evol_virtual_creature import evaluation

    genotype = load_genotype_from_json(GENOTYPE_PATH)
    monkeypatch.setattr(
        evaluation, "_has_nonparent_self_collision", lambda _model, _data: True
    )
    result = evaluate_task(
        genotype,
        SwimmingEvaluationConfig(
            episode_seconds=0.02,
            disallow_collision=True,
        ),
    )

    assert result.fitness == -1_000.0
    assert result.disqualified
    assert not result.build_failed
    assert "self-collision" in result.failure_reason

@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("qvel", 1_001.0),
        ("qacc", 100_001.0),
    ],
)
def test_excessive_velocity_or_acceleration_is_rejected(field_name, value):
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body>
              <freejoint/>
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    data.time = 0.01
    getattr(data, field_name)[0] = value

    failure = simulation_failure_reason(
        data,
        previous_time=0.0,
        config=SwimmingEvaluationConfig(),
    )

    assert failure == NUMERICAL_INSTABILITY_REASON


def test_simulation_time_reset_is_rejected():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body>
              <freejoint/>
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    data.time = 0.01

    failure = simulation_failure_reason(
        data,
        previous_time=0.06,
        config=SwimmingEvaluationConfig(),
    )

    assert failure == NUMERICAL_INSTABILITY_REASON


def test_evaluation_rejects_numerical_instability(monkeypatch):
    from evol_virtual_creature import evaluation

    genotype = load_genotype_from_json(GENOTYPE_PATH)
    original_step = evaluation.mujoco.mj_step

    def inject_excessive_acceleration(model, data):
        original_step(model, data)
        data.qacc[0] = 100_001.0

    monkeypatch.setattr(
        evaluation.mujoco,
        "mj_step",
        inject_excessive_acceleration,
    )
    result = evaluate_task(
        genotype,
        SwimmingEvaluationConfig(episode_seconds=0.02),
    )

    assert result.fitness == -1_000.0
    assert result.build_failed
    assert result.failure_reason == NUMERICAL_INSTABILITY_REASON

@pytest.mark.parametrize(
    ("geom_xml", "expected_volume"),
    [
        ('<geom type="box" size="1 2 3"/>', 48.0),
        ('<geom type="ellipsoid" size="1 2 3"/>', 8.0 * math.pi),
        (
            '<geom type="capsule" size="0.5" fromto="-2 0 0 2 0 0"/>',
            7.0 / 6.0 * math.pi,
        ),
        (
            '<geom type="cylinder" size="0.5" fromto="-2 0 0 2 0 0"/>',
            math.pi,
        ),
    ],
)
def test_creature_volume_sums_generated_body_volumes_only(
    geom_xml,
    expected_volume,
):
    model = mujoco.MjModel.from_xml_string(
        f"""
        <mujoco>
          <worldbody>
            <geom type="plane" size="5 5 0.1"/>
            <body>
              <freejoint/>
              {geom_xml}
            </body>
          </worldbody>
        </mujoco>
        """
    )

    assert _creature_volume(model) == pytest.approx(expected_volume)


def test_volume_penalty_has_cutoff_then_grows_linearly():
    genotype = load_genotype_from_json(GENOTYPE_PATH)
    baseline = evaluate_task(
        genotype,
        SwimmingEvaluationConfig(episode_seconds=0.02, volume_weight=0.0),
    )
    below_cutoff = evaluate_task(
        genotype,
        SwimmingEvaluationConfig(
            episode_seconds=0.02,
            volume_weight=1.0,
            volume_penalty_cutoff=0.1,
        ),
    )
    above_cutoff = evaluate_task(
        genotype,
        SwimmingEvaluationConfig(
            episode_seconds=0.02,
            volume_weight=1.0,
            volume_penalty_cutoff=0.05,
        ),
    )

    assert baseline.total_volume < 0.1
    assert below_cutoff.fitness == pytest.approx(baseline.fitness)
    assert baseline.fitness - above_cutoff.fitness == pytest.approx(
        baseline.total_volume - 0.05
    )


def test_creature_above_maximum_volume_is_rejected_before_simulation():
    genotype = build_genotype(
        root="body",
        spec={
            "body": {
                "size": (0.6, 0.6, 0.6),
                "joint_type": "free",
            }
        },
    )

    result = evaluate_task(
        genotype,
        SwimmingEvaluationConfig(episode_seconds=0.02, max_volume=1.0),
    )

    assert result.fitness == -1_000.0
    assert result.build_failed
    assert "exceeds maximum allowed volume" in result.failure_reason


def test_body_below_minimum_volume_is_rejected_before_flying_bonus():
    genotype = build_genotype(
        root="body",
        spec={
            "body": {
                "size": (0.01, 0.01, 0.001),
                "joint_type": "free",
            }
        },
    )

    result = evaluate_task(
        genotype,
        FlyingEvaluationConfig(episode_seconds=0.02, min_body_volume=1e-6),
    )

    assert result.fitness == -1_000.0
    assert result.build_failed
    assert "below the minimum allowed volume" in result.failure_reason
    assert result.no_ground_touch_bonus == 0.0

def test_flying_total_volume_below_minimum_is_rejected_before_simulation():
    genotype = build_genotype(
        root="body",
        spec={
            "body": {
                "size": (0.02, 0.02, 0.02),
                "joint_type": "free",
            }
        },
    )

    result = evaluate_task(
        genotype,
        FlyingEvaluationConfig(episode_seconds=0.02),
    )

    assert result.fitness == -1_000.0
    assert result.build_failed
    assert "total volume is below the minimum allowed volume" in result.failure_reason
    assert result.no_ground_touch_bonus == 0.0


def _single_motor_genotype(motor_gear):
    return build_genotype(
        root="body",
        spec={
            "body": {
                "size": (0.2, 0.1, 0.1),
                "joint_type": "free",
                "children": [{
                    "child": "limb",
                    "axis": (0, 1, 0),
                    "motor_gear": motor_gear,
                    "control_amp": 0.1,
                    "control_freq": 0.0,
                    "control_phase": math.pi / 2.0,
                }],
            },
            "limb": {"size": (0.1, 0.05, 0.05), "joint_type": "hinge"},
        },
        control_mode="sine",
    )


def test_control_energy_scales_with_squared_motor_gear():
    low_gear = evaluate_task(
        _single_motor_genotype(10.0),
        SwimmingEvaluationConfig(episode_seconds=0.01),
    )
    high_gear = evaluate_task(
        _single_motor_genotype(20.0),
        SwimmingEvaluationConfig(episode_seconds=0.01),
    )

    assert low_gear.control_energy > 0.0
    assert high_gear.control_energy == pytest.approx(4.0 * low_gear.control_energy)


def _neural_single_motor_model(neural_b2):
    genotype = build_genotype(
        root="body",
        spec={
            "body": {
                "size": (0.2, 0.1, 0.1),
                "joint_type": "free",
                "children": [{
                    "child": "limb",
                    "axis": (0, 1, 0),
                    "ctrlrange": (-2.0, 2.0),
                    "neural_b2": neural_b2,
                }],
            },
            "limb": {"size": (0.1, 0.05, 0.05), "joint_type": "hinge"},
        },
    )
    builder = PhenotypeBuilder(genotype, max_node=10)
    model = mujoco.MjModel.from_xml_string(builder.build())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, builder


def test_zero_neural_weights_produce_zero_control():
    model, data, builder = _neural_single_motor_model((0.0,))

    apply_controller(model, data, builder.actuator_controllers)

    assert data.ctrl[0] == pytest.approx(0.0)


def test_neural_output_sign_maps_to_control_direction():
    positive_model, positive_data, positive_builder = _neural_single_motor_model((1.0,))
    negative_model, negative_data, negative_builder = _neural_single_motor_model((-1.0,))

    apply_controller(positive_model, positive_data, positive_builder.actuator_controllers)
    apply_controller(negative_model, negative_data, negative_builder.actuator_controllers)

    assert positive_data.ctrl[0] > 0.0
    assert negative_data.ctrl[0] < 0.0
    assert positive_data.ctrl[0] == pytest.approx(-negative_data.ctrl[0])


def test_away_configs_reject_old_distance_weight_name():
    with pytest.raises(TypeError):
        SwimmingAwayEvaluationConfig(distance_weight=1.0)
    with pytest.raises(TypeError):
        WalkingAwayEvaluationConfig(distance_weight=1.0)
    with pytest.raises(TypeError):
        FlyingAwayEvaluationConfig(distance_weight=0.1)


def test_swimming_away_fitness_maximizes_average_origin_speed():
    genotype = load_genotype_from_json(GENOTYPE_PATH)
    result = evaluate_task(
        genotype,
        SwimmingAwayEvaluationConfig(
            episode_seconds=0.02,
            energy_weight=0.0,
            angular_speed_weight=0.0,
            body_count_weight=0.0,
            volume_weight=0.0,
        ),
    )

    assert result.fitness == pytest.approx(result.average_origin_speed)
    assert result.origin_distance >= abs(result.forward_distance)


def test_flying_fitness_uses_xy_speed_and_penalizes_ground_contact():
    config = FlyingAwayEvaluationConfig(
        speed_weight=0.1,
        energy_weight=0.0,
        height_loss_weight=1.0,
        angular_speed_weight=0.0,
        body_count_weight=0.0,
        volume_weight=0.0,
    )
    metrics = {
        "origin_distance": 20.0,
        "forward_distance": 12.0,
        "average_origin_speed": 4.0,
        "average_forward_speed": 2.0,
        "height_loss": 0.5,
        "ground_touch_penalty": 0.25,
        "no_ground_touch_bonus": 0.0,
        "control_energy": 0.0,
        "mean_angular_speed": 0.0,
        "body_count": 1,
        "total_volume": 0.0,
    }

    assert flying_away_episode_fitness(config, metrics) == pytest.approx(
        0.4 - 0.5 - 0.25
    )

    metrics["height_loss"] = 0.0
    metrics["ground_touch_penalty"] = 0.0
    metrics["no_ground_touch_bonus"] = config.no_ground_touch_bonus
    assert flying_away_episode_fitness(config, metrics) == pytest.approx(
        0.4 + config.no_ground_touch_bonus
    )


def test_x_axis_flying_fitness_uses_forward_speed():
    metrics = {
        "origin_distance": 4.0,
        "average_origin_speed": 4.0,
        "forward_distance": 2.0,
        "average_forward_speed": 2.0,
        "sideways_drift_speed": 0.0,
        "height_loss": 0.0,
        "first_ground_contact_time": None,
        "ground_touch_penalty": 0.0,
        "no_ground_touch_bonus": 0.0,
        "control_energy": 0.0,
        "mean_angular_speed": 0.0,
        "simulated_seconds": 1.0,
        "actuator_count": 0,
        "body_count": 0,
        "total_volume": 0.0,
    }

    config = FlyingEvaluationConfig(
        distance_weight=1.0, energy_weight=0.0, height_loss_weight=0.0,
        angular_speed_weight=0.0, body_count_weight=0.0, volume_weight=0.0,
    )
    assert flying_x_episode_fitness(config, metrics) == pytest.approx(2.0)


def test_flying_passive_baseline_blends_controlled_gain(monkeypatch):
    from evol_virtual_creature import evaluation

    controlled_metrics = {
        "origin_distance": 2.0,
        "average_origin_speed": 2.0,
        "forward_distance": 2.0,
        "average_forward_speed": 2.0,
        "sideways_drift_speed": 0.0,
        "height_loss": 0.0,
        "first_ground_contact_time": None,
        "ground_touch_penalty": 0.0,
        "no_ground_touch_bonus": 0.0,
        "control_energy": 0.0,
        "mean_angular_speed": 0.0,
        "simulated_seconds": 1.0,
        "actuator_count": 0,
        "body_count": 0,
        "total_volume": 0.0,
    }
    passive_metrics = dict(controlled_metrics, average_forward_speed=0.5)

    def fake_run(_model, _data, _builder, _config, apply_controls=True):
        return controlled_metrics.copy() if apply_controls else passive_metrics.copy()

    config = FlyingEvaluationConfig(
        distance_weight=1.0,
        energy_weight=0.0,
        height_loss_weight=0.0,
        angular_speed_weight=0.0,
        body_count_weight=0.0,
        volume_weight=0.0,
        fitness_gain_fraction=0.5,
    )
    monkeypatch.setattr(evaluation, "_copy_simulation_state", lambda *_args: object())
    monkeypatch.setattr(evaluation, "_run_flying_episode", fake_run)

    monkeypatch.setattr(evaluation, "_build_model", lambda *_args: (object(), object(), object()))
    monkeypatch.setattr(evaluation, "initialize_model", lambda *_args: None)
    result = evaluate_task(object(), config)

    assert result.controlled_fitness == pytest.approx(2.0)
    assert result.passive_fitness == pytest.approx(0.5)
    assert result.fitness_gain == pytest.approx(1.5)
    assert result.fitness == pytest.approx(1.75)


def test_walking_away_fitness_maximizes_average_origin_speed():
    genotype = load_genotype_from_json(GENOTYPE_PATH)
    result = evaluate_task(
        genotype,
        WalkingAwayEvaluationConfig(
            episode_seconds=0.02,
            settle_seconds=0.02,
            energy_weight=0.0,
            angular_speed_weight=0.0,
            body_count_weight=0.0,
            volume_weight=0.0,
            height_loss_weight=0.0,
        ),
    )

    assert result.fitness == pytest.approx(result.average_origin_speed)
    assert result.origin_distance >= abs(result.forward_distance)
