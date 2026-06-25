import math
import xml.etree.ElementTree as ET

import mujoco
import pytest

from evol_virtual_creature.evaluation import (
    NUMERICAL_INSTABILITY_REASON,
    INITIAL_FLOOR_OVERLAP_REASON,
    SwimmingEvaluationConfig,
    WalkingEvaluationConfig,
    _has_nonparent_self_collision,
    _creature_volume,
    evaluate_x_axis_swimming,
    evaluate_x_axis_walking,
    initialize_walking_model,
    simulation_failure_reason,
)
from evol_virtual_creature.genotype_io import build_genotype, load_genotype_from_json
from evol_virtual_creature.phenotype import PhenotypeBuilder


GENOTYPE_PATH = "examples/example_genotype.json"


def test_task_physics_settings_are_distinct():
    genotype = load_genotype_from_json(GENOTYPE_PATH)
    swimming = ET.fromstring(
        PhenotypeBuilder(genotype, max_node=500, task="swimming").build()
    )
    walking = ET.fromstring(
        PhenotypeBuilder(genotype, max_node=500, task="walking").build()
    )

    swimming_option = swimming.find("option")
    walking_option = walking.find("option")
    assert swimming_option is not None and walking_option is not None
    assert swimming_option.get("gravity") == "0 0 0"
    assert swimming_option.get("density") == "1000"
    assert walking_option.get("gravity") == "0 0 -9.81"
    assert walking_option.get("density") == "0"
    assert walking_option.get("viscosity") == "0"

    swimming_geom = swimming.find("./default/geom")
    walking_geom = walking.find("./default/geom")
    assert swimming_geom is not None and walking_geom is not None
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
            genotype, max_node=500, task="walking", self_collision=False
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
            genotype, max_node=500, task="walking", self_collision=True
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
            genotype, max_node=500, task="swimming", self_collision=True
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
    swimming = evaluate_x_axis_swimming(
        genotype, SwimmingEvaluationConfig(episode_seconds=0.02)
    )
    walking = evaluate_x_axis_walking(
        genotype,
        WalkingEvaluationConfig(episode_seconds=0.02, settle_seconds=0.02),
    )
    assert not swimming.build_failed
    assert not walking.build_failed
    assert swimming.simulated_seconds == 0.02
    assert walking.simulated_seconds == 0.02
    assert walking.mean_upright_error >= 0.0
    assert walking.height_loss >= 0.0


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
        PhenotypeBuilder(genotype, max_node=500, task="walking").build()
    )
    data = mujoco.MjData(model)

    assert initialize_walking_model(model, data) is None

    floor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
    )
    body_geom_id = _creature_geom_ids(model)[0]
    lowest_z = data.geom_xpos[body_geom_id, 2] - model.geom_size[body_geom_id, 2]
    assert lowest_z == pytest.approx(data.geom_xpos[floor_id, 2] + 0.05)
    assert all(float(contact.dist) >= 0.0 for contact in data.contact)


def test_initial_floor_overlap_assigns_low_fitness(monkeypatch):
    from evol_virtual_creature import evaluation

    monkeypatch.setattr(evaluation, "_has_floor_penetration", lambda *_args: True)
    result = evaluate_x_axis_walking(
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
    result = evaluate_x_axis_swimming(
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
    result = evaluate_x_axis_swimming(
        genotype,
        SwimmingEvaluationConfig(episode_seconds=0.02),
    )

    assert result.fitness == -1_000.0
    assert result.build_failed
    assert result.failure_reason == NUMERICAL_INSTABILITY_REASON

def test_creature_volume_sums_generated_box_volumes_only():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom type="plane" size="5 5 0.1"/>
            <body>
              <freejoint/>
              <geom type="box" size="1 2 3"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )

    assert _creature_volume(model) == pytest.approx(48.0)


def test_volume_penalty_has_cutoff_then_grows_linearly():
    genotype = load_genotype_from_json(GENOTYPE_PATH)
    baseline = evaluate_x_axis_swimming(
        genotype,
        SwimmingEvaluationConfig(episode_seconds=0.02, volume_weight=0.0),
    )
    below_cutoff = evaluate_x_axis_swimming(
        genotype,
        SwimmingEvaluationConfig(
            episode_seconds=0.02,
            volume_weight=1.0,
            volume_penalty_cutoff=0.1,
        ),
    )
    above_cutoff = evaluate_x_axis_swimming(
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

    result = evaluate_x_axis_swimming(
        genotype,
        SwimmingEvaluationConfig(episode_seconds=0.02, max_volume=1.0),
    )

    assert result.fitness == -1_000.0
    assert result.build_failed
    assert "exceeds maximum allowed volume" in result.failure_reason


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
            "limb": {"size": (0.1, 0.05, 0.05)},
        },
    )


def test_control_energy_scales_with_squared_motor_gear():
    low_gear = evaluate_x_axis_swimming(
        _single_motor_genotype(10.0),
        SwimmingEvaluationConfig(episode_seconds=0.01),
    )
    high_gear = evaluate_x_axis_swimming(
        _single_motor_genotype(20.0),
        SwimmingEvaluationConfig(episode_seconds=0.01),
    )

    assert low_gear.control_energy > 0.0
    assert high_gear.control_energy == pytest.approx(4.0 * low_gear.control_energy)
