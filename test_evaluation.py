import xml.etree.ElementTree as ET

import mujoco

from evol_virtual_creature.evaluation import (
    SwimmingEvaluationConfig,
    WalkingEvaluationConfig,
    evaluate_x_axis_swimming,
    evaluate_x_axis_walking,
)
from evol_virtual_creature.genotype_io import load_genotype_from_json
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
