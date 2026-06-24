import xml.etree.ElementTree as ET

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
    assert walking_geom.get("contype") == "1"
    assert walking_geom.get("conaffinity") == "1"


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
