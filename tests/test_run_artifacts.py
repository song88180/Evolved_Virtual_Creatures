"""Tests for evc3d.run_artifacts: files written into an evolve run directory."""

from dataclasses import replace
import xml.etree.ElementTree as ET

from evc3d import run_artifacts
from evc3d.cli import evolve as evolve_cli
from evc3d.evolution_tasks.flying_x import TASK_ENVIRONMENT as FLYING_ENVIRONMENT
from helpers import make_definition, make_creature


def test_saved_best_xml_preserves_walking_self_collision(tmp_path):
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    path = tmp_path / "best.xml"
    run_artifacts.save_best_phenotype(
        path,
        genotype,
        make_definition("walking_x", episode_seconds=0.02, self_collision=True),
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
    run_artifacts.save_best_phenotype(
        path,
        genotype,
        make_definition("swimming_away", episode_seconds=0.02, self_collision=True),
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
    run_artifacts.save_best_phenotype(
        path,
        genotype,
        make_definition("flying_x", episode_seconds=0.02, self_collision=True),
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
    run_artifacts.save_best_phenotype(
        path,
        genotype,
        make_definition("flying_x", episode_seconds=0.02, environment=replace(FLYING_ENVIRONMENT, gravity=-3.5)),
    )
    option = ET.fromstring(path.read_text()).find("option")
    assert option is not None
    assert option.get("gravity") == "0 0 -3.5"


def test_saved_best_xml_preserves_walking_away_physics(tmp_path):
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    path = tmp_path / "best.xml"
    run_artifacts.save_best_phenotype(
        path,
        genotype,
        make_definition("walking_away", episode_seconds=0.02, self_collision=True),
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
    creature = make_creature(genotype, 1.0)

    run_artifacts.save_generation_checkpoint(
        tmp_path,
        generation=3,
        generation_best=creature,
        best_so_far=creature,
        definition=make_definition("swimming_x", episode_seconds=0.02),
        save_generation_history=False,
    )

    assert (tmp_path / "latest_best_genotype.json").is_file()
    assert (tmp_path / "best_genotype.json").is_file()
    assert not (tmp_path / "generation_bests").exists()


def test_save_generation_best_records_history_by_default(tmp_path):
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    creature = make_creature(genotype, 1.0)

    run_artifacts.save_generation_checkpoint(
        tmp_path,
        generation=3,
        generation_best=creature,
        best_so_far=creature,
        definition=make_definition("swimming_x", episode_seconds=0.02),
    )

    assert (
        tmp_path / "generation_bests" / "generation_0003.json"
    ).is_file()


def test_disallow_collision_enables_saved_self_contact_masks(tmp_path):
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    path = tmp_path / "best.xml"
    run_artifacts.save_best_phenotype(
        path,
        genotype,
        make_definition("swimming_x", episode_seconds=0.02,
            self_collision=False,
            disallow_collision=True,),
    )
    default_geom = ET.fromstring(path.read_text()).find("./default/geom")
    assert default_geom is not None
    assert default_geom.get("contype") == "2"
    assert default_geom.get("conaffinity") == "2"
