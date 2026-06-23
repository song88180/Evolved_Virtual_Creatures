import json
import xml.etree.ElementTree as ET

import pytest

from evol_virtual_creature.genes import ConnectionGene, NodeGene
from evol_virtual_creature.genotype import Genotype
from evol_virtual_creature.genotype_io import (
    build_genotype,
    load_genotype_from_json,
    save_genotype_to_json,
)
from evol_virtual_creature.phenotype import PhenotypeBuilder


def _vector(element, attribute):
    return tuple(float(value) for value in element.get(attribute).split())


def test_connection_rejects_invalid_surface_attachment():
    with pytest.raises(ValueError, match="Unknown parent face"):
        ConnectionGene(child="child", axis=(0, 1, 0), parent_face="front")

    with pytest.raises(ValueError, match="between -1 and 1"):
        ConnectionGene(
            child="child",
            axis=(0, 1, 0),
            surface_uv=(1.1, 0.0),
        )


def test_legacy_position_loads_as_nearest_surface_attachment(tmp_path):
    genotype_path = tmp_path / "legacy.json"
    genotype_path.write_text(json.dumps({
        "root": "body",
        "archived_connections": [{
            "child": "limb",
            "pos": [0.0, 0.0, -0.4],
            "axis": [1, 0, 0],
        }],
        "nodes": {
            "body": {
                "size": [0.25, 0.15, 0.1],
                "joint_type": "free",
                "children": [{
                    "child": "limb",
                    "pos": [0.0, -0.30, 0.05],
                    "axis": [1, 0, 0],
                }],
            },
            "limb": {"size": [0.05, 0.08, 0.02]},
        },
    }))

    genotype = load_genotype_from_json(genotype_path)
    connection = genotype.nodes["body"].children[0]

    assert connection.parent_face == "-y"
    assert connection.surface_uv == pytest.approx((0.0, 0.5))
    assert genotype.archived_connections[0].parent_face == "-z"
    assert genotype.archived_connections[0].surface_uv == (0.0, 0.0)

    migrated_path = tmp_path / "migrated.json"
    save_genotype_to_json(genotype, migrated_path)
    saved_connection = json.loads(migrated_path.read_text())["nodes"]["body"][
        "children"
    ][0]
    assert "pos" not in saved_connection
    assert saved_connection["parent_face"] == "-y"


def test_child_hinge_and_geom_meet_parent_surface():
    genotype = build_genotype(
        root="body",
        spec={
            "body": {
                "size": (2.0, 1.0, 0.5),
                "joint_type": "free",
                "children": [{
                    "child": "limb",
                    "axis": (0, 1, 0),
                    "parent_face": "+x",
                    "surface_uv": (0.5, -0.5),
                }],
            },
            "limb": {"size": (0.4, 0.3, 0.2)},
        },
    )

    xml_root = ET.fromstring(PhenotypeBuilder(genotype, max_node=10).build())
    root_body = xml_root.find("./worldbody/body")
    child_body = root_body.find("body")
    child_geom = child_body.find("geom")
    child_joint = child_body.find("joint")

    assert _vector(child_body, "pos") == pytest.approx((2.0, 0.5, -0.25))
    assert _vector(child_joint, "pos") == pytest.approx((0.0, 0.0, 0.0))
    assert _vector(child_geom, "pos") == pytest.approx((0.4, 0.0, 0.0))

    parent_surface_x = 2.0
    child_inner_surface_x = (
        _vector(child_body, "pos")[0]
        + _vector(child_geom, "pos")[0]
        - 0.4
    )
    assert child_inner_surface_x == pytest.approx(parent_surface_x)


def test_recursive_attachment_uses_parent_geom_center():
    genotype = Genotype(
        root="body",
        nodes={
            "body": NodeGene(
                name="body",
                size=(2.0, 1.0, 0.5),
                joint_type="free",
                children=[ConnectionGene(
                    child="segment",
                    axis=(0, 1, 0),
                    parent_face="+x",
                )],
            ),
            "segment": NodeGene(
                name="segment",
                size=(0.4, 0.3, 0.2),
                children=[ConnectionGene(
                    child="tip",
                    axis=(1, 0, 0),
                    parent_face="+y",
                )],
            ),
            "tip": NodeGene(name="tip", size=(0.1, 0.2, 0.1)),
        },
    )

    xml_root = ET.fromstring(PhenotypeBuilder(genotype, max_node=10).build())
    root_body = xml_root.find("./worldbody/body")
    segment_body = root_body.find("body")
    tip_body = segment_body.find("body")

    assert _vector(segment_body, "pos") == pytest.approx((2.0, 0.0, 0.0))
    assert _vector(segment_body.find("geom"), "pos") == pytest.approx(
        (0.4, 0.0, 0.0)
    )
    assert _vector(tip_body, "pos") == pytest.approx((0.4, 0.3, 0.0))
    assert _vector(tip_body.find("geom"), "pos") == pytest.approx(
        (0.0, 0.2, 0.0)
    )
