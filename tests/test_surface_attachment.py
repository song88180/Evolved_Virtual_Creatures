import json
import xml.etree.ElementTree as ET

import mujoco
import pytest

from evol_virtual_creature.genes import ConnectionGene, NodeGene
from evol_virtual_creature.genotype import Genotype
from evol_virtual_creature.genotype_io import (
    build_genotype,
    load_genotype_from_json,
    save_genotype_to_json,
)
from evol_virtual_creature.graph_analysis import GenotypeGraphAnalyzer
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
        "archived_nodes": [{
            "name": "archived_limb",
            "size": [0.04, 0.04, 0.04],
            "joint_axis": [0, 0, 1],
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
            "limb": {
                "size": [0.05, 0.08, 0.02],
                "joint_axis": [1, 0, 0],
            },
        },
    }))

    genotype = load_genotype_from_json(genotype_path)
    connection = genotype.nodes["body"].children[0]
    assert not hasattr(genotype.nodes["limb"], "joint_axis")
    assert not hasattr(genotype.archived_nodes[0], "joint_axis")

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
    saved_data = json.loads(migrated_path.read_text())
    assert "joint_axis" not in saved_data["nodes"]["limb"]
    assert "joint_axis" not in saved_data["archived_nodes"][0]


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


def test_connection_scale_changes_repeated_child_size_by_hierarchy_order():
    recursive_connection = ConnectionGene(
        child="segment",
        axis=(0, 1, 0),
        parent_face="+x",
        symmetry=("xz",),
        scale=0.8,
    )
    genotype = Genotype(
        root="body",
        nodes={
            "body": NodeGene(
                name="body",
                size=(1.0, 1.0, 1.0),
                joint_type="free",
                children=[ConnectionGene(child="segment", axis=(0, 1, 0))],
            ),
            "segment": NodeGene(
                name="segment",
                size=(0.5, 0.25, 0.1),
                recursive_limit=3,
                children=[recursive_connection],
            ),
        },
    )

    xml_root = ET.fromstring(PhenotypeBuilder(genotype, max_node=10).build())
    initial_segment = xml_root.find("./worldbody/body/body")
    first_order = initial_segment.findall("body")
    second_order = [
        child
        for parent in first_order
        for child in parent.findall("body")
    ]

    assert _vector(initial_segment.find("geom"), "size") == pytest.approx(
        (0.5, 0.25, 0.1)
    )
    assert len(first_order) == 2
    assert len(second_order) == 4
    for body in first_order:
        assert _vector(body.find("geom"), "size") == pytest.approx(
            (0.5, 0.25, 0.1)
        )
    for body in second_order:
        assert _vector(body.find("geom"), "size") == pytest.approx(
            (0.4, 0.2, 0.08)
        )


def test_terminal_only_child_is_created_only_on_terminal_parent():
    genotype = Genotype(
        root="body",
        nodes={
            "body": NodeGene(
                name="body",
                size=(1.0, 1.0, 1.0),
                joint_type="free",
                children=[ConnectionGene(child="segment", axis=(0, 1, 0))],
            ),
            "segment": NodeGene(
                name="segment",
                size=(0.4, 0.2, 0.1),
                recursive_limit=3,
                children=[
                    ConnectionGene(child="segment", axis=(0, 1, 0)),
                    ConnectionGene(
                        child="tip",
                        axis=(1, 0, 0),
                        terminal_only=True,
                    ),
                ],
            ),
            "tip": NodeGene(name="tip", size=(0.1, 0.1, 0.1)),
        },
    )

    assert GenotypeGraphAnalyzer(genotype, max_node=5).validate() == 5
    xml_root = ET.fromstring(PhenotypeBuilder(genotype, max_node=5).build())
    first_segment = xml_root.find("./worldbody/body/body")
    second_segment = first_segment.find("./body")
    terminal_segment = second_segment.find("./body")

    assert len(first_segment.findall("./body")) == 1
    assert len(second_segment.findall("./body")) == 1
    terminal_children = terminal_segment.findall("./body")
    assert len(terminal_children) == 1
    assert terminal_children[0].get("name").startswith("tip_")


def test_slide_and_ball_joints_compile_with_expected_actuators():
    genotype = Genotype(
        root="body",
        nodes={
            "body": NodeGene(
                name="body",
                size=(1.0, 1.0, 1.0),
                joint_type="free",
                children=[
                    ConnectionGene(
                        child="slider",
                        axis=(1.0, 0.0, 0.0),
                        motor_gear=20.0,
                    ),
                    ConnectionGene(
                        child="ball",
                        axis=(0.0, 2.0, 0.0),
                        motor_gear=50.0,
                    ),
                ],
            ),
            "slider": NodeGene(
                name="slider",
                size=(0.2, 0.1, 0.1),
                joint_type="slide",
            ),
            "ball": NodeGene(
                name="ball",
                size=(0.2, 0.1, 0.1),
                joint_type="ball",
            ),
        },
    )

    mjcf = PhenotypeBuilder(genotype, max_node=3).build()
    xml_root = ET.fromstring(mjcf)
    child_bodies = xml_root.findall("./worldbody/body/body")
    joints = [body.find("joint") for body in child_bodies]
    motors = xml_root.findall("./actuator/motor")

    assert [joint.get("type") for joint in joints] == ["slide", "ball"]
    assert _vector(joints[0], "axis") == pytest.approx((1.0, 0.0, 0.0))
    assert _vector(joints[0], "range") == pytest.approx((-0.5, 0.5))
    assert joints[1].get("axis") is None
    assert _vector(joints[1], "range") == pytest.approx((0.0, 45.0))
    assert tuple(float(value) for value in motors[1].get("gear").split()) == pytest.approx(
        (0.0, 50.0, 0.0)
    )

    model = mujoco.MjModel.from_xml_string(mjcf)
    assert model.nv == 10
    assert model.nu == 2


def test_slide_axis_uses_vector_reflection_for_symmetry():
    genotype = Genotype(
        root="body",
        nodes={
            "body": NodeGene(
                name="body",
                size=(1.0, 1.0, 1.0),
                joint_type="free",
                children=[
                    ConnectionGene(
                        child="slider",
                        axis=(1.0, 0.0, 0.0),
                        symmetry=("yz",),
                    )
                ],
            ),
            "slider": NodeGene(
                name="slider",
                size=(0.2, 0.1, 0.1),
                joint_type="slide",
            ),
        },
    )

    xml_root = ET.fromstring(PhenotypeBuilder(genotype, max_node=3).build())
    joints = [
        body.find("joint")
        for body in xml_root.findall("./worldbody/body/body")
    ]

    assert [_vector(joint, "axis") for joint in joints] == [
        pytest.approx((1.0, 0.0, 0.0)),
        pytest.approx((-1.0, 0.0, 0.0)),
    ]


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
