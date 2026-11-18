"""Tests for evc3d.phenotype: body attachment, orientation, joints, shapes, and generated MJCF."""

import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from evc3d.genes import ConnectionGene, NodeGene
from evc3d.genotype_io import build_genotype, load_genotype_from_json
from evc3d.graph_analysis import GenotypeGraphAnalyzer
from evc3d.phenotype import PhenotypeBuilder
from helpers import make_definition, tree_genotype, tree_node


def _vector(element, attribute):
    return tuple(float(value) for value in element.get(attribute).split())


def _joint_position_in_body_frame(model, data, joint_name, body_name):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    body_rotation = data.xmat[body_id].reshape(3, 3)
    return body_rotation.T @ (data.xanchor[joint_id] - data.xpos[body_id])


def _joint_position_in_geom_frame(model, data, joint_name, geom_name):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    geom_rotation = data.geom_xmat[geom_id].reshape(3, 3)
    return geom_rotation.T @ (data.xanchor[joint_id] - data.geom_xpos[geom_id])


def _assert_joint_on_child_attachment_face(
    model,
    data,
    joint_name,
    geom_name,
    child_surface_uv=(0.0, 0.0),
):
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    half_sizes = model.geom_size[geom_id, :3]
    joint_local = _joint_position_in_geom_frame(model, data, joint_name, geom_name)

    assert joint_local[0] == pytest.approx(-half_sizes[0])
    assert joint_local[1] == pytest.approx(child_surface_uv[0] * half_sizes[1])
    assert joint_local[2] == pytest.approx(child_surface_uv[1] * half_sizes[2])


def test_neural_hinge_joint_uses_wide_symmetric_range():
    genotype = tree_genotype(
        root="body",
        nodes={
            "body": tree_node(
                name="body",
                size=(0.2, 0.2, 0.2),
                joint_type="free",
                children=[ConnectionGene(child="limb", axis=(0, 1, 0))],
            ),
            "limb": NodeGene(
                name="limb",
                size=(0.1, 0.1, 0.1),
                joint_type="hinge",
            ),
        },
    )

    xml_root = ET.fromstring(PhenotypeBuilder(genotype, make_definition(max_body_count=10)).build())
    joint = xml_root.find("./worldbody/body/body/joint")

    assert joint is not None
    assert _vector(joint, "range") == pytest.approx((-90.0, 90.0))


def test_global_control_frequency_scales_actuator_controllers():
    genotype = build_genotype(
        "body",
        {
            "body": {
                "size": [0.2, 0.2, 0.2],
                "joint_type": "free",
                "children": [
                    {
                        "child": "limb",
                        "axis": [0.0, 1.0, 0.0],
                        "control_freq": 3.0,
                    },
                ],
            },
            "limb": {
                "size": [0.1, 0.1, 0.1],
                "joint_type": "hinge",
            },
        },
        global_control_freq=2.5,
    )

    builder = PhenotypeBuilder(genotype, make_definition(max_body_count=10))
    builder.build()

    assert builder.actuator_controllers[0].freq == pytest.approx(7.5)


@pytest.mark.parametrize(
    ("shape", "expected_type", "expected_size"),
    [
        ("box", mujoco.mjtGeom.mjGEOM_BOX, (0.3, 0.2, 0.1)),
        ("ellipsoid", mujoco.mjtGeom.mjGEOM_ELLIPSOID, (0.3, 0.2, 0.1)),
        ("capsule", mujoco.mjtGeom.mjGEOM_CAPSULE, (0.1, 0.3, 0.0)),
        ("cylinder", mujoco.mjtGeom.mjGEOM_CYLINDER, (0.1, 0.3, 0.0)),
    ],
)
def test_body_shapes_emit_expected_mujoco_geom_and_compile(
    shape,
    expected_type,
    expected_size,
):
    genotype = build_genotype(
        root="body",
        spec={
            "body": {
                "size": (0.3, 0.2, 0.1),
                "shape": shape,
                "joint_type": "free",
            }
        },
    )

    mjcf = PhenotypeBuilder(genotype, make_definition(max_body_count=1)).build()
    xml_root = ET.fromstring(mjcf)
    geom = xml_root.find("./worldbody/body/geom")
    model = mujoco.MjModel.from_xml_string(mjcf)
    body_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "body_1_geom")

    assert geom.get("type") == shape
    assert model.geom_type[body_geom_id] == expected_type
    assert model.geom_size[body_geom_id, :3] == pytest.approx(expected_size)
    if shape in {"capsule", "cylinder"}:
        assert _vector(geom, "fromto") == pytest.approx(
            (-0.3, 0.0, 0.0, 0.3, 0.0, 0.0)
        )
        assert _vector(geom, "size") == pytest.approx((0.1,))
    else:
        assert _vector(geom, "size") == pytest.approx((0.3, 0.2, 0.1))


def test_missing_joint_type_uses_node_gene_default(tmp_path):
    genotype_path = tmp_path / "fixed_default.json"
    genotype_path.write_text(json.dumps({
        "control_mode": "neural",
        "root": "body",
        "nodes": {
            "body": {
                "size": [0.2, 0.2, 0.2],
                "joint_type": "free",
                "child_connections": ["body_to_limb"],
            },
            "limb": {"size": [0.1, 0.1, 0.1]},
            "template": {"size": [0.1, 0.1, 0.1]},
        },
        "connections": {"body_to_limb": {"child": "limb", "axis": [0, 1, 0]}},
    }))

    genotype = load_genotype_from_json(genotype_path)

    assert genotype.nodes["limb"].joint_type == "hinge"
    assert genotype.nodes["template"].joint_type == "hinge"


def test_fixed_joint_emits_rigid_child_without_joint_or_motor():
    genotype = tree_genotype(
        root="body",
        nodes={
            "body": tree_node(
                name="body",
                size=(0.2, 0.2, 0.2),
                joint_type="free",
                children=[ConnectionGene(child="limb", axis=(0, 1, 0))],
            ),
            "limb": NodeGene(
                name="limb",
                size=(0.1, 0.1, 0.1),
                joint_type="fixed",
            ),
        },
    )

    mjcf = PhenotypeBuilder(genotype, make_definition(max_body_count=2)).build()
    xml_root = ET.fromstring(mjcf)
    child_body = xml_root.find("./worldbody/body/body")

    assert child_body.find("joint") is None
    assert xml_root.findall("./actuator/motor") == []

    model = mujoco.MjModel.from_xml_string(mjcf)
    assert model.nbody == 3
    assert model.nv == 6
    assert model.nu == 0


def test_oriented_root_and_child_emit_quat_and_compile():
    genotype = tree_genotype(
        root="body",
        nodes={
            "body": tree_node(
                name="body",
                size=(0.2, 0.2, 0.2),
                joint_type="free",
                orientation=(0.0, 0.0, 45.0),
                children=[ConnectionGene(
                    child="limb",
                    axis=(0, 1, 0),
                    orientation=(0.0, 0.0, 30.0),
                )],
            ),
            "limb": NodeGene(name="limb", size=(0.1, 0.05, 0.05)),
        },
    )

    mjcf = PhenotypeBuilder(genotype, make_definition(max_body_count=2)).build()
    xml_root = ET.fromstring(mjcf)
    root_body = xml_root.find("./worldbody/body")
    child_body = root_body.find("body")

    assert root_body.get("quat") != "1.0 0.0 0.0 0.0"
    assert child_body.get("quat") != "1.0 0.0 0.0 0.0"
    model = mujoco.MjModel.from_xml_string(mjcf)
    assert model.nbody == 3


def test_rotated_child_joint_lies_on_parent_and_child_surfaces():
    genotype = build_genotype(
        root="body",
        spec={
            "body": {
                "size": (2.0, 1.0, 0.5),
                "joint_type": "free",
                "orientation": (0.0, 0.0, 45.0),
                "children": [{
                    "child": "limb",
                    "axis": (0, 1, 0),
                    "parent_face": "+x",
                    "surface_uv": (0.25, -0.5),
                    "child_surface_uv": (-0.25, 0.75),
                    "orientation": (0.0, 0.0, 45.0),
                }],
            },
            "limb": {"size": (0.4, 0.3, 0.2), "joint_type": "hinge"},
        },
    )

    mjcf = PhenotypeBuilder(genotype, make_definition(max_body_count=10)).build()
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    parent_local_joint = _joint_position_in_body_frame(
        model,
        data,
        "limb_joint_2",
        "body_1",
    )

    assert parent_local_joint == pytest.approx(np.array((2.0, 0.25, -0.25)))
    _assert_joint_on_child_attachment_face(
        model,
        data,
        "limb_joint_2",
        "limb_2_geom",
        (-0.25, 0.75),
    )


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
                    "child_surface_uv": (0.25, -0.75),
                }],
            },
            "limb": {"size": (0.4, 0.3, 0.2), "joint_type": "hinge"},
        },
    )

    xml_root = ET.fromstring(PhenotypeBuilder(genotype, make_definition(max_body_count=10)).build())
    root_body = xml_root.find("./worldbody/body")
    child_body = root_body.find("body")
    child_geom = child_body.find("geom")
    child_joint = child_body.find("joint")

    assert _vector(child_body, "pos") == pytest.approx((2.0, 0.5, -0.25))
    assert _vector(child_joint, "pos") == pytest.approx((0.0, 0.0, 0.0))
    assert _vector(child_geom, "pos") == pytest.approx((0.4, -0.075, 0.15))

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
    genotype = tree_genotype(
        root="body",
        nodes={
            "body": tree_node(
                name="body",
                size=(1.0, 1.0, 1.0),
                joint_type="free",
                children=[ConnectionGene(child="segment", axis=(0, 1, 0))],
            ),
            "segment": tree_node(
                name="segment",
                size=(0.5, 0.25, 0.1),
                recursive_limit=3,
                children=[recursive_connection],
            ),
        },
    )

    xml_root = ET.fromstring(PhenotypeBuilder(genotype, make_definition(max_body_count=10)).build())
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
    genotype = tree_genotype(
        root="body",
        nodes={
            "body": tree_node(
                name="body",
                size=(1.0, 1.0, 1.0),
                joint_type="free",
                children=[ConnectionGene(child="segment", axis=(0, 1, 0))],
            ),
            "segment": tree_node(
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

    assert GenotypeGraphAnalyzer(genotype, max_body_count=5).validate() == 5
    xml_root = ET.fromstring(PhenotypeBuilder(genotype, make_definition(max_body_count=5)).build())
    first_segment = xml_root.find("./worldbody/body/body")
    second_segment = first_segment.find("./body")
    terminal_segment = second_segment.find("./body")

    assert len(first_segment.findall("./body")) == 1
    assert len(second_segment.findall("./body")) == 1
    terminal_children = terminal_segment.findall("./body")
    assert len(terminal_children) == 1
    assert terminal_children[0].get("name").startswith("tip_")


def test_slide_and_ball_joints_compile_with_expected_actuators():
    genotype = tree_genotype(
        root="body",
        nodes={
            "body": tree_node(
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

    mjcf = PhenotypeBuilder(genotype, make_definition(max_body_count=3)).build()
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
        (50.0, 0.0, 0.0)
    )
    assert tuple(float(value) for value in motors[2].get("gear").split()) == pytest.approx(
        (0.0, 50.0, 0.0)
    )
    assert tuple(float(value) for value in motors[3].get("gear").split()) == pytest.approx(
        (0.0, 0.0, 50.0)
    )

    model = mujoco.MjModel.from_xml_string(mjcf)
    assert model.nv == 10
    assert model.nu == 4


def test_ball_neural_motors_use_persisted_output_axes():
    connection = ConnectionGene(
        child="ball",
        axis=(0.0, 1.0, 0.0),
        motor_gear=50.0,
        neural_output_axes=(
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        ),
    )
    genotype = tree_genotype(
        root="body",
        nodes={
            "body": tree_node(
                name="body",
                size=(1.0, 1.0, 1.0),
                joint_type="free",
                children=[connection],
            ),
            "ball": NodeGene(
                name="ball",
                size=(0.2, 0.1, 0.1),
                joint_type="ball",
            ),
        },
    )

    xml_root = ET.fromstring(PhenotypeBuilder(genotype, make_definition(max_body_count=2)).build())
    gears = [
        tuple(float(value) for value in motor.get("gear").split())
        for motor in xml_root.findall("./actuator/motor")
    ]

    assert gears == [
        (0.0, 50.0, 0.0),
        (0.0, 0.0, 50.0),
        (50.0, 0.0, 0.0),
    ]


def test_slide_axis_uses_vector_reflection_for_symmetry():
    genotype = tree_genotype(
        root="body",
        nodes={
            "body": tree_node(
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

    xml_root = ET.fromstring(PhenotypeBuilder(genotype, make_definition(max_body_count=3)).build())
    joints = [
        body.find("joint")
        for body in xml_root.findall("./worldbody/body/body")
    ]

    assert [_vector(joint, "axis") for joint in joints] == [
        pytest.approx((1.0, 0.0, 0.0)),
        pytest.approx((-1.0, 0.0, 0.0)),
    ]


def test_recursive_attachment_uses_parent_geom_center():
    genotype = tree_genotype(
        root="body",
        nodes={
            "body": tree_node(
                name="body",
                size=(2.0, 1.0, 0.5),
                joint_type="free",
                children=[ConnectionGene(
                    child="segment",
                    axis=(0, 1, 0),
                    parent_face="+x",
                )],
            ),
            "segment": tree_node(
                name="segment",
                size=(0.4, 0.3, 0.2),
                children=[ConnectionGene(
                    child="tip",
                    axis=(1, 0, 0),
                    parent_face="+y",
                    orientation=(0.0, 0.0, 90.0),
                )],
            ),
            "tip": NodeGene(name="tip", size=(0.1, 0.2, 0.1)),
        },
    )

    xml_root = ET.fromstring(PhenotypeBuilder(genotype, make_definition(max_body_count=10)).build())
    root_body = xml_root.find("./worldbody/body")
    segment_body = root_body.find("body")
    tip_body = segment_body.find("body")

    assert _vector(segment_body, "pos") == pytest.approx((2.0, 0.0, 0.0))
    assert _vector(segment_body.find("geom"), "pos") == pytest.approx(
        (0.4, 0.0, 0.0)
    )
    assert _vector(tip_body, "pos") == pytest.approx((0.4, 0.3, 0.0))
    assert _vector(tip_body.find("geom"), "pos") == pytest.approx(
        (0.1, 0.0, 0.0)
    )


def test_symmetric_rotated_children_attach_at_child_surface():
    genotype = tree_genotype(
        root="body",
        nodes={
            "body": tree_node(
                name="body",
                size=(1.0, 1.0, 1.0),
                joint_type="free",
                children=[ConnectionGene(
                    child="limb",
                    axis=(0, 1, 0),
                    parent_face="+x",
                    orientation=(0.0, 0.0, 45.0),
                    child_surface_uv=(0.25, -0.5),
                    symmetry=("xz",),
                )],
            ),
            "limb": NodeGene(
                name="limb",
                size=(0.4, 0.3, 0.2),
                joint_type="hinge",
            ),
        },
    )

    mjcf = PhenotypeBuilder(genotype, make_definition(max_body_count=3)).build()
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    for index in (2, 3):
        _assert_joint_on_child_attachment_face(
            model,
            data,
            f"limb_joint_{index}",
            f"limb_{index}_geom",
            (0.25, -0.5),
        )
