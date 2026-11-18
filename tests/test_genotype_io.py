"""Tests for evc3d.genotype_io: JSON loading, validation, migration, saving, and control-mode conversion."""

import json
import warnings

import pytest

from evc3d.genes import ConnectionGene, NodeGene
from evc3d.genotype import Genotype
from evc3d.genotype_io import (
    build_genotype,
    convert_control_mode,
    load_genotype_from_json,
    save_genotype_to_json,
)
from evc3d.phenotype import PhenotypeBuilder
from helpers import make_definition, tree_genotype, tree_node


def _two_mode_genotype(control_mode):
    return build_genotype(
        root="body",
        nodes={
            "body": {
                "size": (0.2, 0.2, 0.2),
                "joint_type": "free",
                "child_connections": ["body_to_limb", "body_to_ball"],
            },
            "limb": {"size": (0.1, 0.1, 0.1)},
            "ball": {"size": (0.1, 0.1, 0.1), "joint_type": "ball"},
        },
        connections={
            "body_to_limb": {
                "child": "limb",
                "axis": (0.0, 1.0, 0.0),
                "control_amp": 0.4,
                "control_freq": 7,
                "control_phase": 1.2,
            },
            "body_to_ball": {"child": "ball", "axis": (0.0, 1.0, 0.0)},
        },
        global_control_freq=2.5,
        control_mode=control_mode,
    )


def test_loading_duplicate_active_and_archived_node_names_fails(tmp_path):
    genotype_path = tmp_path / "duplicate_names.json"
    genotype_path.write_text(json.dumps({
        "control_mode": "neural",
        "root": "body",
        "nodes": {
            "body": {
                "size": [0.1, 0.1, 0.1],
                "joint_type": "free",
            },
        },
        "connections": {},
        "archived_connections": {},
        "archived_nodes": {"body": {
            "size": [0.2, 0.2, 0.2],
            "joint_type": "free",
        }},
    }))

    with pytest.raises(ValueError, match="Duplicate node gene name 'body'"):
        load_genotype_from_json(genotype_path)


def test_legacy_embedded_connections_are_rejected(tmp_path):
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

    with pytest.raises(ValueError, match="Embedded 'children' is unsupported"):
        load_genotype_from_json(genotype_path)


def test_missing_orientation_loads_with_identity_for_default_face(tmp_path):
    genotype_path = tmp_path / "identity_orientation.json"
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
        },
        "connections": {"body_to_limb": {"child": "limb", "axis": [0, 1, 0]}},
    }))

    genotype = load_genotype_from_json(genotype_path)

    assert genotype.nodes["body"].orientation == (0.0, 0.0, 0.0)
    assert genotype.outgoing_connections("body")[0].orientation == (0.0, 0.0, 0.0)
    assert genotype.outgoing_connections("body")[0].child_surface_uv == (0.0, 0.0)


def test_orientation_round_trips_through_json(tmp_path):
    genotype = tree_genotype(
        root="body",
        nodes={
            "body": tree_node(
                name="body",
                size=(0.2, 0.2, 0.2),
                joint_type="free",
                orientation=(10.0, 20.0, 30.0),
                children=[ConnectionGene(
                    child="limb",
                    axis=(0, 1, 0),
                    orientation=(0.0, 0.0, 30.0),
                    child_surface_uv=(0.25, -0.5),
                )],
            ),
            "limb": NodeGene(name="limb", size=(0.1, 0.1, 0.1)),
        },
    )
    path = tmp_path / "oriented.json"

    save_genotype_to_json(genotype, path)
    data = json.loads(path.read_text())
    loaded = load_genotype_from_json(path)

    assert data["nodes"]["body"]["orientation"] == [10.0, 20.0, 30.0]
    connection_name = data["nodes"]["body"]["child_connections"][0]
    assert data["connections"][connection_name]["orientation"] == [0.0, 0.0, 30.0]
    assert data["connections"][connection_name]["child_surface_uv"] == [0.25, -0.5]
    assert loaded.nodes["body"].orientation == (10.0, 20.0, 30.0)
    assert loaded.outgoing_connections("body")[0].orientation == (0.0, 0.0, 30.0)
    assert loaded.outgoing_connections("body")[0].child_surface_uv == (0.25, -0.5)


def test_named_connection_can_be_shared_by_different_nodes(tmp_path):
    path = tmp_path / "shared.json"
    path.write_text(json.dumps({
        "control_mode": "neural",
        "root": "body",
        "nodes": {
            "body": {
                "size": [0.2, 0.2, 0.2],
                "joint_type": "free",
                "child_connections": ["to_limb"],
            },
            "branch": {
                "size": [0.1, 0.1, 0.1],
                "child_connections": ["to_limb"],
            },
            "limb": {"size": [0.05, 0.05, 0.05]},
        },
        "connections": {
            "to_limb": {"child": "limb", "axis": [0, 1, 0]},
        },
    }))

    genotype = load_genotype_from_json(path)

    assert genotype.outgoing_connections("body")[0] is genotype.connections["to_limb"]
    assert genotype.outgoing_connections("branch")[0] is genotype.connections["to_limb"]


def test_duplicate_connection_reference_in_one_node_is_rejected():
    connection = ConnectionGene(name="shared", child="limb", axis=(0, 1, 0))
    with pytest.raises(ValueError, match="duplicate child connection"):
        Genotype(
            root="body",
            nodes={
                "body": NodeGene(
                    name="body",
                    size=(0.2, 0.2, 0.2),
                    joint_type="free",
                    child_connections=["shared", "shared"],
                ),
                "limb": NodeGene(name="limb", size=(0.1, 0.1, 0.1)),
            },
            connections={"shared": connection},
        )


def test_removing_one_shared_reference_keeps_connection_active():
    connection = ConnectionGene(name="shared", child="limb", axis=(0, 1, 0))
    genotype = Genotype(
        root="body",
        nodes={
            "body": NodeGene(
                name="body", size=(0.2, 0.2, 0.2), joint_type="free",
                child_connections=["shared"],
            ),
            "branch": NodeGene(
                name="branch", size=(0.1, 0.1, 0.1),
                child_connections=["shared"],
            ),
            "limb": NodeGene(name="limb", size=(0.1, 0.1, 0.1)),
        },
        connections={"shared": connection},
    )

    genotype._mutate_connection_removal(connection, "body")

    assert "shared" not in genotype.nodes["body"].child_connections
    assert genotype.nodes["branch"].child_connections == ["shared"]
    assert genotype.connections["shared"] is connection


def test_legacy_connection_control_mode_is_rejected(tmp_path):
    genotype_path = tmp_path / "legacy_sine.json"
    genotype_path.write_text(json.dumps({
        "root": "body",
        "nodes": {
            "body": {
                "size": [0.2, 0.2, 0.2],
                "joint_type": "free",
                "children": [{
                    "child": "limb",
                    "axis": [0, 1, 0],
                    "control_mode": "sine",
                }],
            },
            "limb": {
                "size": [0.1, 0.1, 0.1],
                "joint_type": "hinge",
            },
        },
    }))

    with pytest.raises(ValueError, match="Embedded 'children' is unsupported"):
        load_genotype_from_json(genotype_path)


def test_mixed_legacy_connection_control_modes_are_rejected(tmp_path):
    genotype_path = tmp_path / "mixed_legacy_control.json"
    genotype_path.write_text(json.dumps({
        "root": "body",
        "nodes": {
            "body": {
                "size": [0.2, 0.2, 0.2],
                "joint_type": "free",
                "children": [
                    {
                        "child": "limb",
                        "axis": [0, 1, 0],
                        "control_mode": "neural",
                    },
                    {
                        "child": "tail",
                        "axis": [0, 1, 0],
                        "control_mode": "sine",
                    },
                ],
            },
            "limb": {"size": [0.1, 0.1, 0.1], "joint_type": "hinge"},
            "tail": {"size": [0.1, 0.1, 0.1], "joint_type": "hinge"},
        },
    }))

    with pytest.raises(ValueError, match="Embedded 'children' is unsupported"):
        load_genotype_from_json(genotype_path)


def test_missing_neural_fields_load_as_zero_neural_controller(tmp_path):
    genotype_path = tmp_path / "legacy_neural.json"
    genotype_path.write_text(json.dumps({
        "root": "body",
        "nodes": {
            "body": {
                "size": [0.2, 0.2, 0.2],
                "joint_type": "free",
                "child_connections": ["body_to_limb"],
            },
            "limb": {
                "size": [0.1, 0.1, 0.1],
                "joint_type": "hinge",
            },
        },
        "control_mode": "neural",
        "connections": {"body_to_limb": {"child": "limb", "axis": [0, 1, 0]}},
    }))

    genotype = load_genotype_from_json(genotype_path)
    connection = genotype.outgoing_connections("body")[0]

    assert genotype.control_mode == "neural"
    assert len(connection.neural_w1) == 16
    assert len(connection.neural_w1[0]) == 12
    assert len(connection.neural_w2) == 1
    assert len(connection.neural_w2[0]) == 16
    assert all(value == 0.0 for row in connection.neural_w1 for value in row)
    assert all(value == 0.0 for value in connection.neural_b1)
    assert connection.neural_output_axes == ((0.0, 1.0, 0.0),)


def test_global_control_frequency_round_trips_through_json(tmp_path):
    genotype = Genotype(
        root="body",
        global_control_freq=2.5,
        nodes={
            "body": NodeGene(
                name="body",
                size=(0.2, 0.2, 0.2),
                joint_type="free",
            ),
        },
    )
    path = tmp_path / "global_freq.json"

    save_genotype_to_json(genotype, path)
    data = json.loads(path.read_text())
    loaded = load_genotype_from_json(path)

    assert data["global_control_freq"] == pytest.approx(2.5)
    assert loaded.global_control_freq == pytest.approx(2.5)


def test_missing_global_control_frequency_loads_default(tmp_path):
    genotype_path = tmp_path / "missing_global_freq.json"
    genotype_path.write_text(
        json.dumps({
            "control_mode": "neural",
            "root": "body",
            "nodes": {
                "body": {"size": [0.2, 0.2, 0.2], "joint_type": "free"},
            },
            "connections": {},
        })
    )

    genotype = load_genotype_from_json(genotype_path)

    assert genotype.global_control_freq == pytest.approx(1.0)


def test_shape_round_trips_through_json(tmp_path):
    genotype = Genotype(
        root="body",
        nodes={
            "body": NodeGene(
                name="body",
                size=(0.2, 0.2, 0.2),
                joint_type="free",
                shape="ellipsoid",
            ),
            "limb": NodeGene(name="limb", size=(0.1, 0.08, 0.04), shape="capsule"),
        },
    )
    path = tmp_path / "shaped.json"

    save_genotype_to_json(genotype, path)
    data = json.loads(path.read_text())
    loaded = load_genotype_from_json(path)

    assert data["nodes"]["body"]["shape"] == "ellipsoid"
    assert data["nodes"]["limb"]["shape"] == "capsule"
    assert loaded.nodes["body"].shape == "ellipsoid"
    assert loaded.nodes["limb"].shape == "capsule"
    assert loaded.nodes["limb"].size == (0.1, 0.04, 0.04)


def test_missing_shape_loads_as_box(tmp_path):
    genotype_path = tmp_path / "missing_shape.json"
    genotype_path.write_text(
        json.dumps({
            "control_mode": "neural",
            "root": "body",
            "nodes": {
                "body": {"size": [0.2, 0.2, 0.2], "joint_type": "free"},
            },
            "connections": {},
        })
    )

    genotype = load_genotype_from_json(genotype_path)

    assert genotype.nodes["body"].shape == "box"


def test_out_of_bounds_json_sizes_warn_and_load_unchanged(tmp_path):
    genotype_path = tmp_path / "out_of_bounds.json"
    genotype_path.write_text(json.dumps({
        "control_mode": "neural",
        "root": "body",
        "nodes": {
            "body": {
                "size": [0.0005, 0.5, 1.5],
                "joint_type": "free",
            },
        },
        "connections": {},
    }))

    with pytest.warns(UserWarning, match=r"Body part 'body'.*\[0.01, 1.0\]"):
        genotype = load_genotype_from_json(genotype_path)

    assert genotype.nodes["body"].size == (0.0005, 0.5, 1.5)


def test_body_size_bounds_are_inclusive_when_loading_json(tmp_path):
    genotype_path = tmp_path / "boundary_sizes.json"
    genotype_path.write_text(json.dumps({
        "control_mode": "neural",
        "root": "body",
        "nodes": {
            "body": {
                "size": [0.01, 0.5, 1.0],
                "joint_type": "free",
            },
        },
        "connections": {},
    }))

    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        genotype = load_genotype_from_json(genotype_path)

    assert not warnings_record
    assert genotype.nodes["body"].size == (0.01, 0.5, 1.0)


def test_missing_control_mode_is_rejected(tmp_path):
    genotype_path = tmp_path / "unlabeled.json"
    genotype_path.write_text(json.dumps({
        "root": "body",
        "nodes": {"body": {"size": [0.2, 0.2, 0.2], "joint_type": "free"}},
        "connections": {},
    }))

    with pytest.raises(ValueError, match="top-level 'control_mode'"):
        load_genotype_from_json(genotype_path)


def test_convert_sine_to_neural_zeroes_controllers():
    original = _two_mode_genotype("sine")

    converted = convert_control_mode(original, "neural")

    assert converted.control_mode == "neural"
    assert converted.global_control_freq == 1.0
    hinge = converted.connections["body_to_limb"]
    assert (hinge.control_amp, hinge.control_freq, hinge.control_phase) == (0.0, 10, 0.0)
    assert len(hinge.neural_w1) == 16 and len(hinge.neural_b2) == 1
    assert all(value == 0.0 for row in hinge.neural_w1 for value in row)
    assert hinge.neural_output_axes == ((0.0, 1.0, 0.0),)
    ball = converted.connections["body_to_ball"]
    assert len(ball.neural_b2) == 3 and len(ball.neural_output_axes) == 3
    # The input genotype is left untouched.
    assert original.control_mode == "sine"
    assert original.connections["body_to_limb"].control_amp == 0.4
    assert original.connections["body_to_limb"].neural_w1 == ()


def test_convert_neural_to_sine_clears_neural_and_zeroes_amplitude():
    original = _two_mode_genotype("neural")

    converted = convert_control_mode(original, "sine")

    assert converted.control_mode == "sine"
    assert converted.global_control_freq == 1.0
    for connection in converted.connections.values():
        assert connection.control_amp == 0.0
        assert connection.neural_w1 == ()
        assert connection.neural_b2 == ()
        assert connection.neural_output_axes == ()
    assert original.connections["body_to_limb"].neural_w1 != ()
    assert PhenotypeBuilder(converted, make_definition(max_body_count=10)).build()


def test_convert_to_same_control_mode_keeps_parameters():
    original = _two_mode_genotype("sine")

    converted = convert_control_mode(original, "sine")

    assert converted is not original
    assert converted.connections["body_to_limb"].control_amp == 0.4
    assert converted.global_control_freq == 2.5
