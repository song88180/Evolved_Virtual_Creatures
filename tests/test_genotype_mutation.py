import random

import pytest

from evol_virtual_creature.genes import ConnectionGene, NodeGene
from evol_virtual_creature.genotype import Genotype
from evol_virtual_creature.graph_analysis import (
    GenotypeGraphAnalyzer,
    GenotypeGraphError,
)


def test_fresh_node_connection_addition_can_grow_single_free_root():
    genotype = Genotype(
        root="body",
        nodes={
            "body": NodeGene(
                name="body",
                size=(0.25, 0.15, 0.1),
                joint_type="free",
                recursive_limit=1,
            )
        },
    )

    description = genotype._mutate_parameter_path(
        ("fresh_node_connection_addition",),
        random.Random(1),
    )

    assert "fresh node-connection addition" in description
    assert "randomized" in description
    assert len(genotype.nodes) == 2
    assert len(genotype.nodes["body"].children) == 1
    connection = genotype.nodes["body"].children[0]
    child_name = connection.child
    assert child_name in genotype.nodes

    child = genotype.nodes[child_name]
    assert child.joint_type == "hinge"
    assert all(0.04 <= value <= 0.30 for value in child.size)
    assert all(-1.0 <= value <= 1.0 for value in child.joint_axis)
    assert 1 <= child.recursive_limit <= 8
    assert child.size != (0.1, 0.05, 0.05)
    assert child.joint_axis != (0.0, 1.0, 0.0)

    assert connection.parent_face in {"+x", "-x", "+y", "-y", "+z", "-z"}
    assert all(-1.0 <= value <= 1.0 for value in connection.surface_uv)
    assert set(connection.symmetry) <= {"xy", "xz", "yz"}
    assert all(-1.0 <= value <= 1.0 for value in connection.axis)
    assert 0.5 <= connection.scale <= 1.5
    assert 20.0 <= connection.motor_gear <= 200.0
    assert -2.0 <= connection.ctrlrange[0] < 0.0
    assert 0.0 < connection.ctrlrange[1] <= 2.0
    assert 0.02 <= connection.control_amp <= 0.50
    assert 1.0 <= connection.control_freq <= 15.0
    assert -3.141592653589793 <= connection.control_phase <= 3.141592653589793
    assert -2.0 <= connection.control_phase_depth_scale <= 2.0
    assert -2.0 <= connection.control_phase_order_scale <= 2.0
    assert (connection.parent_face, connection.surface_uv) != (
        "+x",
        (0.0, 0.0),
    )
    assert connection.axis != (0.0, 1.0, 0.0)


def test_fresh_node_connection_addition_is_available_as_topology_mutation():
    genotype = Genotype(
        root="body",
        nodes={
            "body": NodeGene(
                name="body",
                size=(0.25, 0.15, 0.1),
                joint_type="free",
                recursive_limit=1,
            )
        },
    )

    mutable_parameters = genotype._collect_mutable_parameters(
        allow_topology_mutations=True
    )

    assert ("fresh_node_connection_addition",) in mutable_parameters


def test_terminal_only_mutation_cannot_enable_on_self_connection():
    connection = ConnectionGene(child="segment", axis=(0, 1, 0))
    genotype = Genotype(
        root="segment",
        nodes={
            "segment": NodeGene(
                name="segment",
                size=(0.2, 0.1, 0.1),
                joint_type="free",
                recursive_limit=2,
                children=[connection],
            ),
        },
    )

    description = genotype._mutate_parameter_path(
        ("connection", connection, "terminal_only"),
        random.Random(1),
    )

    assert not connection.terminal_only
    assert "cannot be terminal-only" in description


def test_terminal_only_self_connection_is_rejected_during_build_analysis():
    genotype = Genotype(
        root="segment",
        nodes={
            "segment": NodeGene(
                name="segment",
                size=(0.2, 0.1, 0.1),
                joint_type="free",
                children=[
                    ConnectionGene(
                        child="segment",
                        axis=(0, 1, 0),
                        terminal_only=True,
                    )
                ],
            ),
        },
    )

    with pytest.raises(GenotypeGraphError, match="cannot point to its own node"):
        GenotypeGraphAnalyzer(genotype, max_node=10).validate()


def test_joint_type_mutation_excludes_free_root_and_mutates_articulated_nodes():
    genotype = Genotype(
        root="body",
        nodes={
            "body": NodeGene(
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

    mutable_parameters = genotype._collect_mutable_parameters(
        allow_topology_mutations=False
    )
    assert ("node", "body", "joint_type") not in mutable_parameters
    assert ("node", "limb", "joint_type") in mutable_parameters

    description = genotype._mutate_parameter_path(
        ("node", "limb", "joint_type"),
        random.Random(1),
    )
    assert genotype.nodes["body"].joint_type == "free"
    assert genotype.nodes["limb"].joint_type in {"slide", "ball"}
    assert "joint_type" in description


def test_node_rejects_unknown_joint_type():
    with pytest.raises(ValueError, match="Unknown joint type"):
        NodeGene(name="bad", size=(0.1, 0.1, 0.1), joint_type="fixed")


def test_template_new_node_addition_attaches_connection_before_mutating_it():
    genotype = Genotype(
        root="body",
        nodes={
            "body": NodeGene(
                name="body",
                size=(0.25, 0.15, 0.1),
                joint_type="free",
                recursive_limit=1,
            ),
            "segment": NodeGene(
                name="segment",
                size=(0.12, 0.06, 0.06),
                joint_type="hinge",
                recursive_limit=1,
            ),
        },
    )

    description = genotype._mutate_connection_new_node_addition(random.Random(1))

    assert "connection new-node addition" in description
    assert any(node.children for node in genotype.nodes.values())
