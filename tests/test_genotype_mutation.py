import random

import pytest

from evol_virtual_creature.genes import (
    BODY_SHAPES,
    ROUND_BODY_SHAPES,
    ConnectionGene,
    NodeGene,
    child_orientation_is_valid,
)
from evol_virtual_creature.genotype import Genotype
from evol_virtual_creature.phenotype import PhenotypeBuilder
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
    assert child.joint_type in {"fixed", "hinge", "ball"}
    assert child.shape in BODY_SHAPES
    if child.shape in ROUND_BODY_SHAPES:
        assert child.size[1] == child.size[2]
    assert all(
        parent_dimension * genotype.NEW_CHILD_SIZE_MIN_SCALE
        <= child_dimension
        <= parent_dimension * genotype.NEW_CHILD_SIZE_MAX_SCALE
        for child_dimension, parent_dimension in zip(
            child.size,
            genotype.nodes[genotype.root].size,
        )
    )
    assert 1 <= child.recursive_limit <= 8
    assert child.size != (0.1, 0.05, 0.05)

    assert connection.parent_face in {"+x", "-x", "+y", "-y", "+z", "-z"}
    assert all(-1.0 <= value <= 1.0 for value in connection.surface_uv)
    assert all(-1.0 <= value <= 1.0 for value in connection.child_surface_uv)
    assert set(connection.symmetry) <= {"xy", "xz", "yz"}
    assert all(-1.0 <= value <= 1.0 for value in connection.axis)
    assert all(-180.0 <= value <= 180.0 for value in connection.orientation)
    assert child_orientation_is_valid(connection.parent_face, connection.orientation)
    assert 0.5 <= connection.scale <= 1.5
    assert 20.0 <= connection.motor_gear <= 200.0
    assert -2.0 <= connection.ctrlrange[0] < 0.0
    assert 0.0 < connection.ctrlrange[1] <= 2.0
    assert 0.02 <= connection.control_amp <= 0.50
    assert connection.control_freq in genotype.HARMONIC_CONTROL_FREQS
    assert -3.141592653589793 <= connection.control_phase <= 3.141592653589793
    assert -2.0 <= connection.control_phase_depth_scale <= 2.0
    assert -2.0 <= connection.control_phase_order_scale <= 2.0
    assert (connection.parent_face, connection.surface_uv) != (
        "+x",
        (0.0, 0.0),
    )
    assert connection.child_surface_uv != (0.0, 0.0)
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


def test_topology_mutation_rate_floor_applies_to_topology_operators():
    genotype = Genotype(
        root="body",
        nodes={
            "body": NodeGene(
                name="body",
                size=(0.25, 0.15, 0.1),
                joint_type="free",
            )
        },
    )

    assert genotype._mutation_rate_for_parameter(
        ("connection_addition",),
        mutation_rate=0.01,
        topology_mutation_rate_min=0.05,
        allow_topology_mutations=True,
    ) == pytest.approx(0.05)
    assert genotype._mutation_rate_for_parameter(
        ("node", "body", "size", 0),
        mutation_rate=0.01,
        topology_mutation_rate_min=0.05,
        allow_topology_mutations=True,
    ) == pytest.approx(0.01)
    assert genotype._mutation_rate_for_parameter(
        ("connection_addition",),
        mutation_rate=0.01,
        topology_mutation_rate_min=0.05,
        allow_topology_mutations=False,
    ) == pytest.approx(0.01)


def test_neural_weights_do_not_use_topology_mutation_rate_floor():
    genotype = Genotype(
        root="body",
        nodes={
            "body": NodeGene(
                name="body",
                size=(0.25, 0.15, 0.1),
                joint_type="free",
            )
        },
    )

    assert genotype._mutation_rate_for_parameter(
        ("connection_neural", object(), "neural_w1", 0, 0),
        mutation_rate=0.01,
        topology_mutation_rate_min=0.05,
        allow_topology_mutations=True,
    ) == pytest.approx(0.01)


def test_count_based_mutation_adds_topology_floor_mutations():
    genotype = Genotype(
        root="body",
        nodes={
            "body": NodeGene(
                name="body",
                size=(0.25, 0.15, 0.1),
                joint_type="free",
            )
        },
    )
    selected_parameters = [("node", "body", "size", 0)]
    mutable_parameters = [
        ("node", "body", "size", 0),
        ("connection_addition",),
        ("connection_new_node_addition",),
    ]

    genotype._add_topology_floor_mutations(
        selected_parameters,
        mutable_parameters,
        topology_mutation_rate_min=1.0,
        random_source=random.Random(1),
        allow_topology_mutations=True,
    )

    assert selected_parameters == [
        ("node", "body", "size", 0),
        ("connection_addition",),
        ("connection_new_node_addition",),
    ]


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
    assert genotype.nodes["limb"].joint_type in {"fixed", "ball"}
    assert "joint_type" in description


def test_joint_type_mutation_allows_slide_when_enabled():
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
    genotype._allow_slide_joint_for_mutation = True

    genotype._mutate_parameter_path(
        ("node", "limb", "joint_type"),
        random.Random(0),
    )

    assert genotype.nodes["limb"].joint_type == "slide"


def test_shape_mutation_repairs_round_cross_section():
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
                size=(0.3, 0.2, 0.1),
                joint_type="hinge",
            ),
        },
    )

    genotype._mutate_parameter_path(("node", "limb", "shape"), random.Random(5))

    limb = genotype.nodes["limb"]
    assert limb.shape in BODY_SHAPES
    if limb.shape in ROUND_BODY_SHAPES:
        assert limb.size[1] == limb.size[2]


def test_node_accepts_fixed_joint_type():
    node = NodeGene(name="rigid", size=(0.1, 0.1, 0.1), joint_type="fixed")

    assert node.joint_type == "fixed"


def test_node_rejects_unknown_joint_type():
    with pytest.raises(ValueError, match="Unknown joint type"):
        NodeGene(name="bad", size=(0.1, 0.1, 0.1), joint_type="welded")


def test_node_validates_and_normalizes_body_shape():
    assert NodeGene(name="round", size=(0.3, 0.2, 0.1), shape="capsule").size == (
        0.3,
        0.1,
        0.1,
    )
    node = NodeGene(name="body", size=(0.3, 0.2, 0.1), shape="ellipsoid")
    assert node.shape == "ellipsoid"
    with pytest.raises(ValueError, match="Unknown body shape"):
        NodeGene(name="bad", size=(0.1, 0.1, 0.1), shape="cone")


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


def test_template_new_child_node_size_scales_from_connection_parent(monkeypatch):
    genotype = Genotype(
        root="body",
        nodes={
            "body": NodeGene(
                name="body",
                size=(0.4, 0.2, 0.1),
                joint_type="free",
            ),
            "segment": NodeGene(
                name="segment",
                size=(0.02, 0.02, 0.02),
                joint_type="hinge",
            ),
        },
    )

    def force_oversized_child(node, _rng):
        node.size = (10.0, 10.0, 10.0)
        return "size"

    monkeypatch.setattr(genotype, "_mutate_new_node", force_oversized_child)

    genotype._mutate_connection_new_node_addition(random.Random(1))

    new_node_name = next(
        name for name in genotype.nodes if name.startswith("segment_mut")
    )
    parent_node = next(
        node
        for node in genotype.nodes.values()
        if any(connection.child == new_node_name for connection in node.children)
    )
    child_node = genotype.nodes[new_node_name]
    assert all(
        parent_dimension * genotype.NEW_CHILD_SIZE_MIN_SCALE
        <= child_dimension
        <= parent_dimension * genotype.NEW_CHILD_SIZE_MAX_SCALE
        for child_dimension, parent_dimension in zip(
            child_node.size,
            parent_node.size,
        )
    )


def test_destination_replacement_child_node_size_scales_from_connection_parent(
    monkeypatch,
):
    connection = ConnectionGene(child="limb", axis=(0, 1, 0))
    genotype = Genotype(
        root="body",
        nodes={
            "body": NodeGene(
                name="body",
                size=(0.5, 0.25, 0.125),
                joint_type="free",
                children=[connection],
            ),
            "limb": NodeGene(
                name="limb",
                size=(0.03, 0.03, 0.03),
                joint_type="hinge",
            ),
        },
    )

    def force_undersized_child(node, _rng):
        node.size = (0.0, 0.0, 0.0)
        return "size"

    monkeypatch.setattr(genotype, "_mutate_new_node", force_undersized_child)

    genotype._mutate_connection_destination_node_replacement(
        connection,
        random.Random(1),
    )

    child_node = genotype.nodes[connection.child]
    parent_node = genotype.nodes[genotype.root]
    assert all(
        parent_dimension * genotype.NEW_CHILD_SIZE_MIN_SCALE
        <= child_dimension
        <= parent_dimension * genotype.NEW_CHILD_SIZE_MAX_SCALE
        for child_dimension, parent_dimension in zip(
            child_node.size,
            parent_node.size,
        )
    )

def test_preselected_neural_mutation_skips_archived_replaced_connection():
    connection = ConnectionGene(
        child="limb",
        axis=(0, 1, 0),
    )
    genotype = Genotype(
        root="body",
        nodes={
            "body": NodeGene(
                name="body",
                size=(0.2, 0.2, 0.2),
                joint_type="free",
                children=[connection],
            ),
            "limb": NodeGene(
                name="limb",
                size=(0.1, 0.1, 0.1),
                joint_type="hinge",
            ),
        },
    )
    genotype._ensure_connection_neural_parameters(connection)

    genotype._mutate_parameter_path(
        ("connection_replacement", connection),
        random.Random(1),
    )
    description = genotype._mutate_parameter_path(
        ("connection_neural", connection, "neural_b1", 0),
        random.Random(2),
    )

    assert connection in genotype.archived_connections
    assert "unchanged; connection is archived" in description


def _neural_hinge_genotype():
    connection = ConnectionGene(
        child="limb",
        axis=(0, 1, 0),
    )
    genotype = Genotype(
        root="body",
        nodes={
            "body": NodeGene(
                name="body",
                size=(0.2, 0.2, 0.2),
                joint_type="free",
                children=[connection],
            ),
            "limb": NodeGene(
                name="limb",
                size=(0.1, 0.1, 0.1),
                joint_type="hinge",
            ),
        },
    )
    genotype._ensure_connection_neural_parameters(connection)
    connection.neural_b1 = (1.25, *connection.neural_b1[1:])
    return genotype, connection


def test_neural_tensors_are_preserved_when_child_joint_becomes_fixed():
    genotype, connection = _neural_hinge_genotype()
    original_b1 = connection.neural_b1

    genotype.nodes["limb"].joint_type = "fixed"
    genotype._reset_incoming_neural_parameters("limb")

    assert connection.neural_b1 == original_b1

    genotype.nodes["limb"].joint_type = "hinge"
    genotype._reset_incoming_neural_parameters("limb")

    assert connection.neural_b1 == original_b1


def test_fixed_child_neural_tensors_are_excluded_from_mutation_pool():
    genotype, connection = _neural_hinge_genotype()
    genotype.nodes["limb"].joint_type = "fixed"

    paths = genotype._collect_mutable_parameters(allow_topology_mutations=False)

    assert connection.neural_b1
    assert not any(path[0] == "connection_neural" for path in paths)


def test_preselected_neural_mutation_skips_connection_now_targeting_fixed_child():
    genotype, connection = _neural_hinge_genotype()
    genotype.nodes["rigid"] = NodeGene(
        name="rigid",
        size=(0.1, 0.1, 0.1),
        joint_type="fixed",
    )
    original_b1 = connection.neural_b1

    genotype._mutate_connection_destination(connection, random.Random(1))
    description = genotype._mutate_parameter_path(
        ("connection_neural", connection, "neural_b1", 0),
        random.Random(2),
    )

    assert connection.child == "rigid"
    assert connection.neural_b1 == original_b1
    assert "does not currently support neural control" in description


def test_preselected_stale_neural_index_is_skipped():
    genotype, connection = _neural_hinge_genotype()

    description = genotype._mutate_parameter_path(
        ("connection_neural", connection, "neural_b2", 2),
        random.Random(2),
    )

    assert "unchanged; stale neural parameter path" in description


def test_new_connection_randomly_initializes_neural_tensors():
    genotype = Genotype(
        root="body",
        nodes={
            "body": NodeGene(
                name="body",
                size=(0.2, 0.2, 0.2),
                joint_type="free",
            ),
            "limb": NodeGene(
                name="limb",
                size=(0.1, 0.1, 0.1),
                joint_type="hinge",
            ),
        },
    )

    genotype._add_new_connection(
        genotype.nodes["body"],
        genotype.nodes["limb"],
        random.Random(7),
    )
    connection = genotype.nodes["body"].children[0]

    assert len(connection.neural_w1) == 16
    assert len(connection.neural_w1[0]) == 12
    assert len(connection.neural_w2) == 1
    assert any(value != 0.0 for row in connection.neural_w1 for value in row)
    assert any(value != 0.0 for value in connection.neural_b1)
    assert connection.neural_output_axes == ((0.0, 1.0, 0.0),)


def test_fixed_joint_inherits_compatible_tensor_or_randomly_initializes():
    genotype, connection = _neural_hinge_genotype()
    original = (
        connection.neural_w1,
        connection.neural_b1,
        connection.neural_w2,
        connection.neural_b2,
    )

    genotype.nodes["limb"].joint_type = "fixed"
    genotype._reset_incoming_neural_parameters("limb", random.Random(1))
    genotype.nodes["limb"].joint_type = "hinge"
    genotype._reset_incoming_neural_parameters("limb", random.Random(2))
    assert original == (
        connection.neural_w1,
        connection.neural_b1,
        connection.neural_w2,
        connection.neural_b2,
    )

    connection.neural_w1 = ()
    connection.neural_b1 = ()
    connection.neural_w2 = ()
    connection.neural_b2 = ()
    genotype.nodes["limb"].joint_type = "fixed"
    genotype._reset_incoming_neural_parameters("limb", random.Random(3))
    assert connection.neural_w1 == ()
    genotype.nodes["limb"].joint_type = "hinge"
    genotype._reset_incoming_neural_parameters("limb", random.Random(4))
    assert len(connection.neural_w1) == 16
    assert any(value != 0.0 for row in connection.neural_w1 for value in row)


def test_hinge_to_ball_expands_tensor_and_preserves_axis_output():
    genotype, connection = _neural_hinge_genotype()
    connection.axis = (0.0, 0.0, -2.0)
    row = (1.0, 2.0, 3.0, *tuple(float(i) for i in range(4, 13)))
    connection.neural_w1 = tuple(row for _ in range(16))
    connection.neural_w2 = (tuple(float(i) for i in range(16)),)
    connection.neural_b2 = (2.5,)
    old_b1 = connection.neural_b1
    old_output = connection.neural_w2[0]

    genotype.nodes["limb"].joint_type = "ball"
    genotype._reset_incoming_neural_parameters("limb", random.Random(5))

    assert connection.neural_b1 == old_b1
    assert connection.neural_w2[0] == old_output
    assert connection.neural_b2[0] == 2.5
    assert connection.neural_output_axes[0] == (0.0, 0.0, -1.0)
    assert connection.neural_w1[0][0] == 1.0
    assert connection.neural_w1[0][2:5] == (0.0, 0.0, -2.0)
    assert connection.neural_w1[0][5:8] == (0.0, 0.0, -3.0)
    assert connection.neural_w1[0][8:17] == row[3:12]
    assert len(connection.neural_w2) == 3


def test_ball_to_hinge_keeps_seeded_output_and_its_direction():
    genotype, connection = _neural_hinge_genotype()
    genotype.nodes["limb"].joint_type = "ball"
    genotype._initialize_connection_neural_parameters(
        connection,
        "ball",
        random.Random(8),
    )
    connection.neural_output_axes = (
        (1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    connection.neural_w2 = tuple(
        tuple(float(row * 100 + column) for column in range(16))
        for row in range(3)
    )
    connection.neural_b2 = (10.0, 20.0, 30.0)
    old_w1 = connection.neural_w1
    selected = random.Random(9).randrange(3)

    genotype.nodes["limb"].joint_type = "hinge"
    genotype._reset_incoming_neural_parameters("limb", random.Random(9))

    selected_axis = ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))[selected]
    assert connection.axis == selected_axis
    assert connection.neural_output_axes == (selected_axis,)
    assert connection.neural_w2[0][0] == selected * 100
    assert connection.neural_b2 == ((10.0, 20.0, 30.0)[selected],)
    assert connection.neural_w1[0][3:12] == old_w1[0][8:17]


MUTABLE_NODE_FIELDS = {"size", "joint_type", "shape", "orientation"}
MUTABLE_CONNECTION_FIELDS = {
    "parent_face",
    "surface_uv",
    "child_surface_uv",
    "orientation",
    "axis",
    "scale",
    "motor_enabled",
    "motor_gear",
    "ctrlrange",
    "control_amp",
    "control_freq",
    "control_phase",
    "control_phase_depth_scale",
    "control_phase_order_scale",
}


def _phenotype_and_controller_snapshot(genotype):
    builder = PhenotypeBuilder(genotype, max_node=100)
    mjcf = builder.build()
    controllers = tuple(
        (controller.amp, controller.freq, controller.phase)
        for controller in builder.actuator_controllers
    )
    return mjcf, controllers


def _mutable_effect_genotype(field_name):
    target = ConnectionGene(
        child="limb",
        axis=(0.3, 0.4, 0.5),
        parent_face="+x",
        surface_uv=(0.2, -0.3),
        child_surface_uv=(0.1, -0.2),
        orientation=(0.0, 0.0, 10.0),
        motor_gear=80.0,
        ctrlrange=(-0.8, 1.2),
        control_amp=0.2,
        control_freq=6.0,
        control_phase=0.4,
    )
    body_children = [target]
    nodes = {
        "body": NodeGene(
            name="body",
            size=(0.3, 0.2, 0.1),
            joint_type="free",
            orientation=(0.0, 0.0, 10.0),
            children=body_children,
        ),
        "limb": NodeGene(
            name="limb",
            size=(0.12, 0.08, 0.04),
            joint_type="hinge",
        ),
    }

    if field_name == "recursive_limit":
        target = ConnectionGene(child="segment", axis=(0.3, 0.4, 0.5))
        recursive = ConnectionGene(child="segment", axis=(0.2, 0.5, 0.4))
        nodes["body"].children = [target]
        nodes["segment"] = NodeGene(
            name="segment",
            size=(0.12, 0.08, 0.04),
            recursive_limit=1,
            children=[recursive],
        )
    elif field_name in {"scale", "control_phase_depth_scale"}:
        target = ConnectionGene(
            child="segment",
            axis=(0.3, 0.4, 0.5),
            scale=0.8,
            control_phase_depth_scale=0.3,
        )
        nodes["body"].children = [
            ConnectionGene(child="segment", axis=(0.1, 0.6, 0.2))
        ]
        nodes["segment"] = NodeGene(
            name="segment",
            size=(0.12, 0.08, 0.04),
            recursive_limit=3,
            children=[target],
        )
    elif field_name == "terminal_only":
        recursive = ConnectionGene(child="segment", axis=(0.2, 0.5, 0.4))
        target = ConnectionGene(child="limb", axis=(0.3, 0.4, 0.5))
        nodes["body"].children = [ConnectionGene(child="segment", axis=(0, 1, 0))]
        nodes["segment"] = NodeGene(
            name="segment",
            size=(0.12, 0.08, 0.04),
            recursive_limit=2,
            children=[recursive, target],
        )
    elif field_name == "control_phase_order_scale":
        anchor = ConnectionGene(child="anchor", axis=(0, 1, 0))
        target.control_phase_order_scale = 0.3
        nodes["body"].children = [anchor, target]
        nodes["anchor"] = NodeGene(name="anchor", size=(0.05, 0.05, 0.05))

    return Genotype(root="body", nodes=nodes), target


def test_non_topology_mutations_only_register_body_count_stable_fields():
    genotype, _ = _mutable_effect_genotype("axis")
    paths = genotype._collect_mutable_parameters(allow_topology_mutations=False)
    node_fields = {path[2] for path in paths if path[0] == "node"}
    connection_fields = {path[2] for path in paths if path[0] == "connection"}

    assert ("genotype", "global_control_freq") in paths
    assert node_fields == MUTABLE_NODE_FIELDS
    assert connection_fields == MUTABLE_CONNECTION_FIELDS


MUTABLE_EFFECT_CASES = [
    pytest.param("node", "size", index, id=f"node-size-{index}")
    for index in range(3)
] + [
    pytest.param("node", "joint_type", None, id="node-joint-type"),
    pytest.param("node", "shape", None, id="node-shape"),
    pytest.param("node", "orientation", 0, id="node-orientation-roll"),
    pytest.param("node", "orientation", 1, id="node-orientation-pitch"),
    pytest.param("node", "orientation", 2, id="node-orientation-yaw"),
    pytest.param("node", "recursive_limit", None, id="node-recursive-limit"),
] + [
    pytest.param("connection", "parent_face", None, id="connection-parent-face"),
    pytest.param("connection", "surface_uv", 0, id="connection-surface-u"),
    pytest.param("connection", "surface_uv", 1, id="connection-surface-v"),
    pytest.param("connection", "child_surface_uv", 0, id="connection-child-surface-u"),
    pytest.param("connection", "child_surface_uv", 1, id="connection-child-surface-v"),
    pytest.param("connection", "orientation", 0, id="connection-orientation-roll"),
    pytest.param("connection", "orientation", 1, id="connection-orientation-pitch"),
    pytest.param("connection", "orientation", 2, id="connection-orientation-yaw"),
    pytest.param("connection", "symmetry", None, id="connection-symmetry"),
] + [
    pytest.param("connection", "axis", index, id=f"connection-axis-{index}")
    for index in range(3)
] + [
    pytest.param("connection", "scale", None, id="connection-scale"),
    pytest.param("connection", "terminal_only", None, id="connection-terminal-only"),
    pytest.param("connection", "motor_enabled", None, id="connection-motor-enabled"),
    pytest.param("connection", "motor_gear", None, id="connection-motor-gear"),
    pytest.param("connection", "ctrlrange", 0, id="connection-ctrlrange-min"),
    pytest.param("connection", "ctrlrange", 1, id="connection-ctrlrange-max"),
    pytest.param("connection", "control_amp", None, id="connection-control-amp"),
    pytest.param("connection", "control_freq", None, id="connection-control-freq"),
    pytest.param("connection", "control_phase", None, id="connection-control-phase"),
    pytest.param(
        "connection",
        "control_phase_depth_scale",
        None,
        id="connection-control-phase-depth-scale",
    ),
    pytest.param(
        "connection",
        "control_phase_order_scale",
        None,
        id="connection-control-phase-order-scale",
    ),
]


@pytest.mark.parametrize("owner_type,field_name,index", MUTABLE_EFFECT_CASES)
def test_every_mutable_gene_field_can_change_phenotype_or_controller(
    owner_type,
    field_name,
    index,
):
    genotype, connection = _mutable_effect_genotype(field_name)
    before = _phenotype_and_controller_snapshot(genotype)

    if owner_type == "node":
        if field_name == "recursive_limit":
            node_name = "segment"
        elif field_name == "orientation":
            node_name = "body"
        elif field_name == "shape":
            node_name = "limb"
        else:
            node_name = "limb"
        path = ("node", node_name, field_name)
    else:
        path = ("connection", connection, field_name)
    if index is not None:
        path = (*path, index)

    genotype._mutate_parameter_path(path, random.Random(7))
    after = _phenotype_and_controller_snapshot(genotype)

    assert after != before


def test_global_control_frequency_mutates_continuously():
    genotype, _ = _mutable_effect_genotype("axis")
    genotype.global_control_freq = 2.0

    description = genotype._mutate_parameter_path(
        ("genotype", "global_control_freq"),
        random.Random(7),
    )

    assert "genotype.global_control_freq" in description
    assert genotype.global_control_freq > 0.0
    assert genotype.global_control_freq != pytest.approx(2.0)
    assert genotype.global_control_freq not in genotype.HARMONIC_CONTROL_FREQS


def test_connection_control_frequency_mutates_to_harmonic_multiplier():
    genotype, connection = _mutable_effect_genotype("control_freq")
    connection.control_freq = 2.0

    genotype._mutate_parameter_path(
        ("connection", connection, "control_freq"),
        random.Random(7),
    )

    assert connection.control_freq in genotype.HARMONIC_CONTROL_FREQS
    assert connection.control_freq != 2.0


def test_legacy_connection_control_frequency_snaps_to_nearest_harmonic():
    genotype, connection = _mutable_effect_genotype("control_freq")
    connection.control_freq = 2.6

    genotype._mutate_parameter_path(
        ("connection", connection, "control_freq"),
        random.Random(7),
    )

    assert connection.control_freq == pytest.approx(3.0)

def test_connection_orientation_mutation_preserves_child_normal_constraint():
    genotype, connection = _mutable_effect_genotype("orientation")

    genotype._mutate_parameter_path(
        ("connection", connection, "orientation", 2),
        random.Random(7),
    )
    assert child_orientation_is_valid(connection.parent_face, connection.orientation)

    genotype._mutate_parameter_path(
        ("connection", connection, "parent_face"),
        random.Random(1),
    )
    assert child_orientation_is_valid(connection.parent_face, connection.orientation)


def test_topology_mutations_register_body_count_fields():
    genotype, _ = _mutable_effect_genotype("axis")
    paths = genotype._collect_mutable_parameters(allow_topology_mutations=True)

    assert any(
        path[:3] == ("node", "limb", "recursive_limit") for path in paths
    )
    connection_fields = {
        path[2] for path in paths if path[0] == "connection"
    }
    assert {"symmetry", "terminal_only"} <= connection_fields


def test_non_topology_mutations_preserve_phenotype_body_count():
    genotype = Genotype(
        root="body",
        nodes={
            "body": NodeGene(
                name="body",
                size=(0.3, 0.2, 0.1),
                joint_type="free",
                children=[ConnectionGene(child="segment", axis=(0, 1, 0))],
            ),
            "segment": NodeGene(
                name="segment",
                size=(0.12, 0.08, 0.04),
                recursive_limit=3,
                children=[
                    ConnectionGene(
                        child="segment", axis=(0, 1, 0), symmetry=("xy",)
                    ),
                    ConnectionGene(
                        child="limb", axis=(0, 1, 0), terminal_only=True
                    ),
                ],
            ),
            "limb": NodeGene(
                name="limb",
                size=(0.08, 0.04, 0.04),
            ),
        },
    )
    before_builder = PhenotypeBuilder(genotype, max_node=100)
    before_builder.build()

    genotype.mutation(
        num_mutations=10_000,
        rng=random.Random(7),
        allow_topology_mutations=False,
    )

    after_builder = PhenotypeBuilder(genotype, max_node=100)
    after_builder.build()
    assert after_builder.body_counter == before_builder.body_counter


def test_disallow_root_mutation_excludes_root_node_fields():
    genotype = Genotype(
        root="body",
        nodes={
            "body": NodeGene(
                name="body",
                size=(0.3, 0.2, 0.1),
                joint_type="free",
                orientation=(0.0, 0.0, 0.0),
                children=[ConnectionGene(child="limb", axis=(0, 1, 0))],
            ),
            "limb": NodeGene(
                name="limb",
                size=(0.1, 0.1, 0.1),
                joint_type="hinge",
            ),
        },
    )

    paths = genotype._collect_mutable_parameters(
        allow_topology_mutations=False,
        allow_root_mutation=False,
    )

    assert not any(path[:2] == ("node", "body") for path in paths)
    assert ("node", "limb", "joint_type") in paths
    assert any(path[0] == "connection" for path in paths)


@pytest.mark.parametrize(
    ("delta", "expected"),
    [(10.0, 1.0), (-10.0, 0.001)],
)
def test_size_mutation_clamps_result_not_delta(monkeypatch, delta, expected):
    genotype = Genotype(
        root="body",
        nodes={
            "body": NodeGene(
                name="body",
                size=(0.25, 0.15, 0.1),
                joint_type="free",
            )
        },
    )
    monkeypatch.setattr(
        genotype,
        "_normal_mutation_delta",
        lambda *_args, **_kwargs: delta,
    )

    genotype._mutate_parameter_path(
        ("node", "body", "size", 0),
        random.Random(1),
    )

    assert genotype.nodes["body"].size[0] == expected


@pytest.mark.parametrize(
    ("parent_size", "expected"),
    [((0.0001, 0.0001, 0.0001), 0.001), ((10.0, 10.0, 10.0), 1.0)],
)
def test_fresh_child_size_clamps_after_parent_relative_generation(
    parent_size,
    expected,
):
    genotype = Genotype(
        root="body",
        nodes={
            "body": NodeGene(
                name="body",
                size=parent_size,
                joint_type="free",
            )
        },
    )

    child_size = genotype._random_child_size_from_parent(
        genotype.nodes["body"],
        random.Random(1),
    )

    assert child_size == (expected, expected, expected)
