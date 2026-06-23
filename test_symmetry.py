import random
import xml.etree.ElementTree as ET

import pytest

from evol_virtual_creature.genes import ConnectionGene, NodeGene
from evol_virtual_creature.genotype import Genotype
from evol_virtual_creature.genotype_io import (
    load_genotype_from_json,
    save_genotype_to_json,
)
from evol_virtual_creature.graph_analysis import (
    GenotypeGraphAnalyzer,
    PhenotypeNodeLimitExceeded,
)
from evol_virtual_creature.phenotype import PhenotypeBuilder


def _vector(element, attribute):
    return tuple(float(value) for value in element.get(attribute).split())


def test_connection_validates_and_canonicalizes_symmetry_planes():
    connection = ConnectionGene(
        child="child",
        axis=(0, 1, 0),
        symmetry=("yz", "xy"),
    )
    assert connection.symmetry == ("xy", "yz")

    with pytest.raises(ValueError, match="Unknown symmetry plane"):
        ConnectionGene(child="child", axis=(0, 1, 0), symmetry=("xx",))

    with pytest.raises(ValueError, match="must not contain duplicates"):
        ConnectionGene(
            child="child",
            axis=(0, 1, 0),
            symmetry=("xy", "xy"),
        )


def test_symmetry_round_trips_through_json(tmp_path):
    genotype = Genotype(
        root="body",
        nodes={
            "body": NodeGene(
                name="body",
                size=(1.0, 1.0, 1.0),
                joint_type="free",
                children=[ConnectionGene(
                    child="limb",
                    axis=(0, 1, 0),
                    symmetry=("xy", "yz"),
                )],
            ),
            "limb": NodeGene(name="limb", size=(0.2, 0.1, 0.1)),
        },
    )
    path = tmp_path / "symmetric.json"

    save_genotype_to_json(genotype, path)
    loaded = load_genotype_from_json(path)

    assert loaded.nodes["body"].children[0].symmetry == ("xy", "yz")


def test_symmetry_mutation_toggles_one_plane():
    connection = ConnectionGene(child="limb", axis=(0, 1, 0))
    genotype = Genotype(
        root="body",
        nodes={
            "body": NodeGene(
                name="body",
                size=(1.0, 1.0, 1.0),
                joint_type="free",
                children=[connection],
            ),
            "limb": NodeGene(name="limb", size=(0.2, 0.1, 0.1)),
        },
    )

    mutable_parameters = genotype._collect_mutable_parameters(
        allow_topology_mutations=False
    )
    symmetry_path = ("connection", connection, "symmetry")
    assert symmetry_path in mutable_parameters

    description = genotype._mutate_parameter_path(
        symmetry_path,
        random.Random(4),
    )
    assert ".symmetry:" in description
    assert len(connection.symmetry) == 1
    assert connection.symmetry[0] in {"xy", "xz", "yz"}


def test_single_plane_mirrors_subtree_axes_and_actuator_controls():
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
                    symmetry=("yz",),
                    control_phase=0.25,
                    control_phase_order_scale=1.5,
                )],
            ),
            "segment": NodeGene(
                name="segment",
                size=(0.4, 0.3, 0.2),
                children=[ConnectionGene(
                    child="tip",
                    axis=(0, 0, 1),
                    parent_face="+x",
                    control_phase=0.75,
                    control_phase_order_scale=2.0,
                )],
            ),
            "tip": NodeGene(name="tip", size=(0.1, 0.1, 0.1)),
        },
    )

    builder = PhenotypeBuilder(genotype, max_node=10)
    xml_root = ET.fromstring(builder.build())
    root_body = xml_root.find("./worldbody/body")
    segment_bodies = root_body.findall("body")

    assert len(segment_bodies) == 2
    assert [_vector(body, "pos") for body in segment_bodies] == [
        pytest.approx((2.0, 0.0, 0.0)),
        pytest.approx((-2.0, 0.0, 0.0)),
    ]
    assert [_vector(body.find("geom"), "pos") for body in segment_bodies] == [
        pytest.approx((0.4, 0.0, 0.0)),
        pytest.approx((-0.4, 0.0, 0.0)),
    ]
    assert [_vector(body.find("joint"), "axis") for body in segment_bodies] == [
        pytest.approx((0.0, 1.0, 0.0)),
        pytest.approx((0.0, -1.0, 0.0)),
    ]

    tip_bodies = [body.find("body") for body in segment_bodies]
    assert [_vector(body, "pos") for body in tip_bodies] == [
        pytest.approx((0.8, 0.0, 0.0)),
        pytest.approx((-0.8, 0.0, 0.0)),
    ]
    assert [_vector(body.find("joint"), "axis") for body in tip_bodies] == [
        pytest.approx((0.0, 0.0, 1.0)),
        pytest.approx((0.0, 0.0, -1.0)),
    ]

    controllers = builder.actuator_controllers
    assert len(controllers) == 4
    assert controllers[0].phase == pytest.approx(controllers[2].phase)
    assert controllers[1].phase == pytest.approx(controllers[3].phase)
    assert controllers[0].amp == pytest.approx(controllers[2].amp)
    assert controllers[1].freq == pytest.approx(controllers[3].freq)


def test_three_symmetry_planes_create_eight_children_and_count_them():
    genotype = Genotype(
        root="body",
        nodes={
            "body": NodeGene(
                name="body",
                size=(2.0, 1.0, 0.5),
                joint_type="free",
                children=[ConnectionGene(
                    child="limb",
                    axis=(1, 1, 1),
                    parent_face="+x",
                    surface_uv=(0.25, 0.5),
                    symmetry=("xy", "xz", "yz"),
                )],
            ),
            "limb": NodeGene(name="limb", size=(0.2, 0.1, 0.05)),
        },
    )

    assert GenotypeGraphAnalyzer(genotype, max_node=9).validate() == 9
    with pytest.raises(PhenotypeNodeLimitExceeded):
        GenotypeGraphAnalyzer(genotype, max_node=8).validate()

    xml_root = ET.fromstring(PhenotypeBuilder(genotype, max_node=9).build())
    root_body = xml_root.find("./worldbody/body")
    child_positions = {_vector(body, "pos") for body in root_body.findall("body")}

    assert len(child_positions) == 8
    assert child_positions == {
        (x, y, z)
        for x in (-2.0, 2.0)
        for y in (-0.25, 0.25)
        for z in (-0.25, 0.25)
    }
