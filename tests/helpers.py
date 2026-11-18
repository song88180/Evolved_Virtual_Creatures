"""Shared test helpers.

Genotype trees written inline (``tree_node`` / ``tree_genotype``), private task
definitions (``make_definition``), fake evaluated creatures (``make_creature``),
and small MuJoCo model helpers.

Writing genotype trees inline:

``NodeGene`` only stores connection names (``child_connections``). Tests often
want to declare connection objects next to their parent node instead, so
``tree_node`` pairs a node gene with its outgoing ``ConnectionGene`` objects and
``tree_genotype`` names those connections and registers them on the genotype.
``make_definition`` returns a task definition for ``PhenotypeBuilder`` with only
the overridden fields changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Mapping, Union

import mujoco

from evc3d import evolution
from evc3d.evolution_tasks import task_definition
from evc3d.evolution_tasks.config import EnvironmentFamily
from evc3d.evolution_tasks.shared import TaskDefinition
from evc3d.genes import ConnectionGene, NodeGene
from evc3d.genotype import Genotype
from evc3d.genotype_io import load_genotype_from_json
from evc3d.phenotype import PhenotypeBuilder


@dataclass
class TreeNode:
    gene: NodeGene
    children: List[ConnectionGene] = field(default_factory=list)


NodeLike = Union[NodeGene, TreeNode]


def tree_node(*args, children: Iterable[ConnectionGene] = (), **kwargs) -> TreeNode:
    """Build a ``NodeGene`` together with the connection genes leaving it."""
    return TreeNode(NodeGene(*args, **kwargs), list(children))


def _attach_children(
    node: NodeLike,
    target: Dict[str, ConnectionGene],
) -> NodeGene:
    if isinstance(node, NodeGene):
        return node
    for connection in node.children:
        base = connection.name or "connection"
        name = base
        index = 1
        while name in target and target[name] is not connection:
            index += 1
            name = f"{base}_{index}"
        connection.name = name
        target[name] = connection
        node.gene.child_connections.append(name)
    return node.gene


def tree_genotype(
    root: str,
    nodes: Mapping[str, NodeLike],
    *,
    archived_nodes: Iterable[NodeLike] = (),
    **kwargs,
) -> Genotype:
    """Build a ``Genotype`` from nodes that may carry inline connection genes."""
    connections: Dict[str, ConnectionGene] = dict(kwargs.pop("connections", {}))
    archived_connections: Dict[str, ConnectionGene] = dict(
        kwargs.pop("archived_connections", {})
    )
    node_genes = {
        name: _attach_children(node, connections) for name, node in nodes.items()
    }
    if isinstance(archived_nodes, Mapping):
        archived_nodes = archived_nodes.values()
    archived_genes = [
        _attach_children(node, archived_connections) for node in archived_nodes
    ]
    return Genotype(
        root=root,
        nodes=node_genes,
        connections=connections,
        archived_nodes=archived_genes,
        archived_connections=archived_connections,
        **kwargs,
    )


def make_definition(
    task: str = "swimming_x",
    *,
    environment: EnvironmentFamily | None = None,
    **config_overrides,
) -> TaskDefinition:
    """Return a private copy of a registered task with config/environment overrides.

    ``PhenotypeBuilder`` takes a task definition; geometry tests use this to set
    just the fields they care about (usually ``max_body_count``). The registry's
    shared definition object is never modified.
    """
    base = task_definition(task)
    return replace(
        base,
        config=base.config_type(**config_overrides),
        environment=base.environment if environment is None else environment,
    )


@dataclass(frozen=True)
class FakeEvaluationResult:
    """Minimal stand-in for a task result with the fields evolution reads."""

    fitness: float
    body_count: int = 0
    total_volume: float = 0.0
    control_energy: float = 0.0
    build_failed: bool = False
    disqualified: bool = False
    failure_reason: str | None = None


GENOTYPE_PATH = "examples/example_genotype.json"


def make_creature(genotype, fitness, changed_by_mutation=False, **result_fields):
    return evolution.EvaluatedCreature(
        genotype,
        FakeEvaluationResult(fitness=fitness, **result_fields),
        changed_by_mutation=changed_by_mutation,
    )


def creature_geom_ids(model):
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    return [geom_id for geom_id in range(model.ngeom) if geom_id != floor_id]


def placed_walker(definition):
    genotype = load_genotype_from_json(GENOTYPE_PATH)
    model = mujoco.MjModel.from_xml_string(
        PhenotypeBuilder(genotype, definition).build()
    )
    return model, mujoco.MjData(model)
