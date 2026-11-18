"""Test-only helpers for writing genotype trees inline.

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

from evc3d.evolution_tasks import task_definition
from evc3d.evolution_tasks.shared import EnvironmentFamily, TaskDefinition
from evc3d.genes import ConnectionGene, NodeGene
from evc3d.genotype import Genotype


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
        config=replace(base.config, **config_overrides),
        environment=base.environment if environment is None else environment,
    )
