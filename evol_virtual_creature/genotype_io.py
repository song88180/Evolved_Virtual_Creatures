from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping
from dataclasses import asdict
import json

from .genotype import ConnectionGene, Genotype, NodeGene


GenotypeSpec = Mapping[str, Mapping[str, Any]]


def build_genotype(root: str, spec: GenotypeSpec) -> Genotype:
    """
    Build a concrete genotype from a compact declarative recipe.

    Each spec entry describes one reusable node type. The optional "children"
    list contains connection dictionaries that are passed to ConnectionGene.
    """
    nodes: Dict[str, NodeGene] = {}

    for node_name, node_spec in spec.items():
        node_kwargs = {
            key: value
            for key, value in node_spec.items()
            if key != "children"
        }
        nodes[node_name] = NodeGene(name=node_name, **node_kwargs)

    if root not in nodes:
        raise KeyError(f"Unknown genotype root: {root}")

    for node_name, node_spec in spec.items():
        children = node_spec.get("children", [])
        for connection_spec in children:
            child_name = connection_spec["child"]
            if child_name not in nodes:
                raise KeyError(
                    f"Node '{node_name}' connects to unknown child '{child_name}'"
                )
            nodes[node_name].children.append(ConnectionGene(**connection_spec))

    return Genotype(root=root, nodes=nodes)


def load_genotype_from_json(path: str | Path) -> Genotype:
    """Load a recursive genotype recipe from a JSON file."""
    with Path(path).open() as f:
        genotype_data = json.load(f)

    genotype = build_genotype(
        root=genotype_data["root"],
        spec=genotype_data["nodes"],
    )
    genotype.archived_connections = [
        ConnectionGene(**connection)
        for connection in genotype_data.get("archived_connections", [])
    ]
    genotype.archived_nodes = [
        _node_from_dict(node)
        for node in genotype_data.get("archived_nodes", [])
    ]
    return genotype


def save_genotype_to_json(genotype: Genotype, path: str | Path):
    """Save a genotype recipe to JSON, including archived genetic material."""
    genotype_data = genotype_to_dict(genotype)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(genotype_data, f, indent=2)
        f.write("\n")


def genotype_to_dict(genotype: Genotype) -> Dict[str, Any]:
    """Convert a genotype into the JSON recipe shape used by this project."""
    return {
        "root": genotype.root,
        "nodes": {
            node_name: _node_to_dict(node)
            for node_name, node in genotype.nodes.items()
        },
        "archived_connections": [
            asdict(connection)
            for connection in genotype.archived_connections
        ],
        "archived_nodes": [
            _node_to_dict(node, include_name=True)
            for node in genotype.archived_nodes
        ],
    }


def _node_to_dict(node: NodeGene, include_name: bool = False) -> Dict[str, Any]:
    node_data = asdict(node)
    if not include_name:
        node_data.pop("name")
    return node_data


def _node_from_dict(node_data: Mapping[str, Any]) -> NodeGene:
    node_kwargs = {
        key: value
        for key, value in node_data.items()
        if key not in {"children"}
    }
    node = NodeGene(**node_kwargs)
    node.children = [
        ConnectionGene(**connection)
        for connection in node_data.get("children", [])
    ]
    return node
