from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping
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

    return build_genotype(
        root=genotype_data["root"],
        spec=genotype_data["nodes"],
    )

