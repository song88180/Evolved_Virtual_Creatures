from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping
import warnings

from .control import BALL_NEURAL_AXES, zero_neural_parameters
from .genes import (
    CONTROL_MODES,
    MAX_BODY_SIZE,
    MIN_BODY_SIZE,
    ConnectionGene,
    NodeGene,
    orientation_for_parent_face,
)
from .genotype import Genotype


GeneSpec = Mapping[str, Mapping[str, Any]]


def build_genotype(
    root: str,
    nodes: GeneSpec | None = None,
    connections: GeneSpec | None = None,
    global_control_freq: float = 1.0,
    control_mode: str = "neural",
    *,
    spec: GeneSpec | None = None,
) -> Genotype:
    """Build a genotype from normalized node and connection specifications."""
    if (
        spec is None
        and nodes is not None
        and connections is None
        and any("children" in node_spec for node_spec in nodes.values())
    ):
        spec = nodes
        nodes = None
    if spec is not None:
        if nodes is not None:
            raise ValueError("Use nodes/connections or spec, not both")
        normalized_nodes: Dict[str, Dict[str, Any]] = {}
        normalized_connections: Dict[str, Dict[str, Any]] = {}
        referenced_children = {
            child["child"]
            for node_spec in spec.values()
            for child in node_spec.get("children", [])
        }
        for node_name, node_spec in spec.items():
            node_data = {key: value for key, value in node_spec.items() if key != "children"}
            child_names = []
            for index, connection in enumerate(node_spec.get("children", []), 1):
                base = f"{node_name}_connection"
                name = base if index == 1 else f"{base}_{index}"
                normalized_connections[name] = dict(connection)
                child_names.append(name)
            if child_names:
                node_data["child_connections"] = child_names
            if node_name != root and node_name in referenced_children:
                node_data.setdefault("joint_type", "fixed")
            normalized_nodes[node_name] = node_data
        nodes = normalized_nodes
        connections = normalized_connections
    if nodes is None:
        raise TypeError("build_genotype requires nodes and connections")
    connections = connections or {}
    node_genes = {
        name: NodeGene(name=name, **dict(spec))
        for name, spec in nodes.items()
    }
    if root not in node_genes:
        raise KeyError(f"Unknown genotype root: {root}")

    connection_genes: Dict[str, ConnectionGene] = {}
    for name, spec in connections.items():
        child = spec.get("child")
        if child not in node_genes:
            raise KeyError(f"Connection {name!r} targets unknown node {child!r}")
        data = _with_neural_defaults(
            dict(spec), node_genes[str(child)].joint_type, control_mode
        )
        connection_genes[name] = ConnectionGene(name=name, **data)

    return Genotype(
        root=root,
        nodes=node_genes,
        connections=connection_genes,
        global_control_freq=global_control_freq,
        control_mode=control_mode,
    )


def load_genotype_from_json(path: str | Path) -> Genotype:
    """Load the normalized genotype JSON schema."""
    with Path(path).open() as file:
        data = json.load(file)

    if any("children" in spec for spec in data.get("nodes", {}).values()):
        raise ValueError(
            "Embedded 'children' is unsupported; use node 'child_connections' "
            "and the top-level 'connections' map"
        )
    if "connections" not in data:
        raise ValueError("Genotype JSON must contain a top-level 'connections' map")
    if isinstance(data.get("archived_nodes", {}), list) or isinstance(
        data.get("archived_connections", {}), list
    ):
        raise ValueError("Archived genes must be serialized as name-keyed maps")

    _warn_for_out_of_bounds_sizes(data.get("nodes", {}).items())
    _warn_for_out_of_bounds_sizes(data.get("archived_nodes", {}).items())
    control_mode = _validated_control_mode(data.get("control_mode", "neural"))

    nodes = {
        name: NodeGene(name=name, **dict(spec))
        for name, spec in data["nodes"].items()
    }
    archived_nodes = {
        name: NodeGene(name=name, **dict(spec))
        for name, spec in data.get("archived_nodes", {}).items()
    }
    all_nodes = {**archived_nodes, **nodes}

    def connection_map(specs: Mapping[str, Mapping[str, Any]]) -> Dict[str, ConnectionGene]:
        result = {}
        for name, spec in specs.items():
            child = spec.get("child")
            joint_type = all_nodes[str(child)].joint_type if child in all_nodes else "hinge"
            result[name] = ConnectionGene(
                name=name,
                **_with_neural_defaults(dict(spec), joint_type, control_mode),
            )
        return result

    genotype = Genotype(
        root=data["root"],
        nodes=nodes,
        connections=connection_map(data["connections"]),
        global_control_freq=data.get("global_control_freq", 1.0),
        control_mode=control_mode,
        archived_nodes=archived_nodes,
        archived_connections=connection_map(data.get("archived_connections", {})),
    )
    return genotype


def save_genotype_to_json(genotype: Genotype, path: str | Path) -> None:
    data = genotype_to_dict(genotype)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def genotype_to_dict(genotype: Genotype) -> Dict[str, Any]:
    genotype.validate_node_names()
    genotype.validate_connection_names()
    genotype.validate_references()
    return {
        "root": genotype.root,
        "global_control_freq": genotype.global_control_freq,
        "control_mode": genotype.control_mode,
        "nodes": {name: _gene_to_dict(node) for name, node in genotype.nodes.items()},
        "connections": {
            name: _gene_to_dict(connection)
            for name, connection in genotype.connections.items()
        },
        "archived_nodes": {
            name: _gene_to_dict(node)
            for name, node in genotype.archived_nodes.items()
        },
        "archived_connections": {
            name: _gene_to_dict(connection)
            for name, connection in genotype.archived_connections.items()
        },
    }


def _gene_to_dict(gene: NodeGene | ConnectionGene) -> Dict[str, Any]:
    data = asdict(gene)
    data.pop("name")
    for field_name in list(data):
        if field_name.startswith("_"):
            data.pop(field_name)
    return data


def _validated_control_mode(control_mode: Any) -> str:
    if control_mode not in CONTROL_MODES:
        valid = ", ".join(CONTROL_MODES)
        raise ValueError(f"Unknown control mode {control_mode!r}; expected one of {valid}")
    return str(control_mode)


def _with_neural_defaults(
    connection_data: Mapping[str, Any],
    joint_type: str,
    control_mode: str,
) -> Dict[str, Any]:
    normalized = dict(connection_data)
    if "orientation" not in normalized:
        normalized["orientation"] = orientation_for_parent_face(
            normalized.get("parent_face", "+x")
        )
    if control_mode != "neural":
        return normalized
    if "neural_output_axes" not in normalized:
        output_count = len(normalized.get("neural_b2", ()))
        if output_count == 3 or joint_type == "ball":
            normalized["neural_output_axes"] = BALL_NEURAL_AXES
        elif output_count == 1 or joint_type in {"hinge", "slide"}:
            normalized["neural_output_axes"] = (
                tuple(normalized.get("axis", (0.0, 1.0, 0.0))),
            )
    if joint_type == "fixed" or all(
        field in normalized for field in ("neural_w1", "neural_b1", "neural_w2", "neural_b2")
    ):
        return normalized
    w1, b1, w2, b2 = zero_neural_parameters(joint_type)
    normalized.setdefault("neural_w1", w1)
    normalized.setdefault("neural_b1", b1)
    normalized.setdefault("neural_w2", w2)
    normalized.setdefault("neural_b2", b2)
    return normalized


def _warn_for_out_of_bounds_sizes(
    named_specs: Iterable[tuple[str, Mapping[str, Any]]],
) -> None:
    for name, spec in named_specs:
        values = [
            float(value)
            for value in spec.get("size", ())
            if float(value) < MIN_BODY_SIZE or float(value) > MAX_BODY_SIZE
        ]
        if values:
            warnings.warn(
                f"Body part {name!r} has size component(s) {values} outside the "
                f"recommended range [{MIN_BODY_SIZE}, {MAX_BODY_SIZE}]; values "
                "are being loaded unchanged.",
                UserWarning,
                stacklevel=2,
            )
