from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence
from dataclasses import asdict
import json
import warnings

from .control import BALL_NEURAL_AXES, zero_neural_parameters
from .genotype import ConnectionGene, Genotype, NodeGene
from .genes import (
    CONTROL_MODES,
    MAX_BODY_SIZE,
    MIN_BODY_SIZE,
    orientation_for_parent_face,
)


GenotypeSpec = Mapping[str, Mapping[str, Any]]

# NodeGene.joint_axis was removed; accept and discard it when migrating old JSON.
LEGACY_NODE_FIELDS = {"joint_axis"}
LEGACY_CONNECTION_FIELDS = {"control_mode"}


def _referenced_child_node_names(spec: GenotypeSpec) -> set[str]:
    return {
        connection_spec["child"]
        for node_spec in spec.values()
        for connection_spec in node_spec.get("children", [])
    }


def build_genotype(
    root: str,
    spec: GenotypeSpec,
    global_control_freq: float = 1.0,
    control_mode: str = "neural",
) -> Genotype:
    """
    Build a concrete genotype from a compact declarative recipe.

    Each spec entry describes one reusable node type. The optional "children"
    list contains connection dictionaries that are passed to ConnectionGene.
    """
    nodes: Dict[str, NodeGene] = {}
    child_node_names = _referenced_child_node_names(spec)

    for node_name, node_spec in spec.items():
        node_kwargs = {
            key: value
            for key, value in node_spec.items()
            if key != "children" and key not in LEGACY_NODE_FIELDS
        }
        if (
            node_name != root
            and node_name in child_node_names
            and "joint_type" not in node_kwargs
        ):
            node_kwargs["joint_type"] = "fixed"
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
            connection_data = _normalize_connection_data(
                connection_spec,
                nodes[node_name].size,
            )
            nodes[node_name].children.append(
                ConnectionGene(
                    **_with_neural_defaults(
                        connection_data,
                        nodes[child_name].joint_type,
                        control_mode,
                    )
                )
            )

    return Genotype(
        root=root,
        nodes=nodes,
        global_control_freq=global_control_freq,
        control_mode=control_mode,
    )


def load_genotype_from_json(path: str | Path) -> Genotype:
    """Load a recursive genotype recipe from a JSON file."""
    with Path(path).open() as f:
        genotype_data = json.load(f)

    _warn_for_out_of_bounds_sizes(genotype_data.get("nodes", {}).items())
    _warn_for_out_of_bounds_sizes(
        (
            (node.get("name", f"archived node #{index}"), node)
            for index, node in enumerate(genotype_data.get("archived_nodes", []))
        )
    )
    control_mode = _control_mode_from_json(genotype_data)
    genotype = build_genotype(
        root=genotype_data["root"],
        spec=genotype_data["nodes"],
        global_control_freq=genotype_data.get("global_control_freq", 1.0),
        control_mode=control_mode,
    )
    genotype.archived_connections = [
        ConnectionGene(
            **_with_neural_defaults(
                _normalize_connection_data(connection),
                genotype.nodes[connection["child"]].joint_type
                if connection.get("child") in genotype.nodes
                else "hinge",
                control_mode,
            )
        )
        for connection in genotype_data.get("archived_connections", [])
    ]
    genotype.archived_nodes = [
        _node_from_dict(node, control_mode)
        for node in genotype_data.get("archived_nodes", [])
    ]
    return genotype


def _warn_for_out_of_bounds_sizes(
    named_node_specs: Iterable[tuple[str, Mapping[str, Any]]],
) -> None:
    for node_name, node_spec in named_node_specs:
        size = node_spec.get("size", ())
        out_of_bounds = [
            float(value)
            for value in size
            if float(value) < MIN_BODY_SIZE or float(value) > MAX_BODY_SIZE
        ]
        if not out_of_bounds:
            continue
        warnings.warn(
            f"Body part {node_name!r} has size component(s) {out_of_bounds} "
            f"outside the recommended range [{MIN_BODY_SIZE}, {MAX_BODY_SIZE}]; "
            "values are being loaded unchanged.",
            UserWarning,
            stacklevel=2,
        )


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
        "global_control_freq": genotype.global_control_freq,
        "control_mode": genotype.control_mode,
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


def _control_mode_from_json(genotype_data: Mapping[str, Any]) -> str:
    if "control_mode" in genotype_data:
        return _validated_control_mode(genotype_data["control_mode"])

    legacy_modes = set(_legacy_connection_control_modes(genotype_data))
    if not legacy_modes:
        return "neural"
    if len(legacy_modes) > 1:
        modes = ", ".join(sorted(legacy_modes))
        raise ValueError(
            "legacy genotype has mixed per-connection control modes "
            f"({modes}); add a top-level control_mode"
        )
    return _validated_control_mode(next(iter(legacy_modes)))


def _validated_control_mode(control_mode: Any) -> str:
    if control_mode not in CONTROL_MODES:
        valid_modes = ", ".join(CONTROL_MODES)
        raise ValueError(
            f"Unknown control mode {control_mode!r}; expected one of {valid_modes}"
        )
    return str(control_mode)


def _legacy_connection_control_modes(genotype_data: Mapping[str, Any]) -> Iterable[str]:
    for connection in _connection_specs(genotype_data.get("nodes", {}).values()):
        if "control_mode" in connection:
            yield connection["control_mode"]
    for connection in genotype_data.get("archived_connections", []):
        if "control_mode" in connection:
            yield connection["control_mode"]
    for node in genotype_data.get("archived_nodes", []):
        for connection in node.get("children", []):
            if "control_mode" in connection:
                yield connection["control_mode"]


def _connection_specs(node_specs: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    for node_spec in node_specs:
        yield from node_spec.get("children", [])


def _with_neural_defaults(
    connection_data: Mapping[str, Any],
    joint_type: str,
    control_mode: str,
) -> Dict[str, Any]:
    normalized = {
        key: value
        for key, value in connection_data.items()
        if key not in LEGACY_CONNECTION_FIELDS
    }
    if control_mode != "neural":
        return normalized
    _with_neural_axis_defaults(normalized, joint_type)
    if joint_type == "fixed":
        return normalized
    if all(
        field_name in normalized
        for field_name in ("neural_w1", "neural_b1", "neural_w2", "neural_b2")
    ):
        return normalized
    neural_w1, neural_b1, neural_w2, neural_b2 = zero_neural_parameters(joint_type)
    normalized.setdefault("neural_w1", neural_w1)
    normalized.setdefault("neural_b1", neural_b1)
    normalized.setdefault("neural_w2", neural_w2)
    normalized.setdefault("neural_b2", neural_b2)
    return normalized


def _with_neural_axis_defaults(
    connection_data: Dict[str, Any],
    joint_type: str,
) -> None:
    if "neural_output_axes" in connection_data:
        return
    output_count = len(connection_data.get("neural_b2", ()))
    if output_count == 3 or joint_type == "ball":
        connection_data["neural_output_axes"] = BALL_NEURAL_AXES
    elif output_count == 1 or joint_type in {"hinge", "slide"}:
        connection_data["neural_output_axes"] = (
            tuple(connection_data.get("axis", (0.0, 1.0, 0.0))),
        )


def _node_to_dict(node: NodeGene, include_name: bool = False) -> Dict[str, Any]:
    node_data = asdict(node)
    if not include_name:
        node_data.pop("name")
    return node_data


def _node_from_dict(node_data: Mapping[str, Any], control_mode: str) -> NodeGene:
    node_kwargs = {
        key: value
        for key, value in node_data.items()
        if key != "children" and key not in LEGACY_NODE_FIELDS
    }
    node = NodeGene(**node_kwargs)
    node.children = [
        ConnectionGene(
            **_with_neural_defaults(
                _normalize_connection_data(connection, node.size),
                "hinge",
                control_mode,
            )
        )
        for connection in node_data.get("children", [])
    ]
    return node


def _normalize_connection_data(
    connection_data: Mapping[str, Any],
    parent_size: Sequence[float] | None = None,
) -> Dict[str, Any]:
    normalized = dict(connection_data)
    legacy_pos = normalized.pop("pos", None)
    normalized.setdefault("child_surface_uv", (0.0, 0.0))
    if "parent_face" in normalized or "surface_uv" in normalized:
        normalized.setdefault("parent_face", "+x")
        normalized.setdefault("surface_uv", (0.0, 0.0))
        normalized.setdefault(
            "orientation",
            orientation_for_parent_face(normalized["parent_face"]),
        )
        return normalized

    if legacy_pos is None:
        normalized.setdefault("parent_face", "+x")
        normalized.setdefault("surface_uv", (0.0, 0.0))
        normalized.setdefault("orientation", (0.0, 0.0, 0.0))
        return normalized

    if parent_size is None:
        normalized["parent_face"] = _legacy_pos_to_face(legacy_pos)
        normalized["surface_uv"] = (0.0, 0.0)
        normalized.setdefault(
            "orientation",
            orientation_for_parent_face(normalized["parent_face"]),
        )
        return normalized

    parent_face, surface_uv = _legacy_pos_to_surface(legacy_pos, parent_size)
    normalized.setdefault("parent_face", parent_face)
    normalized.setdefault("surface_uv", surface_uv)
    normalized.setdefault(
        "orientation",
        orientation_for_parent_face(normalized["parent_face"]),
    )
    return normalized


def _legacy_pos_to_face(pos: Sequence[float]) -> str:
    if len(pos) != 3:
        raise ValueError("Legacy connection pos must contain exactly three coordinates")

    normal_axis = max(range(3), key=lambda axis: abs(float(pos[axis])))
    sign = "+" if float(pos[normal_axis]) >= 0.0 else "-"
    return f"{sign}{'xyz'[normal_axis]}"


def _legacy_pos_to_surface(
    pos: Sequence[float],
    parent_size: Sequence[float],
) -> tuple[str, tuple[float, float]]:
    if len(pos) != 3:
        raise ValueError("Legacy connection pos must contain exactly three coordinates")
    if len(parent_size) != 3:
        raise ValueError("Parent size must contain exactly three coordinates")

    normalized_distances = [
        abs(float(position)) / max(float(half_size), 1e-12)
        for position, half_size in zip(pos, parent_size)
    ]
    normal_axis = max(range(3), key=normalized_distances.__getitem__)
    sign = "+" if float(pos[normal_axis]) >= 0.0 else "-"
    parent_face = f"{sign}{'xyz'[normal_axis]}"
    tangent_axes = [axis for axis in range(3) if axis != normal_axis]
    surface_uv = tuple(
        max(
            -1.0,
            min(
                1.0,
                float(pos[axis]) / max(float(parent_size[axis]), 1e-12),
            ),
        )
        for axis in tangent_axes
    )
    return parent_face, surface_uv
