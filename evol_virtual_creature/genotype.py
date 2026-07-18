from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set

from .genes import CONTROL_MODES, ConnectionGene, NodeGene
from .genotype_mutation import GenotypeMutationMixin


@dataclass
class Genotype(GenotypeMutationMixin):
    root: str
    nodes: Dict[str, NodeGene]
    connections: Dict[str, ConnectionGene] = field(default_factory=dict)
    global_control_freq: float = 1.0
    control_mode: str = "neural"
    archived_connections: Dict[str, ConnectionGene] = field(default_factory=dict)
    archived_nodes: Dict[str, NodeGene] = field(default_factory=dict)
    _next_mutation_index: Dict[str, int] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _used_node_names: Set[str] = field(
        default_factory=set,
        init=False,
        repr=False,
    )
    _used_connection_names: Set[str] = field(default_factory=set, init=False, repr=False)
    _next_connection_mutation_index: Dict[str, int] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self):
        if isinstance(self.archived_nodes, list):
            self.archived_nodes = {node.name: node for node in self.archived_nodes}
        if isinstance(self.archived_connections, list):
            self.archived_connections = self._normalize_connection_list(
                self.archived_connections, "archived_connection"
            )
        self._normalize_embedded_connections()
        if (
            not math.isfinite(self.global_control_freq)
            or self.global_control_freq <= 0.0
        ):
            raise ValueError("global_control_freq must be finite and greater than zero")
        if self.control_mode not in CONTROL_MODES:
            valid_modes = ", ".join(CONTROL_MODES)
            raise ValueError(
                f"Unknown control mode {self.control_mode!r}; expected one of "
                f"{valid_modes}"
            )
        self.validate_node_names()
        self.validate_connection_names()
        self.validate_references()
        self.rebuild_node_name_allocator()
        self.rebuild_connection_name_allocator()

    def _normalize_connection_list(
        self, genes: List[ConnectionGene], base: str
    ) -> Dict[str, ConnectionGene]:
        result: Dict[str, ConnectionGene] = {}
        for index, gene in enumerate(genes, 1):
            name = gene.name or (base if index == 1 else f"{base}_{index}")
            while name in result or name in self.connections:
                index += 1
                name = f"{base}_{index}"
            gene.name = name
            result[name] = gene
        return result

    def _normalize_embedded_connections(self) -> None:
        for node in self.nodes.values():
            self._normalize_node_connections(node, self.connections)
        for node in self.archived_nodes.values():
            self._normalize_node_connections(node, self.archived_connections)
            node._connection_lookup = {
                **self.connections,
                **self.archived_connections,
            }

    def _normalize_node_connections(
        self,
        node: NodeGene,
        target: Dict[str, ConnectionGene],
    ) -> None:
        pending = list(node._pending_children)
        for connection in pending:
            base = connection.name or "connection"
            name = base
            index = 1
            while name in target and target[name] is not connection:
                index += 1
                name = f"{base}_{index}"
            connection.name = name
            target[name] = connection
            node.child_connections.append(name)
        node._pending_children = []
        node._connection_lookup = target

    def validate_node_names(self) -> None:
        """Require one globally unique name for every active or archived node gene."""
        mismatched_names = [
            (node_name, node.name)
            for node_name, node in self.nodes.items()
            if node_name != node.name
        ]
        if mismatched_names:
            node_name, gene_name = mismatched_names[0]
            raise ValueError(
                f"Active node key {node_name!r} does not match gene name {gene_name!r}"
            )

        seen = set(self.nodes)
        for node in self.archived_nodes.values():
            if node.name in seen:
                raise ValueError(
                    f"Duplicate node gene name {node.name!r}; node names must be "
                    "unique across active and archived genes"
                )
            seen.add(node.name)

    def rebuild_node_name_allocator(self) -> None:
        """Rebuild O(1) name-allocation state from active and archived genes."""
        names = list(self.nodes)
        names.extend(self.archived_nodes)
        self._used_node_names = set(names)
        self._next_mutation_index = {}
        for name in names:
            match = re.fullmatch(r"(.+)_mut([1-9][0-9]*)", name)
            if match is None:
                continue
            base_name, index_text = match.groups()
            next_index = int(index_text) + 1
            self._next_mutation_index[base_name] = max(
                self._next_mutation_index.get(base_name, 1),
                next_index,
            )

    def validate_connection_names(self) -> None:
        mismatched = [
            (name, gene.name)
            for name, gene in self.connections.items()
            if name != gene.name
        ]
        mismatched.extend(
            (name, gene.name)
            for name, gene in self.archived_connections.items()
            if name != gene.name
        )
        if mismatched:
            key, name = mismatched[0]
            raise ValueError(
                f"Connection key {key!r} does not match gene name {name!r}"
            )
        duplicate = set(self.connections) & set(self.archived_connections)
        if duplicate:
            raise ValueError(
                f"Duplicate connection gene name {sorted(duplicate)[0]!r}"
            )

    def validate_references(self) -> None:
        all_connections = {**self.archived_connections, **self.connections}
        referenced_active_connections: Set[str] = set()
        for node_name, node in self.nodes.items():
            if len(node.child_connections) != len(set(node.child_connections)):
                raise ValueError(
                    f"Node {node_name!r} contains duplicate child connection names"
                )
            for connection_name in node.child_connections:
                if connection_name not in self.connections:
                    raise KeyError(
                        f"Node {node_name!r} references unknown active connection "
                        f"{connection_name!r}"
                    )
                referenced_active_connections.add(connection_name)
        for node_name, node in self.archived_nodes.items():
            if len(node.child_connections) != len(set(node.child_connections)):
                raise ValueError(
                    f"Archived node {node_name!r} contains duplicate child connections"
                )
            for connection_name in node.child_connections:
                if connection_name not in all_connections:
                    raise KeyError(
                        f"Archived node {node_name!r} references unknown connection "
                        f"{connection_name!r}"
                    )
        for connection in all_connections.values():
            if connection.child not in self.nodes and connection.child not in self.archived_nodes:
                raise KeyError(
                    f"Connection {connection.name!r} targets unknown node "
                    f"{connection.child!r}"
                )
        unreferenced = set(self.connections) - referenced_active_connections
        if unreferenced:
            raise ValueError(
                f"Active connection {sorted(unreferenced)[0]!r} is not referenced "
                "by any active node"
            )

    def rebuild_connection_name_allocator(self) -> None:
        names = [*self.connections, *self.archived_connections]
        self._used_connection_names = set(names)
        self._next_connection_mutation_index = {}
        for name in names:
            match = re.fullmatch(r"(.+)_mut([1-9][0-9]*)", name)
            if match is None:
                continue
            base, index = match.groups()
            self._next_connection_mutation_index[base] = max(
                self._next_connection_mutation_index.get(base, 1),
                int(index) + 1,
            )
