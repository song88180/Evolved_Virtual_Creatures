"""Gene bookkeeping: unique naming, archiving and reactivating genes, pruning unreachable nodes, and connection lookups."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import copy

from ..genes import ConnectionGene, NodeGene
from .settings import MutationSettings


class GenePoolMixin(MutationSettings):
    """Gene bookkeeping part of ``GenotypeMutationMixin``."""

    def _new_node_name(self, base_name: str) -> str:
        index = self._next_mutation_index.get(base_name, 1)
        while True:
            candidate = f"{base_name}_mut{index}"
            if candidate not in self._used_node_names:
                self._used_node_names.add(candidate)
                self._next_mutation_index[base_name] = index + 1
                return candidate
            index += 1

    def _new_connection_name(self, base_name: str) -> str:
        index = self._next_connection_mutation_index.get(base_name, 1)
        while True:
            candidate = f"{base_name}_mut{index}"
            if candidate not in self._used_connection_names:
                self._used_connection_names.add(candidate)
                self._next_connection_mutation_index[base_name] = index + 1
                return candidate
            index += 1

    def _archive_node_snapshots(
        self,
        nodes: Sequence[NodeGene],
    ) -> List[NodeGene]:
        """Archive distinct snapshots using the normal nested naming rule."""
        archived_copies = [copy.deepcopy(node) for node in nodes]
        renamed = {
            node.name: self._new_node_name(node.name)
            for node in archived_copies
        }
        for node in archived_copies:
            old_name = node.name
            node.name = renamed[old_name]
            cloned_connection_names = []
            for connection_name in node.child_connections:
                source = self.connections.get(connection_name) or self.archived_connections[connection_name]
                connection = copy.deepcopy(source)
                connection.name = self._new_connection_name(source.name)
                connection.child = renamed.get(connection.child, connection.child)
                self.archived_connections[connection.name] = connection
                cloned_connection_names.append(connection.name)
            node.child_connections = cloned_connection_names
            self.archived_nodes[node.name] = node
        return archived_copies

    def _activate_node_gene(
        self,
        node: NodeGene,
        activating_node_names: Optional[set[str]] = None,
    ) -> NodeGene:
        if activating_node_names is None:
            activating_node_names = set()

        active_node = self.nodes.get(node.name)
        if active_node is not None:
            return active_node

        if self.archived_nodes.get(node.name) is not node:
            raise ValueError(f"Node gene {node.name!r} is neither active nor archived")
        new_node = self.archived_nodes.pop(node.name)
        self.nodes[new_node.name] = new_node
        activating_node_names.add(new_node.name)

        for connection_name in new_node.child_connections:
            connection = self.connections.get(connection_name) or self.archived_connections[connection_name]
            self._activate_connection_gene(connection)
            if connection.child in self.nodes or connection.child in activating_node_names:
                continue

            archived_child_node = self._find_archived_node(connection.child)
            if archived_child_node is not None:
                self._activate_node_gene(archived_child_node, activating_node_names)

        return new_node

    def _activate_connection_gene(self, connection: ConnectionGene) -> None:
        active = self.connections.get(connection.name)
        if active is connection:
            return
        if self.archived_connections.get(connection.name) is not connection:
            raise ValueError(
                f"Connection gene {connection.name!r} is neither active nor archived"
            )
        self.archived_connections.pop(connection.name)
        self.connections[connection.name] = connection

    def _archive_connection_if_unused(self, connection_name: str) -> None:
        if any(
            connection_name in node.child_connections
            for node in self.nodes.values()
        ):
            return
        connection = self.connections.pop(connection_name, None)
        if connection is not None:
            self.archived_connections[connection_name] = connection

    def _find_archived_node(self, node_name: str) -> Optional[NodeGene]:
        return self.archived_nodes.get(node_name)

    def _prune_disconnected_nodes(self) -> Optional[str]:
        reachable_node_names = self._reachable_node_names()
        disconnected_node_names = [
            node_name
            for node_name in self.nodes
            if node_name not in reachable_node_names
        ]
        if not disconnected_node_names:
            return None

        disconnected_nodes = [
            self.nodes[node_name] for node_name in disconnected_node_names
        ]
        for node_name in disconnected_node_names:
            del self.nodes[node_name]
        self.archived_nodes.update({node.name: node for node in disconnected_nodes})
        for connection_name in list(self.connections):
            self._archive_connection_if_unused(connection_name)

        removed_nodes = ", ".join(repr(node_name) for node_name in disconnected_node_names)
        return (
            f"pruned disconnected node(s): {removed_nodes}; archived "
            f"{len(disconnected_node_names)} node gene(s)"
        )

    def _reachable_node_names(self) -> set[str]:
        if self.root not in self.nodes:
            return set()

        reachable_node_names = set()
        pending_node_names = [self.root]
        while pending_node_names:
            node_name = pending_node_names.pop()
            if node_name in reachable_node_names or node_name not in self.nodes:
                continue

            reachable_node_names.add(node_name)
            for connection_name in self.nodes[node_name].child_connections:
                connection = self.connections[connection_name]
                if connection.child in self.nodes:
                    pending_node_names.append(connection.child)

        return reachable_node_names

    def _describe_connection_source(self, connection: ConnectionGene) -> str:
        try:
            origin, child_index = self._find_connection_owner(connection)
            return f"connection '{origin}' -> '{connection.child}' #{child_index}"
        except ValueError:
            archive_index = self._archived_connection_index(connection)
            return f"archived connection '{connection.child}' #{archive_index}"

    def _is_active_connection(self, connection: ConnectionGene) -> bool:
        return self.connections.get(connection.name) is connection

    def _archived_connection_index(self, connection: ConnectionGene) -> str:
        if self.archived_connections.get(connection.name) is connection:
            return connection.name
        raise ValueError("Connection is not archived")

    def _find_connection_owner(
        self, connection: ConnectionGene, origin_name: Optional[str] = None
    ) -> Tuple[str, int]:
        nodes = (
            [(origin_name, self.nodes[origin_name])]
            if origin_name is not None
            else self.nodes.items()
        )
        for node_name, node in nodes:
            for child_index, candidate_name in enumerate(node.child_connections):
                if candidate_name == connection.name:
                    return node_name, child_index

        raise ValueError("Connection is not attached to any genotype node")
