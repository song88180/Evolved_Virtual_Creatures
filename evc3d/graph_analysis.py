from __future__ import annotations

from typing import Dict, Tuple

from .genotype import Genotype


class PhenotypeBuildAbort(RuntimeError):
    """Raised when the genotype should not be expanded into a phenotype."""


class PhenotypeBodyLimitExceeded(PhenotypeBuildAbort):
    """Raised when the expanded creature would have more than max_body_count bodies."""


class GenotypeGraphError(PhenotypeBuildAbort):
    pass


class GenotypeGraphAnalyzer:
    def __init__(self, genotype: Genotype, max_body_count: int):
        if max_body_count < 1:
            raise ValueError("max_body_count must be at least 1")

        self.genotype = genotype
        self.max_body_count = max_body_count
        self.memoized_counts: Dict[Tuple[str, Tuple[Tuple[str, int], ...]], int] = {}
        self.active_states: set[Tuple[str, Tuple[Tuple[str, int], ...]]] = set()

    def validate(self) -> int:
        """
        Analyze expansion from the genotype graph before creating XML.

        Returns the exact number of generated bodies when it is within
        max_body_count. (A genotype node can expand into many bodies through
        recursion and symmetry.)
        Raises if an invalid/unbounded graph or oversized phenotype is found.
        """
        if self.genotype.root not in self.genotype.nodes:
            raise GenotypeGraphError(
                f"Unknown genotype root: {self.genotype.root!r}"
            )

        self._validate_recursive_limits()
        self._validate_terminal_connections()
        return self._count_expanded_bodies(self.genotype.root, {})

    def _validate_recursive_limits(self):
        for node in self.genotype.nodes.values():
            if (
                isinstance(node.recursive_limit, bool)
                or not isinstance(node.recursive_limit, int)
                or node.recursive_limit < 1
            ):
                raise GenotypeGraphError(
                    f"Node {node.name!r} has invalid recursive_limit "
                    f"{node.recursive_limit!r}; use a positive integer."
                )

    def _validate_terminal_connections(self):
        for node in self.genotype.nodes.values():
            for connection_name in node.child_connections:
                connection = self.genotype.connections[connection_name]
                if connection.terminal_only and connection.child == node.name:
                    raise GenotypeGraphError(
                        "Terminal-only connection cannot point to its own node: "
                        f"{node.name!r} -> {connection.child!r}."
                    )

    def _count_expanded_bodies(
        self,
        node_name: str,
        current_depths: Dict[str, int],
    ) -> int:
        state = (node_name, tuple(sorted(current_depths.items())))
        if state in self.active_states:
            raise GenotypeGraphError(
                "Genotype graph contains an unbounded recursive loop involving "
                f"node {node_name!r}."
            )

        cached_count = self.memoized_counts.get(state)
        if cached_count is not None:
            return cached_count

        node = self.genotype.nodes[node_name]
        node_depth = current_depths.get(node.name, 0) + 1
        next_depths = dict(current_depths)
        next_depths[node.name] = node_depth

        self.active_states.add(state)
        total_bodies = 1
        for connection_name in node.child_connections:
            connection = self.genotype.connections[connection_name]
            if connection.terminal_only and node_depth < node.recursive_limit:
                continue

            child_node = self.genotype.nodes.get(connection.child)
            if child_node is None:
                raise GenotypeGraphError(
                    f"Node {node.name!r} connects to unknown child "
                    f"{connection.child!r}."
                )

            child_depth = next_depths.get(child_node.name, 0)
            if child_depth >= child_node.recursive_limit:
                continue

            child_subtree_count = self._count_expanded_bodies(
                child_node.name,
                next_depths,
            )
            total_bodies += (2 ** len(connection.symmetry)) * child_subtree_count
            if total_bodies > self.max_body_count:
                raise PhenotypeBodyLimitExceeded(
                    "Maximum allowed number of bodies exceeded during "
                    f"genotype graph analysis: needed at least {total_bodies}, "
                    f"but max_body_count is {self.max_body_count}."
                )

        self.active_states.remove(state)
        self.memoized_counts[state] = total_bodies
        return total_bodies

