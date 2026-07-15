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
    global_control_freq: float = 1.0
    control_mode: str = "neural"
    archived_connections: List[ConnectionGene] = field(default_factory=list)
    archived_nodes: List[NodeGene] = field(default_factory=list)
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

    def __post_init__(self):
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
        self.rebuild_node_name_allocator()

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
        for node in self.archived_nodes:
            if node.name in seen:
                raise ValueError(
                    f"Duplicate node gene name {node.name!r}; node names must be "
                    "unique across active and archived genes"
                )
            seen.add(node.name)

    def rebuild_node_name_allocator(self) -> None:
        """Rebuild O(1) name-allocation state from active and archived genes."""
        names = list(self.nodes)
        names.extend(node.name for node in self.archived_nodes)
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
