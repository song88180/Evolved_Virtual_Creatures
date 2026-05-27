from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .genes import ConnectionGene, NodeGene
from .genotype_mutation import GenotypeMutationMixin


@dataclass
class Genotype(GenotypeMutationMixin):
    root: str
    nodes: Dict[str, NodeGene]
    archived_connections: List[ConnectionGene] = field(default_factory=list)
    archived_nodes: List[NodeGene] = field(default_factory=list)
