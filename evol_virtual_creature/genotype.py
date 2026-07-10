from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

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
