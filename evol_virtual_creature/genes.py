from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ConnectionGene:
    child: str
    pos: Tuple[float, float, float]
    axis: Tuple[float, float, float]
    scale: float = 1.0
    terminal_only: bool = False
    motor_enabled: bool = True
    motor_gear: float = 100.0
    ctrlrange: Tuple[float, float] = (-1.0, 1.0)
    control_amp: float = 0.1
    control_freq: float = 10
    control_phase: float = 0.0
    control_phase_depth_scale: float = 0.0
    control_phase_order_scale: float = 0.0

    def phase_for(self, depth: int, order: int) -> float:
        return (
            self.control_phase
            + self.control_phase_depth_scale * (depth - 1)
            + self.control_phase_order_scale * order
        )


@dataclass
class NodeGene:
    name: str
    size: Tuple[float, float, float]
    joint_type: str = "hinge"
    joint_axis: Tuple[float, float, float] = (0, 1, 0)
    recursive_limit: int = 1
    children: List[ConnectionGene] = field(default_factory=list)


