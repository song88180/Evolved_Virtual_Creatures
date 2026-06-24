from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Tuple


ATTACHMENT_FACES = ("+x", "-x", "+y", "-y", "+z", "-z")
SYMMETRY_PLANES = ("xy", "xz", "yz")
ARTICULATED_JOINT_TYPES = ("hinge", "slide", "ball")
JOINT_TYPES = ("free", *ARTICULATED_JOINT_TYPES)


@dataclass
class ConnectionGene:
    child: str
    axis: Tuple[float, float, float]
    parent_face: str = "+x"
    surface_uv: Tuple[float, float] = (0.0, 0.0)
    symmetry: Tuple[str, ...] = ()
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

    def __post_init__(self):
        if self.parent_face not in ATTACHMENT_FACES:
            valid_faces = ", ".join(ATTACHMENT_FACES)
            raise ValueError(
                f"Unknown parent face {self.parent_face!r}; expected one of "
                f"{valid_faces}"
            )
        if len(self.surface_uv) != 2:
            raise ValueError("surface_uv must contain exactly two coordinates")

        self.surface_uv = tuple(float(value) for value in self.surface_uv)
        if any(not math.isfinite(value) for value in self.surface_uv):
            raise ValueError("surface_uv coordinates must be finite")
        if any(value < -1.0 or value > 1.0 for value in self.surface_uv):
            raise ValueError("surface_uv coordinates must be between -1 and 1")

        unknown_planes = set(self.symmetry) - set(SYMMETRY_PLANES)
        if unknown_planes:
            valid_planes = ", ".join(SYMMETRY_PLANES)
            unknown = ", ".join(sorted(unknown_planes))
            raise ValueError(
                f"Unknown symmetry plane(s) {unknown}; expected any subset of "
                f"{valid_planes}"
            )
        if len(set(self.symmetry)) != len(self.symmetry):
            raise ValueError("symmetry planes must not contain duplicates")
        self.symmetry = tuple(
            plane for plane in SYMMETRY_PLANES if plane in self.symmetry
        )
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("scale must be finite and greater than zero")

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

    def __post_init__(self):
        if self.joint_type not in JOINT_TYPES:
            valid_types = ", ".join(JOINT_TYPES)
            raise ValueError(
                f"Unknown joint type {self.joint_type!r}; expected one of "
                f"{valid_types}"
            )


