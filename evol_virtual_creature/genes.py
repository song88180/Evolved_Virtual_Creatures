from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Tuple


ATTACHMENT_FACES = ("+x", "-x", "+y", "-y", "+z", "-z")
FACE_NORMALS = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}
SYMMETRY_PLANES = ("xy", "xz", "yz")
ARTICULATED_JOINT_TYPES = ("hinge", "slide", "ball")
CHILD_JOINT_TYPES = ("fixed", *ARTICULATED_JOINT_TYPES)
JOINT_TYPES = ("free", *CHILD_JOINT_TYPES)
BODY_SHAPES = ("box", "ellipsoid", "capsule", "cylinder")
ROUND_BODY_SHAPES = ("capsule", "cylinder")
MIN_BODY_SIZE = 0.01
MAX_BODY_SIZE = 1.0
CONTROL_MODES = ("neural", "sine")
IDENTITY_ORIENTATION = (0.0, 0.0, 0.0)
FACE_ALIGNED_ORIENTATIONS = {
    "+x": (0.0, 0.0, 0.0),
    "-x": (0.0, 0.0, 180.0),
    "+y": (0.0, 0.0, 90.0),
    "-y": (0.0, 0.0, -90.0),
    "+z": (0.0, -90.0, 0.0),
    "-z": (0.0, 90.0, 0.0),
}


def normalize_orientation(orientation: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if len(orientation) != 3:
        raise ValueError("orientation must contain exactly three Euler angles")
    normalized = tuple(float(value) for value in orientation)
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError("orientation angles must be finite")
    return normalized


def normalize_surface_uv(
    surface_uv: Tuple[float, float],
    field_name: str,
) -> Tuple[float, float]:
    if len(surface_uv) != 2:
        raise ValueError(f"{field_name} must contain exactly two coordinates")

    normalized = tuple(float(value) for value in surface_uv)
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError(f"{field_name} coordinates must be finite")
    if any(value < -1.0 or value > 1.0 for value in normalized):
        raise ValueError(f"{field_name} coordinates must be between -1 and 1")
    return normalized


def normalize_size(
    size: Tuple[float, float, float],
    shape: str,
) -> Tuple[float, float, float]:
    if len(size) != 3:
        raise ValueError("size must contain exactly three dimensions")
    normalized = tuple(float(value) for value in size)
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError("size dimensions must be finite")
    if any(value <= 0.0 for value in normalized):
        raise ValueError("size dimensions must be greater than zero")
    if shape in ROUND_BODY_SHAPES:
        radius = min(normalized[1], normalized[2])
        normalized = (normalized[0], radius, radius)
    return normalized


def normalize_numeric_vector(
    values: Tuple[float, ...],
    field_name: str,
) -> Tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError(f"{field_name} values must be finite")
    return normalized


def normalize_numeric_matrix(
    values: Tuple[Tuple[float, ...], ...],
    field_name: str,
) -> Tuple[Tuple[float, ...], ...]:
    return tuple(
        normalize_numeric_vector(tuple(row), f"{field_name} row")
        for row in values
    )


def wrap_degrees(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def euler_degrees_to_matrix(
    orientation: Tuple[float, float, float],
) -> Tuple[Tuple[float, float, float], ...]:
    roll, pitch, yaw = (math.radians(angle) for angle in orientation)
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    return (
        (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
        (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
        (-sy, cy * sx, cy * cx),
    )


def child_orientation_dot(parent_face: str, orientation: Tuple[float, float, float]) -> float:
    normal = FACE_NORMALS[parent_face]
    rotation = euler_degrees_to_matrix(orientation)
    rotated_child_x_axis = (rotation[0][0], rotation[1][0], rotation[2][0])
    return sum(
        axis_component * normal_component
        for axis_component, normal_component in zip(rotated_child_x_axis, normal)
    )


def child_orientation_is_valid(
    parent_face: str,
    orientation: Tuple[float, float, float],
) -> bool:
    return child_orientation_dot(parent_face, orientation) > 1e-12


def orientation_for_parent_face(parent_face: str) -> Tuple[float, float, float]:
    return FACE_ALIGNED_ORIENTATIONS[parent_face]


@dataclass
class ConnectionGene:
    child: str
    axis: Tuple[float, float, float]
    parent_face: str = "+x"
    surface_uv: Tuple[float, float] = (0.0, 0.0)
    child_surface_uv: Tuple[float, float] = (0.0, 0.0)
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
    neural_w1: Tuple[Tuple[float, ...], ...] = ()
    neural_b1: Tuple[float, ...] = ()
    neural_w2: Tuple[Tuple[float, ...], ...] = ()
    neural_b2: Tuple[float, ...] = ()
    neural_output_axes: Tuple[Tuple[float, float, float], ...] = ()
    orientation: Tuple[float, float, float] = IDENTITY_ORIENTATION

    def __post_init__(self):
        if self.parent_face not in ATTACHMENT_FACES:
            valid_faces = ", ".join(ATTACHMENT_FACES)
            raise ValueError(
                f"Unknown parent face {self.parent_face!r}; expected one of "
                f"{valid_faces}"
            )
        self.orientation = normalize_orientation(self.orientation)
        if not child_orientation_is_valid(self.parent_face, self.orientation):
            raise ValueError(
                "connection orientation must rotate the child local +X axis "
                "within 90 degrees of the parent attachment surface normal"
            )
        self.surface_uv = normalize_surface_uv(self.surface_uv, "surface_uv")
        self.child_surface_uv = normalize_surface_uv(
            self.child_surface_uv,
            "child_surface_uv",
        )
        self.neural_w1 = normalize_numeric_matrix(self.neural_w1, "neural_w1")
        self.neural_b1 = normalize_numeric_vector(self.neural_b1, "neural_b1")
        self.neural_w2 = normalize_numeric_matrix(self.neural_w2, "neural_w2")
        self.neural_b2 = normalize_numeric_vector(self.neural_b2, "neural_b2")
        self.neural_output_axes = normalize_numeric_matrix(
            self.neural_output_axes,
            "neural_output_axes",
        )
        if any(len(axis) != 3 for axis in self.neural_output_axes):
            raise ValueError("neural_output_axes rows must contain exactly 3 values")

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
    recursive_limit: int = 1
    children: List[ConnectionGene] = field(default_factory=list)
    orientation: Tuple[float, float, float] = IDENTITY_ORIENTATION
    shape: str = "box"

    def __post_init__(self):
        if self.shape not in BODY_SHAPES:
            valid_shapes = ", ".join(BODY_SHAPES)
            raise ValueError(
                f"Unknown body shape {self.shape!r}; expected one of "
                f"{valid_shapes}"
            )
        self.size = normalize_size(self.size, self.shape)
        self.orientation = normalize_orientation(self.orientation)
        if self.joint_type not in JOINT_TYPES:
            valid_types = ", ".join(JOINT_TYPES)
            raise ValueError(
                f"Unknown joint type {self.joint_type!r}; expected one of "
                f"{valid_types}"
            )
