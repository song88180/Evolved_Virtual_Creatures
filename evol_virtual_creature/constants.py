"""Shared constants for evolutionary virtual creatures."""

from enum import Enum


class EnvironmentFamily(str, Enum):
    """Physical environment used to build a locomotion phenotype."""

    SWIMMING = "swimming"
    WALKING = "walking"
    FLYING = "flying"


FACE_NORMALS = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}
