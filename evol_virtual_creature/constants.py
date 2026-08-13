"""Shared constants for evolutionary virtual creatures."""

from enum import Enum


class EnvironmentFamily(str, Enum):
    """Physical environment used to build a locomotion phenotype."""

    SWIMMING = "swimming"
    WALKING = "walking"
    FLYING = "flying"


class TaskName(str, Enum):
    """Canonical names for supported locomotion tasks."""

    def __new__(cls, value: str, environment_family: EnvironmentFamily):
        member = str.__new__(cls, value)
        member._value_ = value
        member.environment_family = environment_family
        return member

    SWIMMING_X = ("swimming_x", EnvironmentFamily.SWIMMING)
    SWIMMING_AWAY = ("swimming_away", EnvironmentFamily.SWIMMING)
    WALKING_X = ("walking_x", EnvironmentFamily.WALKING)
    WALKING_AWAY = ("walking_away", EnvironmentFamily.WALKING)
    FLYING_X = ("flying_x", EnvironmentFamily.FLYING)
    FLYING_AWAY = ("flying_away", EnvironmentFamily.FLYING)

    environment_family: EnvironmentFamily

    def __str__(self) -> str:
        return self.value


FACE_NORMALS = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}
