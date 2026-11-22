"""Mutating one field value: resolving a parameter path, per-field mutation specs, numeric/categorical mutation, and size/orientation repair."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
import random

from ..genes import (
    CHILD_JOINT_TYPES,
    BODY_SHAPES,
    MAX_BODY_SIZE,
    MIN_BODY_SIZE,
    ATTACHMENT_FACES,
    SYMMETRY_PLANES,
    ConnectionGene,
    NodeGene,
    child_orientation_is_valid,
    normalize_size,
    orientation_for_parent_face,
    wrap_degrees,
)
from .settings import MutationSettings


class ValueMutationMixin(MutationSettings):
    """Mutating one field value part of ``GenotypeMutationMixin``."""

    def _mutate_parameter_path(
        self,
        parameter_path: Tuple[Any, ...],
        random_source: random.Random,
    ) -> str:
        parameter_type = parameter_path[0]
        if parameter_type.startswith(self.BRAIN_MUTATION_PREFIX):
            return self._mutate_brain_parameter(parameter_path, random_source)
        if (
            parameter_type
            in {
                "connection",
                "connection_origin",
                "connection_destination",
                "connection_type",
                "connection_replacement",
                "connection_origin_node_replacement",
                "connection_destination_node_replacement",
                "connection_removal",
                "connection_neural",
            }
            and not self._is_active_connection(parameter_path[1])
        ):
            return (
                f"{self._describe_connection_source(parameter_path[1])}: "
                "unchanged; connection is archived"
            )

        origin_bound_parameter_types = {
            "connection_origin",
            "connection_replacement",
            "connection_origin_node_replacement",
            "connection_removal",
        }
        if (
            parameter_type in origin_bound_parameter_types
            and len(parameter_path) > 2
            and parameter_path[1].name
            not in self.nodes[parameter_path[2]].child_connections
        ):
            return (
                f"connection from {parameter_path[2]!r}: unchanged; "
                "connection is no longer attached to this origin"
            )

        if parameter_type == "connection_neural":
            return self._mutate_connection_neural_parameter(
                parameter_path,
                random_source,
            )
        if parameter_type == "connection_origin":
            return self._mutate_connection_origin(
                parameter_path[1],
                random_source,
                parameter_path[2] if len(parameter_path) > 2 else None,
            )
        if parameter_type == "connection_destination":
            return self._mutate_connection_destination(parameter_path[1], random_source)
        if parameter_type == "connection_type":
            return self._mutate_connection_type(parameter_path[1], random_source)
        if parameter_type == "connection_replacement":
            return self._mutate_connection_replacement(
                parameter_path[1],
                random_source,
                parameter_path[2] if len(parameter_path) > 2 else None,
            )
        if parameter_type == "connection_origin_node_replacement":
            return self._mutate_connection_origin_node_replacement(
                parameter_path[1],
                random_source,
                parameter_path[2] if len(parameter_path) > 2 else None,
            )
        if parameter_type == "connection_destination_node_replacement":
            return self._mutate_connection_destination_node_replacement(
                parameter_path[1],
                random_source,
            )
        if parameter_type == "connection_addition":
            return self._mutate_connection_addition(random_source)
        if parameter_type == "connection_new_node_addition":
            return self._mutate_connection_new_node_addition(random_source)
        if parameter_type == "fresh_node_connection_addition":
            return self._mutate_fresh_node_connection_addition(random_source)
        if parameter_type == "connection_removal":
            return self._mutate_connection_removal(
                parameter_path[1],
                parameter_path[2] if len(parameter_path) > 2 else None,
            )

        owner, field_name, index = self._resolve_parameter_path(parameter_path)
        if (
            parameter_type == "connection"
            and field_name == "terminal_only"
            and not owner.terminal_only
            and self._connection_is_self_pointing(owner)
        ):
            return (
                f"{self._describe_parameter_target(parameter_path)}.terminal_only: "
                "unchanged; self-pointing connections cannot be terminal-only"
            )
        mutation_kind, min_mutation_std, keep_ordered_pair = self._mutation_spec(
            field_name,
        )
        target = self._describe_parameter_target(parameter_path)

        if index is None:
            value = getattr(owner, field_name)
            mutated_value = self._mutate_value(
                value,
                random_source,
                mutation_kind,
                min_mutation_std,
            )
            setattr(owner, field_name, mutated_value)
            self._repair_node_size_if_needed(owner, field_name)
            self._repair_connection_orientation_if_needed(owner, field_name)
            self._repair_neural_parameters_after_mutation(
                parameter_type,
                owner,
                field_name,
                random_source,
            )
            return f"{target}.{field_name}: {value!r} -> {mutated_value!r}"

        original_values = getattr(owner, field_name)
        values = list(original_values)
        old_value = values[index]
        values[index] = self._mutate_value(
            values[index],
            random_source,
            mutation_kind,
            min_mutation_std,
        )
        mutated_value = values[index]

        if keep_ordered_pair:
            values = sorted(values)
            if values[0] == values[1]:
                values[0] -= min_mutation_std
                values[1] += min_mutation_std

        final_values = type(original_values)(values)
        setattr(owner, field_name, final_values)
        self._repair_node_size_if_needed(owner, field_name)
        self._repair_connection_orientation_if_needed(owner, field_name)
        self._repair_neural_parameters_after_mutation(
            parameter_type,
            owner,
            field_name,
            random_source,
        )
        final_values = getattr(owner, field_name)

        return (
            f"{target}.{field_name}[{index}]: {old_value!r} -> {mutated_value!r}"
            f" (final {field_name}: {final_values!r})"
        )

    def _describe_parameter_target(self, parameter_path: Tuple[Any, ...]) -> str:
        parameter_type = parameter_path[0]

        if parameter_type == "genotype":
            return "genotype"

        if parameter_type == "node":
            _, node_name, *_ = parameter_path
            return f"node '{node_name}'"

        if parameter_type == "connection":
            _, connection, *_ = parameter_path
            node_name, child_index = self._find_connection_owner(connection)
            return (
                f"connection '{node_name}' -> '{connection.child}'"
                f" #{child_index}"
            )

        if parameter_type in {
            "connection_origin",
            "connection_destination",
            "connection_type",
            "connection_replacement",
            "connection_origin_node_replacement",
            "connection_destination_node_replacement",
            "connection_removal",
        }:
            _, connection, *maybe_origin = parameter_path
            node_name, child_index = self._find_connection_owner(
                connection, maybe_origin[0] if maybe_origin else None
            )
            return (
                f"connection '{node_name}' -> '{connection.child}'"
                f" #{child_index}"
            )

        raise ValueError(f"Unknown mutable parameter path type: {parameter_type}")

    def _resolve_parameter_path(
        self,
        parameter_path: Tuple[Any, ...],
    ) -> Tuple[Any, str, Optional[int]]:
        parameter_type = parameter_path[0]

        if parameter_type == "genotype":
            _, field_name, *maybe_index = parameter_path
            return self, field_name, self._path_index(maybe_index)

        if parameter_type == "node":
            _, node_name, field_name, *maybe_index = parameter_path
            return self.nodes[node_name], field_name, self._path_index(maybe_index)

        if parameter_type == "connection":
            _, connection, field_name, *maybe_index = parameter_path
            return connection, field_name, self._path_index(maybe_index)

        raise ValueError(f"Unknown mutable parameter path type: {parameter_type}")

    def _path_index(self, maybe_index: List[Any]) -> Optional[int]:
        if not maybe_index:
            return None
        return maybe_index[0]

    def _mutation_spec(
        self,
        field_name: str,
    ) -> Tuple[str, float, bool]:
        field_specs = {
            "size": ("positive", self.MIN_SIZE_MUTATION_STD, False),
            "joint_type": ("child_joint_type", 0.0, False),
            "shape": ("body_shape", 0.0, False),
            "recursive_limit": (
                "positive_integer",
                self.MIN_INTEGER_MUTATION_STD,
                False,
            ),
            "parent_face": ("attachment_face", 0.0, False),
            "surface_uv": (
                "bounded_surface_coordinate",
                self.MIN_SURFACE_COORD_MUTATION_STD,
                False,
            ),
            "child_surface_uv": (
                "bounded_surface_coordinate",
                self.MIN_SURFACE_COORD_MUTATION_STD,
                False,
            ),
            "orientation": (
                "orientation_degrees",
                self.MIN_ORIENTATION_MUTATION_STD,
                False,
            ),
            "symmetry": ("symmetry_planes", 0.0, False),
            "axis": ("relative", self.MIN_AXIS_MUTATION_STD, False),
            "scale": ("positive", self.MIN_SCALE_MUTATION_STD, False),
            "terminal_only": ("boolean", 0.0, False),
            "motor_enabled": ("boolean", 0.0, False),
            "motor_gear": ("positive", self.MIN_MOTOR_GEAR_MUTATION_STD, False),
            "ctrlrange": ("relative", self.MIN_CTRLRANGE_MUTATION_STD, True),
            "control_amp": ("positive", self.MIN_CONTROL_AMP_MUTATION_STD, False),
            "control_freq": ("harmonic_control_freq", 0.0, False),
            "global_control_freq": (
                "positive",
                self.MIN_GLOBAL_CONTROL_FREQ_MUTATION_STD,
                False,
            ),
            "control_phase": ("relative", self.MIN_PHASE_MUTATION_STD, False),
            "control_phase_depth_scale": (
                "relative",
                self.MIN_PHASE_MUTATION_STD,
                False,
            ),
            "control_phase_order_scale": (
                "relative",
                self.MIN_PHASE_MUTATION_STD,
                False,
            ),
        }
        return field_specs[field_name]

    def _repair_node_size_if_needed(
        self,
        owner: Any,
        field_name: str,
    ) -> None:
        if not isinstance(owner, NodeGene):
            return
        if field_name not in {"shape", "size"}:
            return
        normalized = normalize_size(owner.size, owner.shape)
        owner.size = tuple(
            max(MIN_BODY_SIZE, min(value, MAX_BODY_SIZE))
            for value in normalized
        )

    def _repair_connection_orientation_if_needed(
        self,
        owner: Any,
        field_name: str,
    ) -> None:
        if not isinstance(owner, ConnectionGene):
            return
        if field_name not in {"parent_face", "orientation"}:
            return
        if child_orientation_is_valid(owner.parent_face, owner.orientation):
            return
        owner.orientation = orientation_for_parent_face(owner.parent_face)

    def _random_valid_connection_orientation(
        self,
        parent_face: str,
        random_source: random.Random,
    ) -> Tuple[float, float, float]:
        for _ in range(100):
            orientation = tuple(
                random_source.uniform(-180.0, 180.0) for _ in range(3)
            )
            if child_orientation_is_valid(parent_face, orientation):
                return orientation
        return orientation_for_parent_face(parent_face)

    def _child_joint_types_for_mutation(self) -> Tuple[str, ...]:
        if getattr(self, "_allow_slide_joint_for_mutation", False):
            return CHILD_JOINT_TYPES
        return tuple(
            joint_type
            for joint_type in CHILD_JOINT_TYPES
            if joint_type != "slide"
        )

    def _mutate_value(
        self,
        value: Any,
        random_source: random.Random,
        mutation_kind: str,
        min_mutation_std: float,
    ) -> Any:
        mutation_kind = mutation_kind.lower()

        if mutation_kind == "child_joint_type":
            return random_source.choice(
                [
                    joint_type
                    for joint_type in self._child_joint_types_for_mutation()
                    if joint_type != value
                ]
            )

        if mutation_kind == "body_shape":
            return random_source.choice(
                [shape for shape in BODY_SHAPES if shape != value]
            )

        if mutation_kind == "attachment_face":
            return random_source.choice(
                [face for face in ATTACHMENT_FACES if face != value]
            )

        if mutation_kind == "symmetry_planes":
            plane = random_source.choice(SYMMETRY_PLANES)
            selected_planes = set(value)
            if plane in selected_planes:
                selected_planes.remove(plane)
            else:
                selected_planes.add(plane)
            return tuple(
                candidate
                for candidate in SYMMETRY_PLANES
                if candidate in selected_planes
            )

        if mutation_kind == "boolean":
            return not value

        if mutation_kind == "harmonic_control_freq":
            value_as_float = float(value)
            harmonics = self.HARMONIC_CONTROL_FREQS
            if value_as_float not in harmonics:
                return min(
                    harmonics,
                    key=lambda candidate: abs(candidate - value_as_float),
                )
            return random_source.choice(
                [candidate for candidate in harmonics if candidate != value_as_float]
            )

        if mutation_kind in {"integer", "positive_integer"}:
            value_as_int = int(value)
            delta = int(round(
                self._normal_mutation_delta(
                    value_as_int,
                    min_mutation_std,
                    random_source,
                )
            ))
            if delta == 0:
                delta = random_source.choice((-1, 1))

            mutated_value = value_as_int + delta
            if mutation_kind == "positive_integer" and mutated_value < 1:
                mutated_value = value_as_int + abs(delta)
            return mutated_value

        if mutation_kind == "positive":
            value_as_float = float(value)
            delta = self._normal_mutation_delta(
                value_as_float,
                min_mutation_std,
                random_source,
            )
            mutated_value = max(self.MIN_POSITIVE_VALUE, value_as_float + delta)
            if mutated_value == value_as_float:
                mutated_value = value_as_float + abs(delta)
            return mutated_value

        if mutation_kind == "relative":
            return value + self._normal_mutation_delta(
                value,
                min_mutation_std,
                random_source,
            )

        if mutation_kind == "orientation_degrees":
            return wrap_degrees(
                value + self._normal_mutation_delta(
                    value,
                    min_mutation_std,
                    random_source,
                )
            )

        if mutation_kind == "bounded_surface_coordinate":
            return max(
                -1.0,
                min(
                    1.0,
                    value + self._normal_mutation_delta(
                        value,
                        min_mutation_std,
                        random_source,
                    ),
                ),
            )

        raise ValueError(f"Unknown mutation kind: {mutation_kind}")

    def _normal_mutation_delta(
        self,
        value: float,
        min_mutation_std: float,
        random_source: random.Random,
    ) -> float:
        mutation_std = max(
            abs(float(value)) * self.MUTATION_STD_FRACTION,
            min_mutation_std,
        )
        delta = random_source.gauss(0.0, mutation_std)
        if delta == 0.0:
            return mutation_std
        return delta
