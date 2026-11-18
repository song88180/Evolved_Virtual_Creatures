"""Neural controller weights: which connections carry weights, random initialization, per-weight mutation, and reshaping weights when a joint type changes."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
import math
import random

from ..controller_spec import (
    BALL_NEURAL_AXES,
    NEURAL_HIDDEN_SIZE,
    NEURAL_INPUT_DIMS,
    NEURAL_OUTPUT_DIMS,
    zero_neural_parameters,
)
from ..genes import ConnectionGene, NodeGene
from .settings import MutationSettings


class NeuralMutationMixin(MutationSettings):
    """Neural controller weights part of ``GenotypeMutationMixin``."""

    def _add_neural_parameter_paths(
        self,
        mutable_parameters: List[Tuple[Any, ...]],
        connection: ConnectionGene,
    ) -> None:
        for field_name in ("neural_b1", "neural_b2"):
            values = getattr(connection, field_name)
            for index in range(len(values)):
                mutable_parameters.append(("connection_neural", connection, field_name, index))
        for field_name in ("neural_w1", "neural_w2"):
            matrix = getattr(connection, field_name)
            for row_index, row in enumerate(matrix):
                for column_index in range(len(row)):
                    mutable_parameters.append(
                        (
                            "connection_neural",
                            connection,
                            field_name,
                            row_index,
                            column_index,
                        )
                    )

    def _ensure_connection_neural_parameters(
        self,
        connection: ConnectionGene,
    ) -> None:
        joint_type = self._connection_child_joint_type(connection)
        if joint_type not in NEURAL_INPUT_DIMS:
            return
        if self._connection_neural_parameters_match_joint_type(
            connection,
            joint_type,
        ):
            return
        (
            connection.neural_w1,
            connection.neural_b1,
            connection.neural_w2,
            connection.neural_b2,
        ) = zero_neural_parameters(joint_type)
        connection.neural_output_axes = self._default_neural_output_axes(
            connection,
            joint_type,
        )

    def _connection_child_joint_type(self, connection: ConnectionGene) -> str:
        child_node = self.nodes.get(connection.child)
        return child_node.joint_type if child_node is not None else "hinge"

    def _connection_supports_neural_mutation(
        self,
        connection: ConnectionGene,
    ) -> bool:
        return (
            self.control_mode == "neural"
            and self._connection_child_joint_type(connection) in NEURAL_INPUT_DIMS
        )

    def _connection_neural_parameters_match_joint_type(
        self,
        connection: ConnectionGene,
        joint_type: str,
    ) -> bool:
        expected_w1, expected_b1, expected_w2, expected_b2 = zero_neural_parameters(
            joint_type,
        )
        return (
            len(connection.neural_w1) == len(expected_w1)
            and all(
                len(row) == len(expected_row)
                for row, expected_row in zip(connection.neural_w1, expected_w1)
            )
            and len(connection.neural_b1) == len(expected_b1)
            and len(connection.neural_w2) == len(expected_w2)
            and all(
                len(row) == len(expected_row)
                for row, expected_row in zip(connection.neural_w2, expected_w2)
            )
            and len(connection.neural_b2) == len(expected_b2)
        )

    def _default_neural_output_axes(
        self,
        connection: ConnectionGene,
        joint_type: str,
    ) -> Tuple[Tuple[float, float, float], ...]:
        if joint_type == "ball":
            return tuple(BALL_NEURAL_AXES)
        return (self._normalized_axis(connection.axis),)

    def _random_neural_parameters(
        self,
        joint_type: str,
        random_source: random.Random,
    ) -> Tuple[Any, Any, Any, Any]:
        input_dim = NEURAL_INPUT_DIMS[joint_type]
        output_dim = NEURAL_OUTPUT_DIMS[joint_type]

        def sample() -> float:
            return random_source.gauss(0.0, self.NEURAL_INITIALIZATION_STD)

        return (
            tuple(
                tuple(sample() for _ in range(input_dim))
                for _ in range(NEURAL_HIDDEN_SIZE)
            ),
            tuple(sample() for _ in range(NEURAL_HIDDEN_SIZE)),
            tuple(
                tuple(sample() for _ in range(NEURAL_HIDDEN_SIZE))
                for _ in range(output_dim)
            ),
            tuple(sample() for _ in range(output_dim)),
        )

    def _initialize_connection_neural_parameters(
        self,
        connection: ConnectionGene,
        joint_type: str,
        random_source: random.Random,
    ) -> None:
        (
            connection.neural_w1,
            connection.neural_b1,
            connection.neural_w2,
            connection.neural_b2,
        ) = self._random_neural_parameters(joint_type, random_source)
        if joint_type == "ball":
            connection.neural_output_axes = self._orthonormal_axes(connection.axis)
        else:
            connection.neural_output_axes = (self._normalized_axis(connection.axis),)

    def _normalized_axis(
        self,
        axis: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        magnitude = math.sqrt(sum(component * component for component in axis))
        if magnitude <= 1e-12:
            return (0.0, 1.0, 0.0)
        return tuple(component / magnitude for component in axis)

    def _orthonormal_axes(
        self,
        primary_axis: Tuple[float, float, float],
    ) -> Tuple[Tuple[float, float, float], ...]:
        primary = self._normalized_axis(primary_axis)
        seed = min(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            key=lambda candidate: abs(sum(a * b for a, b in zip(primary, candidate))),
        )
        second = self._normalized_axis(self._cross(primary, seed))
        third = self._normalized_axis(self._cross(primary, second))
        return primary, second, third

    @staticmethod
    def _cross(
        left: Tuple[float, float, float],
        right: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        return (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        )

    def _mutate_connection_neural_parameter(
        self,
        parameter_path: Tuple[Any, ...],
        random_source: random.Random,
    ) -> str:
        _, connection, field_name, first_index, *maybe_second_index = parameter_path
        target = self._describe_parameter_target(("connection", connection, field_name))
        if not self._connection_supports_neural_mutation(connection):
            return (
                f"{target}.{field_name}: unchanged; connection does not currently "
                "support neural control"
            )
        self._ensure_connection_neural_parameters(connection)
        values = getattr(connection, field_name)
        if maybe_second_index:
            row_index = first_index
            column_index = maybe_second_index[0]
            matrix = [list(row) for row in values]
            if row_index >= len(matrix) or column_index >= len(matrix[row_index]):
                return (
                    f"{target}.{field_name}[{row_index}][{column_index}]: "
                    "unchanged; stale neural parameter path"
                )
            old_value = matrix[row_index][column_index]
            matrix[row_index][column_index] = old_value + self._normal_mutation_delta(
                old_value,
                self.MIN_NEURAL_WEIGHT_MUTATION_STD,
                random_source,
            )
            setattr(connection, field_name, tuple(tuple(row) for row in matrix))
            return (
                f"{target}.{field_name}[{row_index}][{column_index}]: "
                f"{old_value!r} -> {matrix[row_index][column_index]!r}"
            )

        vector = list(values)
        if first_index >= len(vector):
            return (
                f"{target}.{field_name}[{first_index}]: unchanged; "
                "stale neural parameter path"
            )
        old_value = vector[first_index]
        vector[first_index] = old_value + self._normal_mutation_delta(
            old_value,
            self.MIN_NEURAL_WEIGHT_MUTATION_STD,
            random_source,
        )
        setattr(connection, field_name, tuple(vector))
        return f"{target}.{field_name}[{first_index}]: {old_value!r} -> {vector[first_index]!r}"

    def _repair_neural_parameters_after_mutation(
        self,
        parameter_type: str,
        owner: Any,
        field_name: str,
        random_source: random.Random,
    ) -> None:
        if (
            parameter_type == "connection"
            and isinstance(owner, ConnectionGene)
            and field_name == "child"
        ):
            self._reset_connection_neural_parameters(owner, random_source)
            return
        if (
            parameter_type == "node"
            and isinstance(owner, NodeGene)
            and field_name == "joint_type"
        ):
            self._reset_incoming_neural_parameters(owner.name, random_source)

    def _reset_incoming_neural_parameters(
        self,
        child_name: str,
        random_source: Optional[random.Random] = None,
    ) -> None:
        for connection in self.connections.values():
            if connection.child == child_name:
                self._reset_connection_neural_parameters(connection, random_source)

    def _reset_connection_neural_parameters(
        self,
        connection: ConnectionGene,
        random_source: Optional[random.Random] = None,
    ) -> None:
        if self.control_mode != "neural":
            return
        joint_type = self._connection_child_joint_type(connection)
        if joint_type not in NEURAL_INPUT_DIMS:
            return
        rng = random_source if random_source is not None else random
        if self._connection_neural_parameters_match_joint_type(connection, joint_type):
            if joint_type == "ball" and len(connection.neural_output_axes) != 3:
                connection.neural_output_axes = self._orthonormal_axes(connection.axis)
            elif joint_type != "ball":
                connection.neural_output_axes = (self._normalized_axis(connection.axis),)
            return
        if self._connection_neural_parameters_match_joint_type(connection, "hinge"):
            self._expand_single_axis_neural_parameters(connection, rng)
            return
        if self._connection_neural_parameters_match_joint_type(connection, "ball"):
            self._contract_ball_neural_parameters(connection, rng)
            return
        self._initialize_connection_neural_parameters(connection, joint_type, rng)

    def _expand_single_axis_neural_parameters(
        self,
        connection: ConnectionGene,
        random_source: random.Random,
    ) -> None:
        old_w1 = connection.neural_w1
        old_w2 = connection.neural_w2[0]
        old_b2 = connection.neural_b2[0]
        retained_axis = self._normalized_axis(connection.axis)
        new_w1, _, new_w2, new_b2 = self._random_neural_parameters(
            "ball",
            random_source,
        )
        migrated_w1 = [list(row) for row in new_w1]
        for hidden_index, old_row in enumerate(old_w1):
            migrated_w1[hidden_index][0] = old_row[0]
            for component, axis_component in enumerate(retained_axis):
                migrated_w1[hidden_index][2 + component] = old_row[1] * axis_component
                migrated_w1[hidden_index][5 + component] = old_row[2] * axis_component
            migrated_w1[hidden_index][8:17] = old_row[3:12]
        output_rows = [list(row) for row in new_w2]
        output_rows[0] = list(old_w2)
        output_biases = list(new_b2)
        output_biases[0] = old_b2
        connection.neural_w1 = tuple(tuple(row) for row in migrated_w1)
        connection.neural_w2 = tuple(tuple(row) for row in output_rows)
        connection.neural_b2 = tuple(output_biases)
        connection.neural_output_axes = self._orthonormal_axes(retained_axis)

    def _contract_ball_neural_parameters(
        self,
        connection: ConnectionGene,
        random_source: random.Random,
    ) -> None:
        selected_index = random_source.randrange(3)
        axes = (
            connection.neural_output_axes
            if len(connection.neural_output_axes) == 3
            else BALL_NEURAL_AXES
        )
        selected_axis = self._normalized_axis(tuple(axes[selected_index]))
        migrated_w1 = []
        for old_row in connection.neural_w1:
            position_weight = sum(
                old_row[2 + component] * selected_axis[component]
                for component in range(3)
            )
            velocity_weight = sum(
                old_row[5 + component] * selected_axis[component]
                for component in range(3)
            )
            migrated_w1.append(
                (old_row[0], position_weight, velocity_weight, *old_row[8:17])
            )
        connection.neural_w1 = tuple(migrated_w1)
        connection.neural_w2 = (connection.neural_w2[selected_index],)
        connection.neural_b2 = (connection.neural_b2[selected_index],)
        connection.neural_output_axes = (selected_axis,)
        connection.axis = selected_axis
