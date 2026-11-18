"""Topology operators: adding, removing, relinking, and replacing connections and nodes, plus fresh-gene generation and child sizing."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
import copy
import random

from ..controller_spec import NEURAL_INPUT_DIMS
from ..genes import (
    BODY_SHAPES,
    MAX_BODY_SIZE,
    MIN_BODY_SIZE,
    ROUND_BODY_SHAPES,
    ATTACHMENT_FACES,
    SYMMETRY_PLANES,
    ConnectionGene,
    NodeGene,
)
from .settings import MutationSettings


class TopologyMutationMixin(MutationSettings):
    """Topology operators part of ``GenotypeMutationMixin``."""

    def _mutate_connection_origin(
        self,
        connection: ConnectionGene,
        random_source: random.Random,
        origin_name: Optional[str] = None,
    ) -> str:
        old_origin, old_index = self._find_connection_owner(connection, origin_name)
        possible_origins = [
            node_name
            for node_name in self.nodes
            if node_name != old_origin
            and (not connection.terminal_only or node_name != connection.child)
            and connection.name not in self.nodes[node_name].child_connections
        ]
        if not possible_origins:
            return (
                f"connection '{old_origin}' -> '{connection.child}'"
                f" #{old_index}.origin: unchanged; no alternative nodes"
            )

        new_origin = random_source.choice(possible_origins)
        self.nodes[old_origin].child_connections.pop(old_index)
        self.nodes[new_origin].child_connections.append(connection.name)
        new_index = len(self.nodes[new_origin].child_connections) - 1

        return (
            f"connection origin for '{old_origin}' -> '{connection.child}'"
            f" #{old_index}: {old_origin!r} -> {new_origin!r}"
            f" (now connection '{new_origin}' -> '{connection.child}' #{new_index})"
        )

    def _mutate_connection_destination(
        self,
        connection: ConnectionGene,
        random_source: random.Random,
    ) -> str:
        origin, child_index = self._find_connection_owner(connection)
        old_destination = connection.child
        possible_destinations = [
            node_name
            for node_name, node in self.nodes.items()
            if node_name != old_destination
            and node.joint_type != "free"
            and (
                not connection.terminal_only
                or not any(
                    connection.name in candidate.child_connections
                    and candidate.name == node_name
                    for candidate in self.nodes.values()
                )
            )
        ]
        if not possible_destinations:
            return (
                f"connection '{origin}' -> '{old_destination}'"
                f" #{child_index}.destination: unchanged; no alternative nodes"
            )

        new_destination = random_source.choice(possible_destinations)
        connection.child = new_destination
        self._reset_connection_neural_parameters(connection, random_source)

        return (
            f"connection destination for '{origin}' -> '{old_destination}'"
            f" #{child_index}: {old_destination!r} -> {new_destination!r}"
            f" (now connection '{origin}' -> '{new_destination}' #{child_index})"
        )

    def _mutate_connection_type(
        self,
        connection: ConnectionGene,
        random_source: random.Random,
    ) -> str:
        origin, child_index = self._find_connection_owner(connection)
        possible_connections = [
            candidate
            for candidate in self._connection_coding_pool()
            if candidate is not connection
        ]
        if not possible_connections:
            return (
                f"connection '{origin}' -> '{connection.child}'"
                f" #{child_index}.type: unchanged; no alternative connections"
            )

        source_connection = random_source.choice(possible_connections)
        source_label = self._describe_connection_source(source_connection)
        destination = connection.child
        connection_fields = self.CONNECTION_MUTABLE_FIELDS
        old_values = {
            field_name: getattr(connection, field_name)
            for field_name in connection_fields
        }

        for field_name in connection_fields:
            setattr(connection, field_name, getattr(source_connection, field_name))
        connection.child = destination
        if self._connection_is_self_pointing(connection):
            connection.terminal_only = False
        self._repair_connection_orientation_if_needed(connection, "orientation")
        self._reset_connection_neural_parameters(connection, random_source)

        changed_fields = [
            field_name
            for field_name in connection_fields
            if getattr(connection, field_name) != old_values[field_name]
        ]
        changed_field_list = ", ".join(changed_fields) if changed_fields else "none"

        return (
            f"connection type for '{origin}' -> '{destination}' #{child_index}: "
            f"copied from {source_label}"
            f" (changed fields: {changed_field_list})"
        )

    def _mutate_connection_replacement(
        self,
        connection: ConnectionGene,
        random_source: random.Random,
        origin_name: Optional[str] = None,
    ) -> str:
        origin, child_index = self._find_connection_owner(connection, origin_name)
        old_connection = connection
        new_connection = copy.deepcopy(old_connection)
        new_connection.name = self._new_connection_name(old_connection.name)
        self.connections[new_connection.name] = new_connection
        self.nodes[origin].child_connections[child_index] = new_connection.name
        self._archive_connection_if_unused(old_connection.name)

        mutation_description = self._mutate_new_connection(new_connection, random_source)

        return (
            f"connection replacement for '{origin}' -> '{new_connection.child}'"
            f" #{child_index}: created new connection and archived old connection"
            f" {old_connection.name!r} if unused; {mutation_description}"
        )

    def _mutate_connection_addition(
        self,
        random_source: random.Random,
    ) -> str:
        candidates = []
        for connection in self._connection_coding_pool():
            for origin in self.nodes.values():
                if connection.name in origin.child_connections:
                    continue
                if connection.terminal_only and connection.child == origin.name:
                    continue
                candidates.append((origin, connection))
        if not candidates:
            return "connection addition: unchanged; no reusable connection and origin"
        origin, connection = random_source.choice(candidates)
        if connection.child not in self.nodes:
            child = self._find_archived_node(connection.child)
            if child is None:
                return "connection addition: unchanged; destination is unavailable"
            self._activate_node_gene(child)
        self._activate_connection_gene(connection)
        origin.child_connections.append(connection.name)
        return (
            f"connection addition: attached shared connection {connection.name!r} "
            f"to node {origin.name!r}"
        )

    def _mutate_connection_new_node_addition(
        self,
        random_source: random.Random,
    ) -> str:
        possible_origins = list(self.nodes.values())
        possible_node_templates = [
            node
            for node in self._node_coding_pool()
            if node.joint_type != "free"
        ]
        if not possible_origins or not possible_node_templates:
            return "connection new-node addition: unchanged; no valid node template"

        origin_node = random_source.choice(possible_origins)
        node_template = random_source.choice(possible_node_templates)
        new_node = self._new_node_from_existing(node_template, random_source)
        new_node.size = self._random_child_size_from_parent(
            origin_node,
            random_source,
        )
        new_node.child_connections = []
        mutated_node_field = self._mutate_new_node(new_node, random_source)
        self._clamp_child_size_to_parent(new_node, origin_node)
        self.nodes[new_node.name] = new_node
        new_index, source_label, mutated_connection_description = self._add_new_connection(
            origin_node,
            new_node,
            random_source,
        )

        return (
            f"connection new-node addition: created node '{new_node.name}' "
            f"from '{node_template.name}' with mutated {mutated_node_field}; "
            f"created connection '{origin_node.name}' -> '{new_node.name}' "
            f"#{new_index} from {source_label}; {mutated_connection_description}"
        )

    def _mutate_fresh_node_connection_addition(
        self,
        random_source: random.Random,
    ) -> str:
        possible_origins = list(self.nodes.values())
        if not possible_origins:
            return "fresh node-connection addition: unchanged; no valid origin node"

        origin_node = random_source.choice(possible_origins)
        new_node = self._random_fresh_node(origin_node, random_source)
        self.nodes[new_node.name] = new_node
        new_index, source_label, connection_description = self._add_new_connection(
            origin_node,
            new_node,
            random_source,
            use_template=False,
        )

        return (
            f"fresh node-connection addition: created randomized node "
            f"{new_node.name!r}; created randomized connection "
            f"{origin_node.name!r} -> {new_node.name!r} #{new_index} "
            f"from {source_label}; {connection_description}"
        )

    def _random_fresh_node(
        self,
        parent_node: NodeGene,
        random_source: random.Random,
    ) -> NodeGene:
        return NodeGene(
            name=self._new_node_name("node"),
            size=self._random_child_size_from_parent(parent_node, random_source),
            joint_type=random_source.choice(
                self._child_joint_types_for_mutation()
            ),
            shape=random_source.choice(BODY_SHAPES),
            recursive_limit=random_source.randint(1, 8),
            orientation=tuple(
                random_source.uniform(-180.0, 180.0) for _ in range(3)
            ),
        )

    def _random_fresh_connection(
        self,
        destination_node: NodeGene,
        random_source: random.Random,
    ) -> ConnectionGene:
        ctrlrange_limit = random_source.uniform(0.5, 2.0)
        parent_face = random_source.choice(ATTACHMENT_FACES)
        return ConnectionGene(
            name=self._new_connection_name("connection"),
            child=destination_node.name,
            axis=self._random_axis(random_source),
            parent_face=parent_face,
            surface_uv=(
                random_source.uniform(-1.0, 1.0),
                random_source.uniform(-1.0, 1.0),
            ),
            child_surface_uv=(
                random_source.uniform(-1.0, 1.0),
                random_source.uniform(-1.0, 1.0),
            ),
            orientation=self._random_valid_connection_orientation(
                parent_face,
                random_source,
            ),
            symmetry=tuple(
                plane
                for plane in SYMMETRY_PLANES
                if random_source.choice((False, True))
            ),
            scale=random_source.uniform(0.5, 1.5),
            terminal_only=random_source.choice((False, True)),
            motor_enabled=random_source.random() < 0.85,
            motor_gear=random_source.uniform(20.0, 200.0),
            ctrlrange=(-ctrlrange_limit, ctrlrange_limit),
            control_amp=random_source.uniform(0.02, 0.50),
            control_freq=random_source.choice(self.HARMONIC_CONTROL_FREQS),
            control_phase=random_source.uniform(-3.141592653589793, 3.141592653589793),
            control_phase_depth_scale=random_source.uniform(-2.0, 2.0),
            control_phase_order_scale=random_source.uniform(-2.0, 2.0),
        )

    def _random_axis(self, random_source: random.Random) -> Tuple[float, float, float]:
        axis = tuple(random_source.uniform(-1.0, 1.0) for _ in range(3))
        if all(abs(component) < 1e-6 for component in axis):
            return (0.0, 1.0, 0.0)
        return axis

    def _random_child_size_from_parent(
        self,
        parent_node: NodeGene,
        random_source: random.Random,
    ) -> Tuple[float, float, float]:
        return tuple(
            max(
                MIN_BODY_SIZE,
                min(
                    dimension
                    * random_source.uniform(
                        self.NEW_CHILD_SIZE_MIN_SCALE,
                        self.NEW_CHILD_SIZE_MAX_SCALE,
                    ),
                    MAX_BODY_SIZE,
                ),
            )
            for dimension in parent_node.size
        )

    def _clamp_child_size_to_parent(
        self,
        child_node: NodeGene,
        parent_node: NodeGene,
    ) -> None:
        if child_node.shape in ROUND_BODY_SHAPES:
            parent_radius_extent = min(parent_node.size[1], parent_node.size[2])
            radius = min(child_node.size[1], child_node.size[2])
            child_node.size = (
                self._clamp_child_dimension(child_node.size[0], parent_node.size[0]),
                self._clamp_child_dimension(radius, parent_radius_extent),
                self._clamp_child_dimension(radius, parent_radius_extent),
            )
            return

        child_node.size = tuple(
            self._clamp_child_dimension(child_dimension, parent_dimension)
            for child_dimension, parent_dimension in zip(
                child_node.size,
                parent_node.size,
            )
        )

    def _clamp_child_dimension(
        self,
        child_dimension: float,
        parent_dimension: float,
    ) -> float:
        parent_relative = max(
            parent_dimension * self.NEW_CHILD_SIZE_MIN_SCALE,
            min(child_dimension, parent_dimension * self.NEW_CHILD_SIZE_MAX_SCALE),
        )
        return max(MIN_BODY_SIZE, min(parent_relative, MAX_BODY_SIZE))

    def _add_new_connection(
        self,
        origin_node: NodeGene,
        destination_node: NodeGene,
        random_source: random.Random,
        use_template: bool = True,
    ) -> Tuple[int, str, str]:
        source_connection = (
            self._random_connection_template(random_source)
            if use_template
            else None
        )

        if source_connection is None:
            if use_template:
                new_connection = ConnectionGene(
                    name=self._new_connection_name("connection"),
                    child=destination_node.name,
                    axis=(0.0, 1.0, 0.0),
                    parent_face="+x",
                    surface_uv=(0.0, 0.0),
                )
                source_label = "default connection coding"
                should_mutate_connection = True
            else:
                new_connection = self._random_fresh_connection(
                    destination_node,
                    random_source,
                )
                source_label = "fresh randomized connection coding"
                should_mutate_connection = False
        else:
            new_connection = copy.deepcopy(source_connection)
            new_connection.name = self._new_connection_name(source_connection.name)
            new_connection.child = destination_node.name
            source_label = self._describe_connection_source(source_connection)
            should_mutate_connection = True

        self.connections[new_connection.name] = new_connection
        origin_node.child_connections.append(new_connection.name)
        self._disable_terminal_only_for_self_link(new_connection, origin_node.name)
        new_index = len(origin_node.child_connections) - 1
        if should_mutate_connection:
            connection_description = self._mutate_new_connection(
                new_connection,
                random_source,
            )
        else:
            connection_description = "randomized all connection fields"

        self._disable_terminal_only_for_self_link(new_connection, origin_node.name)
        self._repair_connection_orientation_if_needed(new_connection, "orientation")
        destination_joint_type = destination_node.joint_type
        if self.control_mode == "neural" and destination_joint_type in NEURAL_INPUT_DIMS:
            self._initialize_connection_neural_parameters(
                new_connection,
                destination_joint_type,
                random_source,
            )
        elif destination_joint_type == "fixed":
            new_connection.neural_w1 = ()
            new_connection.neural_b1 = ()
            new_connection.neural_w2 = ()
            new_connection.neural_b2 = ()
            new_connection.neural_output_axes = ()

        return new_index, source_label, connection_description

    def _mutate_connection_removal(
        self,
        connection: ConnectionGene,
        origin_name: Optional[str] = None,
    ) -> str:
        origin, child_index = self._find_connection_owner(connection, origin_name)
        destination = connection.child

        self.nodes[origin].child_connections.pop(child_index)
        self._archive_connection_if_unused(connection.name)

        return (
            f"connection removal for '{origin}' -> '{destination}' #{child_index}: "
            f"archived connection {connection.name!r} if it became unused"
        )

    def _mutate_connection_origin_node_replacement(
        self,
        connection: ConnectionGene,
        random_source: random.Random,
        origin_name: Optional[str] = None,
    ) -> str:
        old_origin, old_index = self._find_connection_owner(connection, origin_name)
        old_node = self.nodes[old_origin]
        new_node = self._new_node_from_existing(old_node, random_source)
        mutated_field = self._mutate_new_node(new_node, random_source)

        archived_node = self._archive_node_snapshots([old_node])[0]
        self.nodes[old_origin].child_connections.pop(old_index)
        new_node.child_connections = [connection.name]
        self.nodes[new_node.name] = new_node

        return (
            f"connection origin node replacement for '{old_origin}' -> "
            f"'{connection.child}' #{old_index}: {old_origin!r} -> "
            f"{new_node.name!r}; archived node '{archived_node.name}'"
            f" #{len(self.archived_nodes) - 1}; mutated new node {mutated_field}"
        )

    def _mutate_connection_destination_node_replacement(
        self,
        connection: ConnectionGene,
        random_source: random.Random,
    ) -> str:
        origin, child_index = self._find_connection_owner(connection)
        old_destination = connection.child
        old_node = self.nodes[old_destination]
        if old_node.joint_type == "free":
            return (
                f"connection '{origin}' -> '{old_destination}'"
                f" #{child_index}.destination node replacement: unchanged; "
                "free-jointed nodes cannot be connection children"
            )

        new_node = self._new_node_from_existing(old_node, random_source)
        new_node.size = self._random_child_size_from_parent(
            self.nodes[origin],
            random_source,
        )
        mutated_field = self._mutate_new_node(new_node, random_source)
        self._clamp_child_size_to_parent(new_node, self.nodes[origin])

        archived_node = self._archive_node_snapshots([old_node])[0]
        self.nodes[new_node.name] = new_node
        connection.child = new_node.name
        self._reset_connection_neural_parameters(connection, random_source)

        return (
            f"connection destination node replacement for '{origin}' -> "
            f"'{old_destination}' #{child_index}: {old_destination!r} -> "
            f"{new_node.name!r}; archived node '{archived_node.name}'"
            f" #{len(self.archived_nodes) - 1}; mutated new node {mutated_field}"
        )

    def _new_node_from_existing(
        self,
        node: NodeGene,
        random_source: random.Random,
    ) -> NodeGene:
        new_node = copy.deepcopy(node)
        new_node.name = self._new_node_name(node.name)
        return new_node

    def _mutate_new_node(
        self,
        node: NodeGene,
        random_source: random.Random,
    ) -> str:
        mutable_fields = (
            "size",
            "joint_type",
            "shape",
            "recursive_limit",
            "orientation",
        )
        field_name = random_source.choice(mutable_fields)
        base_path = ("new_node", node, field_name)
        value = getattr(node, field_name)

        if isinstance(value, (list, tuple)):
            base_path = (*base_path, random_source.randrange(len(value)))

        owner, field_name, index = self._resolve_new_node_path(base_path)
        mutation_kind, min_mutation_std, keep_ordered_pair = self._mutation_spec(
            field_name,
        )

        if index is None:
            value = getattr(owner, field_name)
            setattr(
                owner,
                field_name,
                self._mutate_value(
                    value,
                    random_source,
                    mutation_kind,
                    min_mutation_std,
                ),
            )
            self._repair_node_size_if_needed(owner, field_name)
            return field_name

        original_values = getattr(owner, field_name)
        values = list(original_values)
        values[index] = self._mutate_value(
            values[index],
            random_source,
            mutation_kind,
            min_mutation_std,
        )

        if keep_ordered_pair:
            values = sorted(values)
            if values[0] == values[1]:
                values[0] -= min_mutation_std
                values[1] += min_mutation_std

        setattr(owner, field_name, type(original_values)(values))
        self._repair_node_size_if_needed(owner, field_name)
        return f"{field_name}[{index}]"

    def _resolve_new_node_path(
        self,
        parameter_path: Tuple[Any, ...],
    ) -> Tuple[NodeGene, str, Optional[int]]:
        _, node, field_name, *maybe_index = parameter_path
        return node, field_name, self._path_index(maybe_index)

    def _mutate_new_connection(
        self,
        connection: ConnectionGene,
        random_source: random.Random,
    ) -> str:
        mutable_fields = self.CONNECTION_MUTABLE_FIELDS
        field_name = random_source.choice(mutable_fields)
        base_path = ("connection", connection, field_name)
        value = getattr(connection, field_name)

        if isinstance(value, (list, tuple)) and field_name != "symmetry":
            base_path = (*base_path, random_source.randrange(len(value)))

        return self._mutate_parameter_path(base_path, random_source)

    def _connection_coding_pool(self) -> List[ConnectionGene]:
        return list(self.connections.values()) + list(self.archived_connections.values())

    def _connection_is_self_pointing(self, connection: ConnectionGene) -> bool:
        return any(
            connection.name in node.child_connections and connection.child == node.name
            for node in self.nodes.values()
        )

    def _disable_terminal_only_for_self_link(
        self,
        connection: ConnectionGene,
        origin_name: str,
    ) -> None:
        if connection.child == origin_name:
            connection.terminal_only = False

    def _random_connection_template(
        self,
        random_source: random.Random,
    ) -> Optional[ConnectionGene]:
        connection_pool = self._connection_coding_pool()
        if not connection_pool:
            return None
        return random_source.choice(connection_pool)

    def _node_coding_pool(self) -> List[NodeGene]:
        return list(self.nodes.values()) + list(self.archived_nodes.values())
