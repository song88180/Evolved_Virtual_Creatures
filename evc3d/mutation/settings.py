"""Mutation tuning constants shared by all mutation mixins."""

from __future__ import annotations

from typing import ClassVar, Tuple




class MutationSettings:
    """Class-level mutation constants (standard deviations, harmonics, field lists)."""

    MUTATION_STD_FRACTION: ClassVar[float] = 0.05
    MIN_SIZE_MUTATION_STD: ClassVar[float] = 0.01
    MIN_AXIS_MUTATION_STD: ClassVar[float] = 0.01
    MIN_INTEGER_MUTATION_STD: ClassVar[float] = 1.0
    MIN_SURFACE_COORD_MUTATION_STD: ClassVar[float] = 0.05
    MIN_SCALE_MUTATION_STD: ClassVar[float] = 0.01
    MIN_MOTOR_GEAR_MUTATION_STD: ClassVar[float] = 0.1
    MIN_CTRLRANGE_MUTATION_STD: ClassVar[float] = 0.01
    MIN_CONTROL_AMP_MUTATION_STD: ClassVar[float] = 0.01
    MIN_GLOBAL_CONTROL_FREQ_MUTATION_STD: ClassVar[float] = 0.01
    MIN_PHASE_MUTATION_STD: ClassVar[float] = 0.05
    MIN_NEURAL_WEIGHT_MUTATION_STD: ClassVar[float] = 0.05
    NEURAL_INITIALIZATION_STD: ClassVar[float] = 0.05
    HARMONIC_CONTROL_FREQS: ClassVar[Tuple[float, ...]] = (
        0.5,
        1.0,
        2.0,
        3.0,
        4.0,
    )
    MIN_ORIENTATION_MUTATION_STD: ClassVar[float] = 5.0
    MIN_POSITIVE_VALUE: ClassVar[float] = 1e-6
    DEFAULT_TOPOLOGY_MUTATION_RATE_MIN: ClassVar[float] = 0.05
    TOPOLOGY_MUTATION_PARAMETER_TYPES: ClassVar[frozenset[str]] = frozenset(
        {
            "connection_addition",
            "connection_new_node_addition",
            "fresh_node_connection_addition",
            "connection_origin",
            "connection_destination",
            "connection_type",
            "connection_replacement",
            "connection_origin_node_replacement",
            "connection_destination_node_replacement",
            "connection_removal",
        }
    )
    # Every mutable ConnectionGene field. The order matters: random choices
    # index into it, so reordering would change seeded evolution results.
    # _mutation_spec() must define how each one mutates.
    CONNECTION_MUTABLE_FIELDS: ClassVar[Tuple[str, ...]] = (
        "parent_face",
        "surface_uv",
        "child_surface_uv",
        "orientation",
        "symmetry",
        "axis",
        "scale",
        "terminal_only",
        "motor_enabled",
        "motor_gear",
        "ctrlrange",
        "control_amp",
        "control_freq",
        "control_phase",
        "control_phase_depth_scale",
        "control_phase_order_scale",
    )
    # Connection fields read only by the sine controller; not mutated in neural mode.
    SINE_CONTROL_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "control_amp",
            "control_freq",
            "control_phase",
            "control_phase_depth_scale",
            "control_phase_order_scale",
        }
    )
    NEW_CHILD_SIZE_MIN_SCALE: ClassVar[float] = 0.1
    NEW_CHILD_SIZE_MAX_SCALE: ClassVar[float] = 1.9
