"""Compile a genotype's nested brain into a flat ``BrainSpec`` (MuJoCo-free).

``PhenotypeBuilder`` records one ``BrainInstance`` per generated body part, in
build order. ``compile_brain`` gives each instance its own copy of its part
gene's neurons, resolves every source string (see ``brain_genes``) to a signal
slot, and returns the flat description that ``control.BrainRuntime`` runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .brain_genes import (
    SENSOR_ANGLE,
    SENSOR_CONTACT,
    BrainEffector,
    BrainInput,
    joint_dofs,
    parse_source,
)
from .controller_spec import (
    BrainSpec,
    EffectorSpec,
    NeuronSpec,
    SensorSpec,
)
from .genotype import Genotype

GLOBAL = -1  # pseudo instance index of the global neuron set


@dataclass
class BrainInstance:
    """One generated body part, as the brain compiler needs to see it."""

    node_name: str
    body_name: str
    connection_name: Optional[str]
    parent: Optional[int]
    children: Dict[str, List[int]] = field(default_factory=dict)
    joint_name: str = ""
    joint_type: str = ""
    motor_names: Tuple[str, ...] = ()
    ctrlrange: Tuple[float, float] = (-1.0, 1.0)


def compile_brain(
    genotype: Genotype,
    instances: List[BrainInstance],
) -> BrainSpec:
    """Flatten the genotype's brains over ``instances`` into a ``BrainSpec``."""
    return _Compiler(genotype, instances).compile()


class _Compiler:
    def __init__(self, genotype: Genotype, instances: List[BrainInstance]):
        self.genotype = genotype
        self.instances = instances
        self.first_by_node: Dict[str, int] = {}
        for index, instance in enumerate(instances):
            self.first_by_node.setdefault(instance.node_name, index)
        self.neuron_slots: Dict[Tuple[int, str], int] = {}
        self.sensor_slots: Dict[Tuple, int] = {}
        self.sensors: List[SensorSpec] = []

    def compile(self) -> BrainSpec:
        owners: List[Tuple[int, object]] = [
            (GLOBAL, neuron) for neuron in self.genotype.global_neurons
        ]
        for index, instance in enumerate(self.instances):
            brain = self.genotype.nodes[instance.node_name].brain
            owners.extend((index, neuron) for neuron in brain.neurons)
        for slot, (owner, neuron) in enumerate(owners):
            self.neuron_slots[(owner, neuron.name)] = slot
        self.neuron_count = len(owners)

        neuron_specs = [
            NeuronSpec(
                function=neuron.function,
                inputs=tuple(
                    self._input(owner, item) for item in neuron.inputs
                ),
            )
            for owner, neuron in owners
        ]

        effector_specs: List[EffectorSpec] = []
        for index, instance in enumerate(self.instances):
            brain = self.genotype.nodes[instance.node_name].brain
            for dof, motor_name in enumerate(instance.motor_names):
                gene = (
                    brain.effectors[dof]
                    if dof < len(brain.effectors)
                    else BrainEffector()
                )
                slot, weight = self._input(index, gene)
                effector_specs.append(
                    EffectorSpec(motor_name, slot, weight, instance.ctrlrange)
                )
        return BrainSpec(
            neurons=tuple(neuron_specs),
            sensors=tuple(self.sensors),
            effectors=tuple(effector_specs),
        )

    def _input(self, owner: int, item: BrainInput | BrainEffector) -> Tuple[int, float]:
        if item.source is None:
            return (-1, item.weight)
        slot = self._resolve(owner, item.source)
        if slot is None:
            return (-1, 0.0)
        return (slot, item.weight)

    def _target(self, owner: int, scope: str, scope_arg: Optional[str]) -> Optional[int]:
        """Instance index (or ``GLOBAL``) a scope refers to, if any."""
        if scope == "global":
            return GLOBAL
        if scope == "root":
            return 0 if self.instances else None
        if scope == "node":
            return self.first_by_node.get(scope_arg or "")
        if owner == GLOBAL:
            return None  # self/parent/child are meaningless for global neurons
        if scope == "self":
            return owner
        if scope == "parent":
            return self.instances[owner].parent
        children = self.instances[owner].children.get(scope_arg or "")
        return children[0] if children else None

    def _resolve(self, owner: int, source: str) -> Optional[int]:
        parsed = parse_source(source)
        target = self._target(owner, parsed.scope, parsed.scope_arg)
        if target is None:
            return None
        if parsed.kind == "neuron":
            return self.neuron_slots.get((target, parsed.item))
        if target == GLOBAL:
            return None  # the global set has no body, so no sensors
        instance = self.instances[target]
        if parsed.kind == SENSOR_ANGLE:
            index = int(parsed.item)
            if not instance.joint_name or index >= joint_dofs(instance.joint_type):
                return None
            key = (SENSOR_ANGLE, instance.joint_name, index)
            spec = SensorSpec(
                SENSOR_ANGLE,
                joint_name=instance.joint_name,
                joint_type=instance.joint_type,
                index=index,
            )
        else:
            key = (SENSOR_CONTACT, instance.body_name, parsed.item)
            spec = SensorSpec(
                SENSOR_CONTACT, body_name=instance.body_name, face=parsed.item
            )
        if key not in self.sensor_slots:
            self.sensor_slots[key] = self.neuron_count + len(self.sensors)
            self.sensors.append(spec)
        return self.sensor_slots[key]
