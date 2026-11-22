"""Graph brain mutation: weights, input sources, neuron functions, neuron growth, seeding, and garbage collection.

This follows Sims' procedure for directed graphs, adapted to this project's
parameter-pool mutation: parameters of nodes (neuron functions, weights) and of
connections (input sources) are mutated, a new random neuron can be added, and
neurons no effector depends on are garbage collected. Neural graphs have a
fixed number of inputs per neuron, so inputs are never added or removed.

Brain mutations run after all body mutations in a batch (``mutation()`` sorts
them last), because valid input sources depend on the body's topology.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from ..brain_functions import NEURON_ARITY, NEURON_FUNCTIONS
from ..brain_genes import (
    CONTACT_FACES,
    BrainEffector,
    BrainGene,
    BrainInput,
    BrainNeuron,
    format_source,
    joint_dofs,
    next_neuron_name,
    parse_source,
)
from ..genes import NodeGene
from .settings import MutationSettings

Context = Tuple[str, ...]  # ("global",) or ("node", node_name)
GLOBAL_CONTEXT: Context = ("global",)


class BrainMutationMixin(MutationSettings):
    """Graph brain part of ``GenotypeMutationMixin``."""

    BRAIN_MUTATION_PREFIX = "brain_"

    # ------------------------------------------------------------------
    # Parameter pool
    # ------------------------------------------------------------------

    def _brain_parameter_paths(self) -> List[Tuple[Any, ...]]:
        """Every brain parameter that can mutate (assumes brains are seeded)."""
        paths: List[Tuple[Any, ...]] = [("brain_neuron_addition", GLOBAL_CONTEXT)]
        for neuron in self.global_neurons:
            paths.extend(self._neuron_paths(GLOBAL_CONTEXT, neuron))
        for node_name, node in self.nodes.items():
            context: Context = ("node", node_name)
            if joint_dofs(node.joint_type) == 0 and not node.brain.neurons:
                continue
            paths.append(("brain_neuron_addition", context))
            for neuron in node.brain.neurons:
                paths.extend(self._neuron_paths(context, neuron))
            for index in range(len(node.brain.effectors)):
                paths.append(("brain_effector_weight", node_name, index))
                paths.append(("brain_effector_source", node_name, index))
        return paths

    def _neuron_paths(
        self,
        context: Context,
        neuron: BrainNeuron,
    ) -> List[Tuple[Any, ...]]:
        paths: List[Tuple[Any, ...]] = [("brain_function", context, neuron.name)]
        for index in range(NEURON_ARITY[neuron.function]):
            paths.append(("brain_input_weight", context, neuron.name, index))
            paths.append(("brain_input_source", context, neuron.name, index))
        return paths

    # ------------------------------------------------------------------
    # Seeding and repair
    # ------------------------------------------------------------------

    def _seed_graph_brains(self, random_source: random.Random) -> None:
        """Make every active articulated part's brain match its joint.

        A part whose joint has ``d`` degrees of freedom needs ``d`` effectors.
        Parts with no controls yet get a random oscillator driving them (so a
        fresh creature moves); effector lists are padded or truncated when a
        joint type changes. Existing neurons and weights are kept.
        """
        for node in self.nodes.values():
            dofs = joint_dofs(node.joint_type)
            brain = node.brain
            if dofs == 0:
                brain.effectors = []
                continue
            if not brain.neurons and not any(
                effector.source is not None for effector in brain.effectors
            ):
                brain.neurons.append(self._random_oscillator(brain, random_source))
                brain.effectors = []
            source = (
                format_source("self", brain.neurons[0].name)
                if brain.neurons
                else None
            )
            del brain.effectors[dofs:]
            while len(brain.effectors) < dofs:
                brain.effectors.append(
                    BrainEffector(source, random_source.uniform(-1.0, 1.0))
                )

    def _random_oscillator(
        self,
        brain: BrainGene,
        random_source: random.Random,
    ) -> BrainNeuron:
        return BrainNeuron(
            name=brain.new_neuron_name(),
            function="oscillate-wave",
            inputs=[
                BrainInput(None, random_source.uniform(*self.BRAIN_OSCILLATOR_FREQ_RANGE)),
                BrainInput(None, random_source.uniform(-math.pi, math.pi)),
                BrainInput(None, 0.0),
            ],
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _mutate_brain_parameter(
        self,
        parameter_path: Tuple[Any, ...],
        random_source: random.Random,
    ) -> str:
        kind = parameter_path[0]
        if kind == "brain_neuron_addition":
            return self._mutate_brain_neuron_addition(parameter_path[1], random_source)
        if kind in ("brain_effector_weight", "brain_effector_source"):
            _, node_name, index = parameter_path
            node = self.nodes.get(node_name)
            if node is None or index >= len(node.brain.effectors):
                return f"{node_name}.effector[{index}]: unchanged; stale brain path"
            effector = node.brain.effectors[index]
            label = f"{node_name}.effector[{index}]"
            if kind == "brain_effector_weight":
                return self._mutate_weight(effector, label, random_source)
            context: Context = ("node", node_name)
            return self._mutate_source(effector, label, context, random_source)

        _, context, neuron_name, *rest = parameter_path
        neuron = self._find_neuron(context, neuron_name)
        label = f"{self._context_label(context)}.{neuron_name}"
        if neuron is None:
            return f"{label}: unchanged; stale brain path"
        if kind == "brain_function":
            return self._mutate_function(neuron, label, random_source)
        item = neuron.inputs[rest[0]]
        label = f"{label}.input[{rest[0]}]"
        if kind == "brain_input_weight":
            return self._mutate_weight(item, label, random_source)
        return self._mutate_source(item, label, context, random_source)

    @staticmethod
    def _context_label(context: Context) -> str:
        return "global" if context[0] == "global" else context[1]

    def _find_neuron(self, context: Context, name: str) -> Optional[BrainNeuron]:
        if context[0] == "global":
            return next((n for n in self.global_neurons if n.name == name), None)
        node = self.nodes.get(context[1])
        return node.brain.neuron_named(name) if node is not None else None

    def _neuron_list(self, context: Context) -> Optional[List[BrainNeuron]]:
        if context[0] == "global":
            return self.global_neurons
        node = self.nodes.get(context[1])
        return node.brain.neurons if node is not None else None

    # ------------------------------------------------------------------
    # Single-parameter mutations
    # ------------------------------------------------------------------

    def _mutate_weight(
        self,
        target: BrainInput | BrainEffector,
        label: str,
        random_source: random.Random,
    ) -> str:
        old = target.weight
        if random_source.random() < self.BRAIN_NEGATE_PROBABILITY:
            new = -old if old != 0.0 else random_source.gauss(0.0, 1.0)
        else:
            new = old + self._normal_mutation_delta(
                old, self.MIN_BRAIN_WEIGHT_MUTATION_STD, random_source
            )
        target.weight = new
        return f"{label}.weight: {old!r} -> {new!r}"

    def _mutate_function(
        self,
        neuron: BrainNeuron,
        label: str,
        random_source: random.Random,
    ) -> str:
        old = neuron.function
        neuron.function = random_source.choice(
            [name for name in NEURON_FUNCTIONS if name != old]
        )
        return f"{label}.function: {old!r} -> {neuron.function!r}"

    def _mutate_source(
        self,
        target: BrainInput | BrainEffector,
        label: str,
        context: Context,
        random_source: random.Random,
    ) -> str:
        old = target.source
        target.source = self._random_source(
            context, random_source, allow_constant=True
        )
        if target.source is None and old is not None:
            target.weight = random_source.uniform(-1.0, 1.0)
        return f"{label}.source: {old!r} -> {target.source!r}"

    def _mutate_brain_neuron_addition(
        self,
        context: Context,
        random_source: random.Random,
    ) -> str:
        """Add a random neuron and wire one existing consumer to it.

        Sims adds a node that is garbage collected unless some connection
        mutates to point at it. Here the new neuron is wired in at once so
        growth is not lost to garbage collection before a later mutation could
        reach it.
        """
        neurons = self._neuron_list(context)
        if neurons is None:
            return f"{self._context_label(context)}: unchanged; stale brain path"
        consumers = self._consumers_in_context(context)
        function = random_source.choice(NEURON_FUNCTIONS)
        neuron = BrainNeuron(name=next_neuron_name(neurons), function=function)
        for index in range(NEURON_ARITY[function]):
            neuron.inputs[index] = self._random_input(
                function, index, context, random_source
            )
        neurons.append(neuron)

        label = f"{self._context_label(context)}.{neuron.name}"
        if not consumers:
            return f"added neuron {label} ({function}); no consumer to wire it to"
        consumer, description = random_source.choice(consumers)
        consumer.source = format_source(
            "self" if context[0] != "global" else "global", neuron.name
        )
        return (
            f"added neuron {label} ({function}); rewired "
            f"{description} to read it"
        )

    def _consumers_in_context(
        self,
        context: Context,
    ) -> List[Tuple[BrainInput | BrainEffector, str]]:
        label = self._context_label(context)
        consumers: List[Tuple[BrainInput | BrainEffector, str]] = []
        neurons = self._neuron_list(context) or []
        for neuron in neurons:
            for index, item in enumerate(neuron.used_inputs):
                consumers.append((item, f"{label}.{neuron.name}.input[{index}]"))
        if context[0] == "node":
            for index, effector in enumerate(self.nodes[context[1]].brain.effectors):
                consumers.append((effector, f"{label}.effector[{index}]"))
        return consumers

    def _random_input(
        self,
        function: str,
        index: int,
        context: Context,
        random_source: random.Random,
    ) -> BrainInput:
        if function.startswith("oscillate") and index == 0:
            return BrainInput(
                None, random_source.uniform(*self.BRAIN_OSCILLATOR_FREQ_RANGE)
            )
        source = self._random_source(context, random_source, allow_constant=True)
        if source is None:
            return BrainInput(None, random_source.uniform(-1.0, 1.0))
        return BrainInput(source, random_source.uniform(-1.0, 1.0))

    # ------------------------------------------------------------------
    # Valid sources
    # ------------------------------------------------------------------

    def _random_source(
        self,
        context: Context,
        random_source: random.Random,
        allow_constant: bool,
    ) -> Optional[str]:
        if allow_constant and random_source.random() < self.BRAIN_CONSTANT_PROBABILITY:
            return None
        candidates = self._source_candidates(context)
        if not candidates:
            return None
        return random_source.choice(candidates)

    def _source_candidates(self, context: Context) -> List[str]:
        """Source strings that currently resolve for an input owned by ``context``."""
        candidates: List[str] = []
        if context[0] == "global":
            candidates.extend(
                format_source("global", neuron.name) for neuron in self.global_neurons
            )
            for node_name, node in self.nodes.items():
                candidates.extend(
                    self._part_sources("node", node_name, node, node_name)
                )
            return candidates

        node_name = context[1]
        node = self.nodes.get(node_name)
        if node is None:
            return candidates
        candidates.extend(self._part_sources("self", None, node, node_name))
        parent_sources: Set[str] = set()
        for parent in self.nodes.values():
            for connection in self.outgoing_connections(parent.name):
                if connection.child == node_name:
                    parent_sources.update(
                        self._part_sources("parent", None, parent, parent.name)
                    )
        candidates.extend(sorted(parent_sources))
        for connection in self.outgoing_connections(node_name):
            child = self.nodes.get(connection.child)
            if child is not None:
                candidates.extend(
                    self._part_sources("child", connection.name, child, child.name)
                )
        candidates.extend(
            format_source("global", neuron.name) for neuron in self.global_neurons
        )
        return candidates

    def _part_sources(
        self,
        scope: str,
        scope_arg: Optional[str],
        node: NodeGene,
        _label: str,
    ) -> List[str]:
        sources = [
            format_source(scope, neuron.name, scope_arg) for neuron in node.brain.neurons
        ]
        sources.extend(
            format_source(scope, f"angle{index}", scope_arg)
            for index in range(joint_dofs(node.joint_type))
        )
        sources.extend(
            format_source(scope, f"contact{face}", scope_arg) for face in CONTACT_FACES
        )
        return sources

    # ------------------------------------------------------------------
    # Garbage collection
    # ------------------------------------------------------------------

    def _collect_brain_garbage(self) -> str:
        """Remove neurons that no effector depends on (Sims' step 5).

        Connectedness propagates backwards from every effector through neuron
        inputs. A source is followed to every part it could refer to (a
        ``parent`` source to every part that has this part as a child, and so
        on), so a neuron is only removed when nothing could be reading it.
        """
        parents: Dict[str, Set[str]] = {name: set() for name in self.nodes}
        for node in self.nodes.values():
            for connection in self.outgoing_connections(node.name):
                if connection.child in parents:
                    parents[connection.child].add(node.name)

        reachable: Set[Tuple[str, str]] = set()  # (owner, neuron name)
        pending: List[Tuple[str, str]] = []

        def follow(owner: str, source: Optional[str]) -> None:
            if source is None:
                return
            parsed = parse_source(source)
            if parsed.kind != "neuron":
                return
            if parsed.scope == "global":
                owners = {"global"}
            elif parsed.scope == "root":
                owners = {self.root}
            elif parsed.scope == "node":
                owners = {parsed.scope_arg or ""}
            elif owner == "global":
                owners = set()
            elif parsed.scope == "self":
                owners = {owner}
            elif parsed.scope == "parent":
                owners = parents.get(owner, set())
            else:
                owners = {
                    connection.child
                    for connection in self.outgoing_connections(owner)
                    if connection.name == parsed.scope_arg
                }
            for target in owners:
                key = (target, parsed.item)
                if key not in reachable:
                    reachable.add(key)
                    pending.append(key)

        for node_name, node in self.nodes.items():
            for effector in node.brain.effectors:
                follow(node_name, effector.source)
        while pending:
            owner, name = pending.pop()
            neuron = self._find_neuron(
                GLOBAL_CONTEXT if owner == "global" else ("node", owner), name
            )
            if neuron is None:
                continue
            for item in neuron.used_inputs:
                follow(owner, item.source)

        removed: List[str] = []
        for node_name, node in self.nodes.items():
            kept = [n for n in node.brain.neurons if (node_name, n.name) in reachable]
            removed.extend(
                f"{node_name}.{n.name}"
                for n in node.brain.neurons
                if (node_name, n.name) not in reachable
            )
            node.brain.neurons = kept
        kept_global = [n for n in self.global_neurons if ("global", n.name) in reachable]
        removed.extend(
            f"global.{n.name}"
            for n in self.global_neurons
            if ("global", n.name) not in reachable
        )
        self.global_neurons = kept_global
        if not removed:
            return ""
        return "garbage collected unconnected neuron(s): " + ", ".join(removed)
