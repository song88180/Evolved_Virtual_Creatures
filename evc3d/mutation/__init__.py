"""Genotype mutation, split by subsystem.

``GenotypeMutationMixin`` combines the mixins below; ``Genotype`` inherits it,
so ``genotype.mutation(...)`` is the single public entry point.

* ``selection``: choose what to mutate (``mutation()``, parameter pool, rates)
* ``values``: mutate one field value (specs, numeric/categorical, repairs)
* ``neural``: neural controller weights
* ``brain``: graph brain (neurons, sources, seeding, garbage collection)
* ``topology``: add/remove/relink connections and nodes, fresh genes
* ``gene_pool``: naming, archiving, reactivation, pruning
* ``settings``: shared constants
"""

from __future__ import annotations

from .brain import BrainMutationMixin
from .gene_pool import GenePoolMixin
from .neural import NeuralMutationMixin
from .selection import MutationSelectionMixin
from .settings import MutationSettings
from .topology import TopologyMutationMixin
from .values import ValueMutationMixin


class GenotypeMutationMixin(
    MutationSelectionMixin,
    ValueMutationMixin,
    NeuralMutationMixin,
    BrainMutationMixin,
    TopologyMutationMixin,
    GenePoolMixin,
):
    """All genotype mutation behaviour, inherited by ``Genotype``."""


__all__ = ["GenotypeMutationMixin", "MutationSettings"]
