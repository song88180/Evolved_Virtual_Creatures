"""Run output for ``evc3d evolve``: the files written into a run directory.

Layout::

    config.json                         save_run_config
    run_info.txt                        save_run_info
    latest_best_genotype.json           save_generation_checkpoint
    best_genotype.json                  save_generation_checkpoint
    best_metrics.json                   save_generation_checkpoint
    best_creature.xml                   save_best_phenotype
    generation_bests/generation_NNNN.json   save_generation_checkpoint
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evolution import EvaluatedCreature
from .evolution_tasks.shared import TaskDefinition
from .genotype import Genotype
from .genotype_io import save_genotype_to_json
from .graph_analysis import PhenotypeBuildAbort
from .phenotype import PhenotypeBuilder

GENERATION_HISTORY_DIR = "generation_bests"


def save_run_config(run_dir: Path, options: dict[str, Any]) -> None:
    """Write the run's command-line options to ``config.json`` for reproducibility."""
    _write_json(run_dir / "config.json", options)


def save_run_info(run_dir: Path, header: str) -> None:
    """Write the settings summary printed at the start of the run to ``run_info.txt``."""
    (run_dir / "run_info.txt").write_text(header + "\n")


def save_generation_checkpoint(
    run_dir: Path,
    generation: int,
    generation_best: EvaluatedCreature,
    best_so_far: EvaluatedCreature,
    definition: TaskDefinition,
    save_generation_history: bool = True,
) -> None:
    """Persist the generation best and all-time best genotypes, metrics, and MJCF."""
    save_genotype_to_json(generation_best.genotype, run_dir / "latest_best_genotype.json")
    save_genotype_to_json(best_so_far.genotype, run_dir / "best_genotype.json")
    _write_json(run_dir / "best_metrics.json", best_so_far.metrics_dict())
    save_best_phenotype(run_dir / "best_creature.xml", best_so_far.genotype, definition)

    if save_generation_history:
        generation_dir = run_dir / GENERATION_HISTORY_DIR
        generation_dir.mkdir(exist_ok=True)
        save_genotype_to_json(
            generation_best.genotype,
            generation_dir / f"generation_{generation:04d}.json",
        )


def save_best_phenotype(path: Path, genotype: Genotype, definition: TaskDefinition) -> None:
    """Write the creature's MJCF for ``definition``; skip silently if the build aborts."""
    try:
        mjcf = PhenotypeBuilder(genotype, definition).build()
    except PhenotypeBuildAbort:
        return
    path.write_text(mjcf)


def _write_json(path: Path, data: Any) -> None:
    """Write a JSON-serializable object to ``path``, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, default=str)
        f.write("\n")
