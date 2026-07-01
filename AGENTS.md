# Project Instructions

When running Python scripts, tests, or checks for this project, always use the conda environment named `mujoco`.

Use:

```bash
conda run -n mujoco --no-capture-output python -m pytest
```


# Project File Map

Use this map to choose files before broad searches:

- `evol_virtual_creature/genes.py`: core gene dataclasses and low-level validation. Edit for new `NodeGene` or `ConnectionGene` fields, attachment-face rules, symmetry rules, orientation math, or constructor validation.
- `evol_virtual_creature/genotype.py`: top-level `Genotype` container. Edit when active or archived genetic material fields change.
- `evol_virtual_creature/genotype_io.py`: JSON schema loading, old-genotype migration, and serialization. Edit when genotype JSON fields, defaults, or backward compatibility behavior changes.
- `evol_virtual_creature/genotype_mutation.py`: mutation parameter discovery and mutation operators. Edit when adding mutable fields, topology mutations, repair rules, or random fresh gene generation.
- `evol_virtual_creature/phenotype.py`: genotype-to-MJCF expansion. Edit when body placement, joints, motors, symmetry reflection, body quaternions, geometry, collision settings, or task-specific XML output changes.
- `evol_virtual_creature/graph_analysis.py`: graph validity and phenotype node-count analysis. Edit for recursion, terminal-only, missing-node, or max-node validation rules.
- `evol_virtual_creature/evaluation.py`: MuJoCo rollout metrics and task fitness functions. Edit for swimming, walking, flying, collision disqualification, floor clearance, or volume penalties.
- `evol_virtual_creature/evolve.py`: reusable population/evolution helpers. Edit for selection, population initialization, generation outputs, or mutation scheduling.
- `evol_virtual_creature/video.py`: video rendering helpers used by evaluation output. Edit for camera, lighting, floor rendering, or tracked-body video behavior.
- `evol_virtual_creature/viewer.py`: interactive MuJoCo viewer loop. Edit for manual visualization and live controller stepping.
- `generate_model.py`: CLI for loading, mutating, building, writing, and viewing one creature.
- `evaluate.py`: CLI for scoring one genotype and optionally rendering video.
- `evolve.py`: CLI for running population-based evolution and writing run artifacts.
- `examples/*.json`: seed genotype recipes used by CLIs and tests.
- `tests/test_surface_attachment.py`: tests for JSON migration, attachment placement, joint/body XML, symmetry attachment, and orientation constraints.
- `tests/test_genotype_mutation.py`: tests for mutation registration, field mutation effects, topology mutation behavior, and mutation invariants.
- `tests/test_symmetry.py`: tests for symmetry validation, JSON round trips, mirrored child generation, and reflected joint axes.
- `tests/test_evaluation.py`: tests for task physics, MuJoCo compilation, collision settings, volume checks, and evaluation metrics.
- `tests/test_evolve.py`: tests for evolution CLI parsing, task config forwarding, population helpers, and run artifact behavior.
- `docs/`: static GitHub Pages documentation site. Update `docs/index.html` for user-facing concepts and `docs/api-reference.json` for generated API reference content.
- `CODE_EXPLANATION.md`: narrative architecture walkthrough for humans and agents.

# Codex Git Workflow

When modifying tracked files in this project:

1. Before editing, inspect `git status` and avoid overwriting user changes.
2. After editing, do not commit automatically.
3. At the end of the response, include:
   - Proposed commit message for `git commit -m`.
   - Ask whether to (1) commit, (2) keep changes uncommitted, (3) commit to a new branch, or (4) revert Codex's changes.
4. If the user approves the commit:
   - Run `git add` only for files changed for this task.
   - Run `git commit -m "<proposed message>"`.
5. If the user says to keep changes without committing:
   - Leave the working tree as-is.
6. If the user rejects the changes:
   - Revert only Codex's changes from this task.
   - Do not revert unrelated user changes.
7. Never push unless the user explicitly asks.
