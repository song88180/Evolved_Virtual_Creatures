# Project Instructions

When running Python scripts, tests, or checks for this project, always use the conda environment named `mujoco`.

Use:

```bash
conda run -n mujoco --no-capture-output python -m pytest
```


# Project File Map

Use this map to choose files before broad searches:

- `evc3d/genes.py`: core gene dataclasses and low-level validation. Edit for new `NodeGene` or `ConnectionGene` fields, attachment-face rules, symmetry rules, orientation math, or constructor validation.
- `evc3d/genotype.py`: top-level `Genotype` container. Edit when active or archived genetic material fields change.
- `evc3d/genotype_io.py`: JSON schema loading, old-genotype migration, and serialization. Edit when genotype JSON fields, defaults, or backward compatibility behavior changes.
- `evc3d/genotype_mutation.py`: mutation parameter discovery and mutation operators. Edit when adding mutable fields, topology mutations, repair rules, or random fresh gene generation.
- `evc3d/phenotype.py`: genotype-to-MJCF expansion. Edit when body placement, joints, motors, symmetry reflection, body quaternions, geometry, collision settings, or task-specific XML output changes.
- `evc3d/graph_analysis.py`: graph validity and phenotype node-count analysis. Edit for recursion, terminal-only, missing-node, or max-node validation rules.
- `evc3d/evaluation.py`: MuJoCo rollout metrics and task fitness functions. Edit for swimming, walking, flying, collision disqualification, floor clearance, or volume penalties.
- `evc3d/evolve.py`: reusable population/evolution helpers. Edit for selection, population initialization, generation outputs, or mutation scheduling.
- `evc3d/scene.py`: shared display setup for viewer and video (build for a task, task initialization, lighting, floor, one controlled step with failure checks). Edit for anything both display modes should share.
- `evc3d/video.py`: MP4 recording loop (`evc3d view --mp4`). Edit for frame capture, video camera motion, or shadow tweaks.
- `evc3d/viewer.py`: interactive MuJoCo window loop (`evc3d view`). Edit for live stepping, pacing, or viewer camera.
- `evc3d/cli/__init__.py`: the `evc3d` command entry point (`evc3d <command>`, also `python -m evc3d`); subcommand registration, parsing, and usage-error handling.
- `evc3d/cli/args.py`: option groups shared by subcommands (genotype, environment, scoring, mutation) and their validation. Edit when adding a task, environment, scoring, or mutation option.
- `evc3d/cli/evolve.py`: `evc3d evolve`, population-based evolution and run artifacts.
- `evc3d/cli/evaluate.py`: `evc3d evaluate`, scoring one genotype and printing metrics.
- `evc3d/cli/mutate.py`: `evc3d mutate`, writing a randomly mutated copy of a genotype.
- `evc3d/cli/build.py`: `evc3d build`, writing the MuJoCo XML scene for a genotype.
- `evc3d/cli/view.py`: `evc3d view`, watching a genotype in an interactive window or recording it with `--mp4`.
- `pyproject.toml`: package metadata and the `evc3d` console script (`pip install -e .`).
- `examples/*.json`: seed genotype recipes used by CLIs and tests.
- `tests/test_surface_attachment.py`: tests for JSON migration, attachment placement, joint/body XML, symmetry attachment, and orientation constraints.
- `tests/test_genotype_mutation.py`: tests for mutation registration, field mutation effects, topology mutation behavior, and mutation invariants.
- `tests/test_symmetry.py`: tests for symmetry validation, JSON round trips, mirrored child generation, and reflected joint axes.
- `tests/test_evaluation.py`: tests for task physics, MuJoCo compilation, collision settings, volume checks, and evaluation metrics.
- `tests/test_evolve.py`: tests for `evc3d` CLI parsing and subcommands, evolution task config forwarding, population helpers, and run artifact behavior.
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
