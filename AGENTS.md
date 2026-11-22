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
- `evc3d/mutation/`: genotype mutation, one mixin per subsystem, combined into `GenotypeMutationMixin` (inherited by `Genotype`; `genotype.mutation()` is the entry point). `selection.py` chooses what to mutate; `values.py` mutates one field (per-field specs, repairs); `neural.py` neural weights; `brain.py` graph brain mutation (weights, sources, functions, neuron growth, seeding, garbage collection); `topology.py` connection/node operators and fresh genes; `gene_pool.py` naming, archiving, pruning; `settings.py` constants, including `CONNECTION_MUTABLE_FIELDS` (the single ordered list of mutable connection fields; keep its order, random choices index into it).
- `evc3d/phenotype.py`: genotype-to-MJCF expansion. Edit when body placement, joints, motors, symmetry reflection, body quaternions, geometry, collision settings, or task-specific XML output changes.
- `evc3d/brain_functions.py`: the graph brain's 23 neuron functions (`evaluate_neuron`) and their state. Edit to add or change a neuron function.
- `evc3d/brain_genes.py`: graph brain genes (`BrainNeuron`, `BrainInput`, `BrainEffector`, `BrainGene`) and source-string parsing. Edit for source syntax or brain gene fields.
- `evc3d/brain_compile.py`: flattens a genotype's nested brain over the generated body parts into a `BrainSpec`. Edit for how sources resolve across parts.
- `evc3d/controller_spec.py`: MuJoCo-free controller sizes, the `ActuatorController` record, neural-weight helpers, and the compiled `BrainSpec`. Genotype modules import this, never `control.py`, so genotype loading does not need MuJoCo.
- `evc3d/control.py`: applies controllers to a running simulation (`apply_controller`, actuator ID lookup, `BrainRuntime` for graph mode). Edit for per-step sine, neural, or graph control and brain sensors.
- `evc3d/graph_analysis.py`: graph validity and generated body-count analysis. Edit for recursion, terminal-only, missing-node, or max-body-count validation rules.
- `evc3d/evaluation.py`: scoring (`evaluate_task`): scoring rules (volume limits, start rules, collision disqualification) and the metric rollout (`_run_episode`, with `_FloorContactTracker` and `_UprightTracker`). The engine reports every metric it can measure and holds no task-specific scoring terms; those live in the task modules. Edit for metrics or scoring rules.
- `evc3d/simulation.py`: mechanics shared by scoring and viewing (`start_episode` placement and settling, state copy, floor/height geometry, instability and self-collision checks). Edit for anything evaluation, the viewer, and MP4 recording must do identically.
- `evc3d/evolution_tasks/__init__.py`: task registry (`task_definition`, `task_names`, `load_task_definition`); task modules live alongside. Each task is one module, `evolution_tasks/<task>.py`: its config fields, environment, and fitness function. Helper modules (listed in `NON_TASK_MODULES`, never loaded as tasks): `config.py` (`EnvironmentFamily`, `EvaluationConfig`, CLI option-name tuples, default constants), `shared.py` (`TaskDefinition`, `RolloutPolicy`, result bases, `make_result`, `failed_result`, registry state), and `families.py` (environments, config and result bases, and flying score terms shared by the `_x`/`_away` task pairs).
- `evc3d/evolution.py`: population helpers (evaluate a population, elite/tournament selection, mutation of children, generation summaries). Edit for selection, population initialization, or mutation scheduling.
- `evc3d/run_artifacts.py`: files written into an evolve run directory (config, best genotypes, metrics, best XML, generation checkpoints). Edit for run output layout.
- `evc3d/scene.py`: shared display setup for viewer and video (build for a task, placement and settling without scoring rules, lighting, floor, one controlled step with failure checks). Edit for anything both display modes should share.
- `evc3d/video.py`: MP4 recording loop (`evc3d view --mp4`). Edit for frame capture, video camera motion, or shadow tweaks.
- `evc3d/viewer.py`: interactive MuJoCo window loop (`evc3d view`). Edit for live stepping, pacing, or viewer camera.
- `evc3d/cli/__init__.py`: the `evc3d` command entry point (`evc3d <command>`, also `python -m evc3d`); subcommand registration, parsing, and usage-error handling.
- `evc3d/cli/args.py`: option groups shared by subcommands (genotype, environment, scoring, mutation). Environment and scoring values are validated by `EnvironmentFamily` and `EvaluationConfig` themselves (`evolution_tasks/config.py`); `validate_task_options` rewords their errors with option names. When adding a task option, add it here, name the option after its config field and add the field to `CONFIG_FIELD_OPTIONS` (or `ENVIRONMENT_FIELD_OPTIONS`) in `config.py`, and put any value rule in the config's `__post_init__`.
- `evc3d/cli/evolve.py`: `evc3d evolve`, population-based evolution and run artifacts.
- `evc3d/cli/run_header.py`: the settings summary `evc3d evolve` prints at start and saves as `run_info.txt`.
- `evc3d/cli/evaluate.py`: `evc3d evaluate`, scoring one genotype and printing metrics.
- `evc3d/cli/mutate.py`: `evc3d mutate`, writing a randomly mutated copy of a genotype.
- `evc3d/cli/build.py`: `evc3d build`, writing the MuJoCo XML scene for a genotype.
- `evc3d/cli/view.py`: `evc3d view`, watching a genotype in an interactive window or recording it with `--mp4`.
- `pyproject.toml`: package metadata and the `evc3d` console script (`pip install -e .`).
- `examples/*.json`: seed genotype recipes used by CLIs and tests.
- Tests mirror the package: `tests/test_<module>.py` tests `evc3d/<module>.py`. `test_cli.py` covers all `evc3d` subcommands' parsing and behaviour; `test_view.py` covers `evc3d view`, `viewer.py`, and `scene.py`; `test_evaluation.py` covers `evaluation.py` and the task modules; `test_genotype_mutation.py` covers `evc3d/mutation/`; `test_symmetry.py` covers symmetry across genes, I/O, and phenotype; `test_brain.py` covers the graph brain (functions, genes, compile, runtime, mutation, I/O).
- `tests/helpers.py`: shared test helpers (`tree_node`/`tree_genotype` for inline genotypes, `make_definition` for private task-definition copies, `make_creature`, `placed_walker`). Never modify `task_definition(...)` results in tests; they are shared registry objects.
- `tests/test_api_reference.py`: checks `docs/api-reference.json` against the code (names exist, parameters and defaults match, public members documented).
- `docs/`: static GitHub Pages documentation site. Update `docs/index.html` for user-facing concepts and `docs/api-reference.json` for the API reference. The API reference is hand-written but checked by `tests/test_api_reference.py`: after changing a public signature, default, class, or module, update its entry until that test passes.
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
