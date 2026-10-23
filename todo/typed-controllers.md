# Typed controllers instead of mode-dependent fields

## Problem

`ConnectionGene` carries both the sine fields (`control_amp`, `control_freq`,
`control_phase`, `control_phase_depth_scale`, `control_phase_order_scale`) and the
neural fields (`neural_w1`, `neural_b1`, `neural_w2`, `neural_b2`,
`neural_output_axes`). `Genotype` additionally carries `global_control_freq`, which
only the sine controller uses. Whichever family is inactive still exists as inert
data. Consequences:

- Invalid states are representable (a neural genotype with live sine fields).
- Mutation can select behaviorally inactive fields (report.md, Var 7:
  `global_control_freq` is always in the mutation pool).
- `convert_control_mode()` (`genotype_io.py`) has to reset fields to inert values
  instead of swapping a controller.
- The saved JSON of every genotype contains fields that do nothing.

## Proposal

Use composition, not subclassing. A connection owns one swappable controller:

```python
@dataclass
class SineController:
    amp: float = 0.0
    freq: float = 10.0          # harmonic multiplier
    phase: float = 0.0
    phase_depth_scale: float = 0.0
    phase_order_scale: float = 0.0

@dataclass
class NeuralController:
    w1: tuple; b1: tuple; w2: tuple; b2: tuple
    output_axes: tuple

@dataclass
class ConnectionGene:
    ...
    controller: SineController | NeuralController
```

- `global_control_freq` moves into the sine side, e.g. a genotype-level
  `SineControllerSettings`, or stays on `Genotype` and is `None` for neural mode.
- `Genotype.control_mode` becomes derived from (or validated against) the
  controller types, and `Genotype` validation enforces that all connections share
  one family.
- `convert_control_mode()` becomes: replace `connection.controller` with a fresh
  inert controller of the target type. No field resetting.

Why composition rather than `NeuralConnectionGene` / `SineConnectionGene`
subclasses: the controller is the only part that varies. Subclassing would need
parallel subclasses to propagate through `Genotype`, node child lookup, archiving,
topology mutation (which creates and copies connections), and JSON loading.

## Scope of change

- `genes.py`: new controller dataclasses, `ConnectionGene.controller`.
- `genotype.py`: validate single controller family; handle `global_control_freq`.
- `genotype_io.py`: nested JSON schema (`"controller": {"type": "sine", ...}`),
  migration from the flat schema, simplify `convert_control_mode`.
- `genotype_mutation.py`: field registry (lines near 227, 890, 1192, 1387, 1736)
  registers only the active controller's fields; fresh-gene generation builds the
  right controller; neural tensor repair operates on `NeuralController`.
- `phenotype.py`: read controller fields through `connection.controller`.
- `control.py`: `ActuatorController` construction from controller dataclasses.
- `examples/*.json`, `README.md`, `docs/`, `CODE_EXPLANATION.md`: schema updates.
- Tests: most tests touching `control_amp` / `neural_*` need updating
  (`test_genotype_mutation.py`, `test_surface_attachment.py`, `test_symmetry.py`,
  `test_evaluation.py`).

## Decisions to make

- Keep reading the flat JSON schema (migration) or require a one-time conversion
  script for existing files and old `runs/` outputs.
- Whether fixed-joint connections have a controller (`None`) or an empty neural one.
- Whether `global_control_freq` stays genotype-level.

## Related

report.md Var 7 (dormant controller parameters mutate), Var 3
(`--control-mode` duplicates genotype state; resolved by `--override-control-mode`).
