# Speed up the per-step neural controller

## Problem

In neural mode, `control.apply_controller()` runs in Python once per simulation
step and costs far more than the MuJoCo physics step itself. Measured on
2026-11-17 (`mujoco` conda env, single process, after `mj_forward`):

| Creature | Motors | `apply_controller` / step | `mj_step` / step |
|---|---|---|---|
| `examples/example_genotype.json` | 21 | ~513 µs | ~14 µs |
| `examples/example_genotype_neural.json` | 12 | ~123 µs | ~12 µs |

So evaluation time is dominated by the controller, not physics (roughly
10-35x). Breakdown for `example_genotype.json` (21 hinge controllers):

| Part of `apply_controller` | µs / step | Share |
|---|---|---|
| `np.asarray(...)` on `neural_w1/b1/w2/b2` tuples, every step | ~203 | ~40% |
| `_neural_observation()` (joint `mj_name2id`, slicing, `np.concatenate` of 6 small arrays) | ~141 | ~27% |
| MLP math itself (two matmuls + `tanh`), per joint | ~86 | ~17% |
| Actuator `mj_name2id` lookups (`_actuator_ids_for_controller`) | ~17 | ~3% |
| Remainder (`_scale_unit_output` per motor, loop overhead) | ~65 | ~13% |

None of the per-step work above depends on simulation state except the
observation values and the MLP evaluation; everything else is constant for a
compiled model.

## Proposal

Compile the controllers once per `MjModel`, then run a cheap per-step update.

1. **Pre-convert weights once.** Turn the tuple weights into `np.ndarray`
   when the episode starts (or when `ActuatorController` is built), not on
   every step. Expected: removes ~40% of controller time on its own.
2. **Resolve names once.** Look up joint IDs, `qposadr`/`dofadr`, and
   actuator IDs once per compiled model instead of calling `mj_name2id` every
   step. Store them in a per-episode "compiled controller" object; keep
   `ActuatorController.motor_names`/`joint_name` as the stored identity (see
   discussion: names are build-time identity, IDs are per-model indices).
3. **Batch the MLP across joints.** Group controllers by joint type (hinge and
   slide share input/output sizes; ball differs). For each group, stack
   observations into an `(n_joints, input_dim)` array and weights into
   `(n_joints, 16, input_dim)` / `(n_joints, out_dim, 16)` arrays, then use one
   `np.einsum`/batched `@` per layer. Root-body terms (`root_up`, root
   velocities) are shared by all joints and can be computed once per step.
4. **Vectorize output scaling.** Clip and scale all outputs with array ops and
   write `data.ctrl[actuator_ids] = ...` in one assignment.

Sine mode is already cheap, but steps 2 and 4 apply to it too.

## Constraints and checks

- Results must stay numerically identical (or within float tolerance) to the
  current controller: compare `metrics.jsonl` from small seeded `evc3d evolve`
  runs before/after, as done for report.md Var 6.
- `video.py` and `viewer.py` call `apply_controller` directly; keep that entry
  point (or update all three callers together).
- Overlaps with `todo/typed-controllers.md` (sine vs neural parameter
  ownership); do that refactor first or design the compiled form to fit it.
- The obsolete open-loop helper (`apply_open_loop_controller`, report.md
  "Script/module redundancy #4") is not part of this; remove it separately.
