# Evolved Virtual Creatures

### Evolve 3D virtual creatures for swimming, walking, and flying in MuJoCo.

This project represents virtual creatures as compact, mutable graphs and expands them into MuJoCo models for simulation. A population-based evolutionary loop searches for body plans and controllers that perform locomotion tasks such as moving along the x-axis or traveling away from the starting point.

Check this [video](https://www.youtube.com/watch?v=vstG57H8ZOY) here!

## Table of Contents

1. [Getting Started](#getting-started)
2. [Usage](#usage)
   1. [Evolving creatures](#evolving-creatures)
   2. [Evaluating a creature](#evaluating-a-creature)
   3. [Generating a phenotype](#generating-a-phenotype)
3. [File Formats](#file-formats)
   1. [Genotype JSON](#genotype-json)
   2. [Phenotype MJCF XML](#phenotype-mjcf-xml)

## Getting Started

### Prerequisites

- [Git](https://git-scm.com/)
- [Conda](https://docs.conda.io/) or another Python environment manager
- Python 3.11 or newer

An available graphical display is needed for the interactive MuJoCo viewer (`evc3d view`). On Linux, `evc3d view --mp4` automatically selects EGL, so video rendering can also work on a headless machine with a compatible graphics driver.

### Installation

Clone the repository:

```bash
git clone https://github.com/song88180/Evolved_Virtual_Creatures.git
cd Evolved_Virtual_Creatures
```

Create and activate a conda environment. The name `evol-virtual-creature` is only an example; you may choose any environment name.

```bash
conda create --name evol-virtual-creature python=3.11 pip
conda activate evol-virtual-creature
```

Install the runtime dependencies and the `evc3d` package in editable mode (code changes take effect without reinstalling):

```bash
python -m pip install mujoco numpy imageio imageio-ffmpeg
python -m pip install -e .
```

This provides the `evc3d` command. Confirm that it loads correctly:

```bash
evc3d --help
evc3d evolve --help
```

`python -m evc3d ...` works the same way without the editable install.

### Commands

| Command | Purpose |
| --- | --- |
| `evc3d evolve` | Run mutation-based evolution and write a run directory. |
| `evc3d evaluate` | Score one genotype on a task and print its metrics. |
| `evc3d mutate` | Write a randomly mutated copy of a genotype. |
| `evc3d build` | Write the MuJoCo XML scene for a genotype. |
| `evc3d view` | Watch a genotype perform its task in an interactive window, or record it with `--mp4`. |

Options are grouped by purpose in each command's `--help`: *environment* (task, fluid, collision, size limit), *scoring* (episode duration and fitness penalties), and *mutation* (which mutations are allowed).

## Usage

### Evolving creatures

`evc3d evolve` creates an initial population from a seed genotype, mutates and evaluates the population, selects fitter creatures, and repeats this process for the requested number of generations.

The controller family for the run comes from the seed genotype's top-level `control_mode`. `neural` uses an evolved two-layer neural network that reacts to simulation observations, while `sine` uses evolved open-loop oscillations defined by amplitude, frequency, and phase. If `--control-mode` is given and differs from the seed's mode, the run stops with an error. Add `--override-control-mode` to convert the seed instead: the old controller's parameters are discarded and the new controller starts motionless (zero neural weights, or zero sine amplitude).

Its main inputs are:

| Option | Description |
| --- | --- |
| `--task` | Locomotion task used to calculate fitness. Defaults to `swimming_x`. |
| `--genotype` | Seed genotype JSON file. Defaults to `examples/example_genotype.json`. |
| `--control-mode` | Optional controller family for the run: observation-driven `neural` or open-loop `sine`. Defaults to the seed genotype's `control_mode`; a mismatch is an error unless `--override-control-mode` is set. |
| `--override-control-mode` | Convert the seed genotype to `--control-mode` when the two differ. Requires `--control-mode`. |
| `--population-size` | Number of creatures evaluated in each generation. |
| `--generations` | Number of evolutionary generations after generation zero. |
| `--elite-count` | Number of highest-fitness creatures copied unchanged into the next generation. Defaults to `5`. |
| `--tournament-size` | Number of candidates sampled for each parent-selection tournament; larger values increase selection pressure toward fitter creatures. Defaults to `4`. |
| `--max-mutations` | Maximum number of genotype mutations applied to each non-elite child; must be at least `--min-mutations`. Defaults to `5`. |
| `--threads` | Number of worker processes used for population evaluation. |
| `--output-dir` | Destination for run artifacts. By default, a timestamped directory is created under `runs/`. |

For a small swimming run with the default neural controller:

```bash
evc3d evolve \
  --task swimming_x \
  --genotype examples/single_root.json \
  --population-size 10 \
  --elite-count 1 \
  --tournament-size 20 \
  --max-mutations 5 \
  --generations 5 \
  --threads 2 \
  --latest-best-only \
  --output-dir runs/readme_swimming_example \
  --seed 2
```

Evolution prints a progress line for every generation. The output directory contains:

| Output | Description |
| --- | --- |
| `config.json` | Snapshot of the command-line configuration. |
| `metrics.jsonl` | One JSON object per generation with population and fitness statistics. |
| `best_genotype.json` | Best genotype found across the entire run. |
| `latest_best_genotype.json` | Best genotype from the most recently evaluated generation. |
| `best_metrics.json` | Detailed evaluation metrics for the all-time best creature. |
| `best_creature.xml` | MuJoCo MJCF phenotype generated from the all-time best genotype. |
| `generation_bests/generation_NNNN.json` | Generation-best checkpoints, unless `--latest-best-only` is used. |

Use `evc3d evolve --help` to see mutation, selection, environment, collision, volume, and checkpoint options.

### Evaluating a creature

`evc3d evaluate` builds and simulates one genotype on a selected task. It prints fitness together with movement, energy, angular-speed, body-count, actuator-count, and volume measurements. Invalid builds or disqualified creatures instead report their failure reason and assigned fitness.

Evaluate the winner of the example evolution run:

```bash
evc3d evaluate \
  --task swimming_x \
  --genotype runs/readme_swimming_example/best_genotype.json
```

The input is a genotype JSON file, and the results are written to the terminal. To watch the creature, use `evc3d view`.

### Watching a creature

`evc3d view` simulates a genotype in its task environment, starting exactly like the scored episode (same placement, settling, and controller). Scoring rules such as volume limits and the task's start rules are not applied, so any creature that can be built can be watched, including ones that evaluation rejects. By default it opens an interactive MuJoCo window that keeps simulating until you close it, paced to real time (scaled by `--speed`):

```bash
evc3d view \
  --task swimming_x \
  --genotype runs/readme_swimming_example/best_genotype.json \
  --track-root
```

Add `--mp4 PATH` to record the episode to a video file instead of opening a window:

```bash
evc3d view \
  --task swimming_x \
  --genotype runs/readme_swimming_example/best_genotype.json \
  --mp4 videos/best_swimmer.mp4 \
  --duration 10 \
  --track-root \
  --fps 30 \
  --width 960 \
  --height 544
```

Passing `--mp4` without a path writes `example_<task>.mp4` in the current directory. `--track-root`, `--speed`, `--spotlight`, and `--shadowsize` apply to both modes; `--duration`, `--fps`, `--width`, `--height`, and `--camera-circle-around` only apply with `--mp4`. See `evc3d view --help` for all options.

### Evolution tasks

The available locomotion tasks are:

| Task | Objective |
| --- | --- |
| `swimming_x` | Swim in the positive x direction. |
| `swimming_away` | Swim away from the starting point. |
| `walking_x` | Walk in the positive x direction. |
| `walking_away` | Walk away from the starting point. |
| `flying_x` | Fly in the positive x direction. |
| `flying_away` | Fly away from the starting point. |

You can customize new tasks by adding task-specifying .py files in [`evc3d/evolution_tasks/`](evc3d/evolution_tasks)

### Mutating a genotype

`evc3d mutate` applies random mutations to a genotype and saves the result as a new genotype file. The same `--seed` always produces the same result, and `--output` must differ from `--genotype`, so the source is never overwritten.

```bash
evc3d mutate \
  --genotype examples/example_genotype.json \
  --output mutated.json \
  --mutations 3 \
  --seed 42
```

The mutation options (`--allow-slide-joint`, `--disallow-topology-mutations`, `--disallow-root-mutation`, `--topology-mutation-rate-min`) match those of `evc3d evolve`.

### Generating a phenotype

`evc3d build` expands a genotype into the MuJoCo MJCF XML scene. The scene contains the creature and the selected task's physical environment (gravity, fluid, floor collision); it does not contain the controller, which runs in Python.

```bash
evc3d build \
  --genotype mutated.json \
  --task walking_x \
  --output generated_creature.xml
```

## File Formats

### Genotype JSON

A genotype is the evolvable recipe for a creature. It is a directed graph whose named node genes describe reusable body parts and whose named connection genes describe how those parts attach. Recursion and symmetry allow a small genotype to expand into a much larger body.

The top-level fields are:

| Field | Description |
| --- | --- |
| `root` | Name of the node where phenotype expansion begins. |
| `control_mode` | Required global controller family: `neural` or `sine`. |
| `global_control_freq` | Sine mode only. Base **angular** frequency in radians per second (not hertz), shared by all sine controllers. Defaults to `1.0`. |
| `nodes` | Name-keyed map of active node genes. |
| `connections` | Name-keyed map of active connection genes. |
| `archived_nodes` | Optional map of inactive node genes retained by topology mutation. |
| `archived_connections` | Optional map of inactive connection genes retained by topology mutation. |

A node gene supports these principal fields:

| Field | Description |
| --- | --- |
| `size` | Three positive dimensions describing the body's local half extents. |
| `shape` | `box`, `ellipsoid`, `capsule`, or `cylinder`; defaults to `box`. |
| `joint_type` | Root or child joint type: `free`, `fixed`, `hinge`, `slide`, or `ball`. |
| `recursive_limit` | Maximum occurrences of this node type along one recursive path. |
| `child_connections` | Ordered list of connection names to expand from this node. |
| `orientation` | Euler orientation in degrees; used as the root's initial orientation. |

A connection gene supports these principal fields:

| Field | Description |
| --- | --- |
| `child` | Name of the node gene attached by this connection. |
| `axis` | Three-component joint axis in the child body's local frame. |
| `parent_face` | Parent attachment face: `+x`, `-x`, `+y`, `-y`, `+z`, or `-z`. |
| `surface_uv` | Two coordinates from `-1` to `1` that position the attachment on the parent face. |
| `child_surface_uv` | Two coordinates from `-1` to `1` that select the attachment point on the child surface. |
| `orientation` | Child Euler orientation in degrees relative to the attachment frame. |
| `symmetry` | Optional combination of `xy`, `xz`, and `yz` reflection planes for duplicating the child subtree. |
| `scale` | Positive size multiplier applied to the child and its descendants. |
| `terminal_only` | If `true`, expands this connection only at the end of a recursive chain. |
| `motor_enabled` | Whether an articulated child joint receives an actuator. |
| `motor_gear` | MuJoCo actuator gear value. |
| `ctrlrange` | Minimum and maximum actuator control values. |
| `control_amp` | Amplitude used by the sine controller. |
| `control_freq` | Sine mode only. Dimensionless **multiplier** on `global_control_freq` (not a frequency itself). The joint's command is `amp * sin(global_control_freq * control_freq * t + phase)`, so the defaults `1.0 × 10` give 10 rad/s ≈ 1.6 cycles per second. Mutation snaps it to the harmonics 0.5, 1, 2, 3, or 4. |
| `control_phase` | Base phase used by the sine controller. |
| `control_phase_depth_scale` | Phase offset added for each recursive depth level. |
| `control_phase_order_scale` | Phase offset added according to sibling expansion order. |
| `neural_w1` | First-layer weight matrix for the neural controller. |
| `neural_b1` | First-layer bias vector for the neural controller. |
| `neural_w2` | Output-layer weight matrix for the neural controller. |
| `neural_b2` | Output-layer bias vector for the neural controller. |
| `neural_output_axes` | Local actuator axes corresponding to neural-network outputs. |

This abbreviated genotype creates a body followed by a recursive chain of segments:

```json
{
  "root": "body",
  "control_mode": "sine",
  "global_control_freq": 1.0,
  "nodes": {
    "body": {
      "size": [0.25, 0.15, 0.1],
      "joint_type": "free",
      "recursive_limit": 1,
      "child_connections": ["body_to_segment"]
    },
    "segment": {
      "size": [0.18, 0.08, 0.08],
      "joint_type": "hinge",
      "recursive_limit": 4,
      "child_connections": ["segment_recursive"]
    }
  },
  "connections": {
    "body_to_segment": {
      "child": "segment",
      "axis": [0, 1, 0],
      "parent_face": "-x",
      "orientation": [0, 0, 180]
    },
    "segment_recursive": {
      "child": "segment",
      "axis": [0, 1, 0],
      "parent_face": "-x",
      "orientation": [0, 0, 180],
      "control_phase_depth_scale": -0.8
    }
  }
}
```

See [`examples/example_genotype_neural.json`](examples/example_genotype_neural.json) for a complete neural-controller example.

### Phenotype MJCF XML

The phenotype is the concrete [MuJoCo MJCF](https://mujoco.readthedocs.io/en/stable/XMLreference.html) model generated from a genotype. During expansion, recursive and symmetric graph references become individual XML bodies with unique names.

The generated document includes:

- `<option>` settings for gravity, timestep, fluid density, and viscosity.
- A `<worldbody>` containing the environment and expanded creature bodies.
- Concrete `<geom>` and `<joint>` elements for body shape and articulation.
- An `<actuator>` section containing motors for enabled articulated joints.
- Collision and rendering settings appropriate to the selected environment.

Phenotype XML is generated output, not the genetic material mutated during evolution. To continue an evolutionary run or reproduce a creature under another task, use its genotype JSON. Use the XML when inspecting or loading the already-expanded MuJoCo model.

`evc3d build` writes `generated_creature.xml` by default, while `evc3d evolve` writes the current all-time winner to `best_creature.xml` inside the run directory.
