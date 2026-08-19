# Evolved Virtual Creatures

### Evolve 3D virtual creatures for swimming, walking, and flying in MuJoCo.

This project represents virtual creatures as compact, mutable graphs and expands them into MuJoCo models for simulation. A population-based evolutionary loop searches for body plans and controllers that perform locomotion tasks such as moving along the x-axis or traveling away from the starting point.

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

An available graphical display is needed for the interactive MuJoCo viewer. On Linux, `evaluate.py` automatically selects EGL when an MP4 video is requested, so video rendering can also work on a headless machine with a compatible graphics driver.

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

Install the runtime dependencies:

```bash
python -m pip install mujoco numpy imageio imageio-ffmpeg
```

Confirm that the command-line programs load correctly:

```bash
python evolve.py --help
python evaluate.py --help
```

## Usage

### Evolving creatures

`evolve.py` creates an initial population from a seed genotype, mutates and evaluates the population, selects fitter creatures, and repeats this process for the requested number of generations.

`--control-mode` selects one controller family for the entire evolutionary run. `neural` uses an evolved two-layer neural network that reacts to simulation observations, while `sine` uses evolved open-loop oscillations defined by amplitude, frequency, and phase. The selected mode must match the seed genotype's top-level `control_mode`; evolution does not convert a seed between controller families. If the option is omitted, it defaults to `neural`.

Its main inputs are:

| Option | Description |
| --- | --- |
| `--task` | Locomotion task used to calculate fitness. Defaults to `swimming_x`. |
| `--genotype` | Seed genotype JSON file. Defaults to `examples/example_genotype.json`. |
| `--control-mode` | Controller family for the run: observation-driven `neural` or open-loop `sine`. Must match the seed genotype; defaults to `neural`. |
| `--population-size` | Number of creatures evaluated in each generation. |
| `--generations` | Number of evolutionary generations after generation zero. |
| `--elite-count` | Number of highest-fitness creatures copied unchanged into the next generation. Defaults to `5`. |
| `--tournament-size` | Number of candidates sampled for each parent-selection tournament; larger values increase selection pressure toward fitter creatures. Defaults to `4`. |
| `--max-mutations` | Maximum number of genotype mutations applied to each non-elite child; must be at least `--min-mutations`. Defaults to `5`. |
| `--threads` | Number of worker processes used for population evaluation. |
| `--output-dir` | Destination for run artifacts. By default, a timestamped directory is created under `runs/`. |

For a small swimming run with the default neural controller:

```bash
python evolve.py \
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

Use `python evolve.py --help` to see mutation, selection, environment, collision, volume, and checkpoint options.

### Evaluating a creature

`evaluate.py` builds and simulates one genotype on a selected task. It prints fitness together with movement, energy, angular-speed, body-count, actuator-count, and volume measurements. Invalid builds or disqualified creatures instead report their failure reason and assigned fitness.

Evaluate the winner of the example evolution run:

```bash
python evaluate.py \
  --task swimming_x \
  --genotype runs/readme_swimming_example/best_genotype.json
```

Generate a video while evaluating a creature:

```bash
python evaluate.py \
  --task swimming_x \
  --genotype runs/readme_swimming_example/best_genotype.json \
  --duration 10 \
  --video videos/best_swimmer.mp4 \
  --track-root \
  --fps 30 \
  --width 960 \
  --height 544
```

The input is a genotype JSON file. Evaluation results are written to the terminal, and `--video PATH` additionally writes an MP4 to `PATH`. Passing `--video` without a path uses `example_<task>.mp4` in the repository root. Camera rotation, playback speed, lighting, resolution, and other rendering settings are available through `python evaluate.py --help`. Run commands from the repository root.

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

You can customize new tasks by adding task-specifying .py files in [`evol_virtual_creature/evolution_tasks/`](evol_virtual_creature/evolution_tasks)

### Generating a phenotype

`generate_model.py` is a direct genotype-to-phenotype utility. It optionally mutates a genotype, writes the expanded MJCF XML, and opens the model in the interactive MuJoCo viewer.

```bash
python generate_model.py \
  --genotype examples/example_genotype.json \
  --output generated_creature.xml \
  --mutations 0
```

The input genotype is not modified on disk. The generated XML is written to the path given by `--output`; closing the viewer ends the program.

## File Formats

### Genotype JSON

A genotype is the evolvable recipe for a creature. It is a directed graph whose named node genes describe reusable body parts and whose named connection genes describe how those parts attach. Recursion and symmetry allow a small genotype to expand into a much larger body.

The top-level fields are:

| Field | Description |
| --- | --- |
| `root` | Name of the node where phenotype expansion begins. |
| `control_mode` | Global controller family: `neural` or `sine`. Defaults to `neural`. |
| `global_control_freq` | Positive base controller frequency. Defaults to `1.0`. |
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
| `control_freq` | Sine-frequency multiplier applied to `global_control_freq`. |
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

`generate_model.py` writes `generated_creature.xml` by default, while `evolve.py` writes the current all-time winner to `best_creature.xml` inside the run directory.
