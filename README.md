# Evolved Virtual Creatures

<div align="center">
  <h3>Evolve simulated creatures for swimming, walking, and flying in MuJoCo.</h3>
</div>

This project represents virtual creatures as compact, mutable graphs and expands them into MuJoCo models for simulation. A population-based evolutionary loop searches for body plans and controllers that perform locomotion tasks such as moving along the x-axis or traveling away from the starting point.

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#getting-started">Getting Started</a></li>
    <li>
      <a href="#usage">Usage</a>
      <ol>
        <li><a href="#evolving-creatures">Evolving creatures</a></li>
        <li><a href="#evaluating-a-creature">Evaluating a creature</a></li>
        <li><a href="#generating-a-phenotype">Generating a phenotype</a></li>
      </ol>
    </li>
    <li>
      <a href="#file-formats">File Formats</a>
      <ol>
        <li><a href="#genotype-json">Genotype JSON</a></li>
        <li><a href="#phenotype-mjcf-xml">Phenotype MJCF XML</a></li>
      </ol>
    </li>
  </ol>
</details>

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

The available locomotion tasks are:

| Task | Objective |
| --- | --- |
| `swimming_x` | Swim in the positive x direction. |
| `swimming_away` | Swim away from the starting point. |
| `walking_x` | Walk in the positive x direction. |
| `walking_away` | Walk away from the starting point. |
| `flying_x` | Fly in the positive x direction. |
| `flying_away` | Fly away from the starting point. |

Run commands from the repository root.

### Evolving creatures

`evolve.py` creates an initial population from a seed genotype, mutates and evaluates the population, selects fitter creatures, and repeats this process for the requested number of generations.

Its main inputs are:

| Option | Description |
| --- | --- |
| `--task` | Locomotion task used to calculate fitness. Defaults to `swimming_x`. |
| `--genotype` | Seed genotype JSON file. Defaults to `examples/example_genotype.json`. |
| `--control-mode` | Controller family, either `neural` or `sine`. It must match the seed genotype. |
| `--population-size` | Number of creatures evaluated in each generation. |
| `--generations` | Number of evolutionary generations after generation zero. |
| `--threads` | Number of worker processes used for population evaluation. |
| `--output-dir` | Destination for run artifacts. By default, a timestamped directory is created under `runs/`. |

For a small swimming run:

```bash
python evolve.py \
  --task swimming_x \
  --genotype examples/example_genotype.json \
  --population-size 10 \
  --generations 5 \
  --threads 2 \
  --output-dir runs/readme_swimming_example \
  --seed 42
```

For evolution with the open-loop sine controller:

```bash
python evolve.py \
  --task swimming_away \
  --control-mode sine \
  --genotype examples/single_root_sine.json \
  --population-size 20 \
  --generations 20 \
  --latest-best-only
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

Evaluate the bundled example:

```bash
python evaluate.py \
  --task swimming_x \
  --genotype examples/example_genotype.json \
  --duration 10
```

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

The input is a genotype JSON file. Evaluation results are written to the terminal, and `--video PATH` additionally writes an MP4 to `PATH`. Passing `--video` without a path uses `example_<task>.mp4` in the repository root. Camera rotation, playback speed, lighting, resolution, and other rendering settings are available through `python evaluate.py --help`.

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

A connection gene identifies its `child` node and controls attachment through `parent_face`, `surface_uv`, `child_surface_uv`, and `orientation`. It can reflect a complete child subtree with `symmetry`, resize repeated descendants with `scale`, and restrict an attachment to the end of a recursive chain with `terminal_only`. Joint axis, motor settings, and sine or neural controller parameters also live on connection genes.

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

See [`examples/example_genotype.json`](examples/example_genotype.json) for a complete neural-controller example and [`examples/single_root_sine.json`](examples/single_root_sine.json) for a sine-controller seed.

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
