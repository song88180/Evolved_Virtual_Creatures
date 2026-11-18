# Python Code Explanation

This project turns a small graph-like genotype into a MuJoCo creature model. The runnable entrypoint is the `evc3d` command (`evc3d/cli/`), with subcommands to evolve, evaluate, mutate, build, and view creatures. For example, `evc3d mutate` applies random mutations to a genotype and `evc3d build` writes the generated MJCF XML to `generated_creature.xml`:

```bash
conda run -n mujoco --no-capture-output evc3d mutate --output mutated.json --mutations 3 --seed 42
conda run -n mujoco --no-capture-output evc3d build --genotype mutated.json
```

## Main Idea

The code separates a creature into two conceptual layers:

1. **Genotype**: a compact description of body-part types and how they connect.
2. **Phenotype**: the expanded MuJoCo XML model produced from that genotype.

This mirrors evolutionary robotics terminology. The genotype is the recipe, and the phenotype is the actual simulated body generated from that recipe.

## Module Layout

The code is split by responsibility:

- `evc3d/cli/`: the `evc3d` command. `args.py` defines the option groups shared by subcommands (genotype, environment, scoring, mutation); `evolve.py`, `evaluate.py`, `mutate.py`, `build.py`, and `view.py` each define one subcommand. `evc3d/__main__.py` enables `python -m evc3d`.
- `evc3d/genes.py`: `NodeGene` and `ConnectionGene` dataclasses plus validation helpers for attachment faces, symmetry, and orientation.
- `evc3d/genotype.py`: top-level `Genotype` dataclass that combines active genes, archived genes, and mutation behavior.
- `evc3d/genotype_io.py`: JSON genotype loading, old-schema migration, and serialization.
- `evc3d/genotype_mutation.py`: in-place mutation operators for node fields, connection fields, and topology changes.
- `evc3d/graph_analysis.py`: graph validation and node-count checks.
- `evc3d/phenotype.py`: MJCF phenotype generation, including body quaternions and rotated attachment placement.
- `evc3d/evaluation.py`: shared MuJoCo evaluation engine, rollout mechanics, and lazy task lookup.
- `evc3d/evolution_tasks/`: task-specific configs, environments, results, fitness/failure callbacks, and registration metadata; `shared.py` owns the declarative environment template, task metadata types, registry state, and rollout policies.
- `evc3d/scene.py`: shared display setup: builds a creature for a task, places and settles it like the scored rollout (without enforcing scoring rules such as start rules or volume limits), tunes lighting and floor, and steps the controlled simulation with evaluation's failure checks.
- `evc3d/video.py`: MP4 recording on top of `scene.py` (`evc3d view --mp4`).
- `evc3d/viewer.py`: interactive MuJoCo window on top of `scene.py` (`evc3d view`), running until closed.

## Genotype Data Structures

The genotype section defines three dataclasses: `ConnectionGene`, `NodeGene`, and `Genotype`. This walkthrough explains what their fields mean; exact types and default values are in the API reference (`docs/api-reference.json`, rendered on the documentation site), which a test keeps in sync with the code.

### `ConnectionGene`

A `ConnectionGene` describes how one body-part node connects to another. Its important fields are:

- `name`: the globally unique connection-gene identifier.
- `child`: the name of the child node to attach.
- `parent_face`: the parent box face where the child joint is attached.
- `surface_uv`: normalized coordinates from `-1` to `1` along the two remaining axes in X/Y/Z order.
- `orientation`: child body Euler orientation in degrees relative to the parent body.
- `symmetry`: any subset of `"xy"`, `"xz"`, and `"yz"`; every selected plane doubles the child subtree.
- `axis`: the translation axis for slide joints, rotation axis for hinge joints, or motor torque axis for ball joints.
- `scale`: geometric scale applied by connection hierarchy order: `node.size * scale ** (order - 1)`. Symmetry siblings created from the same parent share an order.
- `terminal_only`: adds the child only on the terminal occurrence of the parent node. A terminal-only connection cannot point back to its own node.
- `motor_enabled`: whether the articulated joint on this connection gets a motor.
- `motor_gear` and `ctrlrange`: MuJoCo actuator settings for the generated motor.
- `control_amp`, `control_freq`, `control_phase`, `control_phase_depth_scale`, and `control_phase_order_scale`: open-loop sine controller parameters. `control_freq` is a dimensionless multiplier (mutation snaps it to the harmonics 0.5, 1, 2, 3, 4) applied to the genotype's `global_control_freq`, a base angular frequency in rad/s. The joint's sine angular frequency is their product in rad/s; divide by 2π for cycles per second.
- `neural_w1`, `neural_b1`, `neural_w2`, and `neural_b2`: the two-layer neural controller tensors for this connection.
- `neural_output_axes`: signed physical axes associated with the neural output rows, allowing joint-type mutations to preserve an output's direction.

New neural connections receive Gaussian-random tensors. Joint-type mutation preserves compatible tensors, expands one-output hinge/slide networks when they become three-output ball networks, and keeps one randomly selected directional output when a ball network becomes hinge or slide. Fixed joints retain their tensors unchanged but exclude them from mutation until they become articulated again.

The connection orientation is constrained: after applying the Euler rotation, the child body's local `+X` axis must point within 90 degrees of the selected parent face normal. This prevents children from being genetically oriented back into or sideways across the parent attachment surface. The `phase_for()` helper combines the base phase with depth-based and actuator-order-based offsets. This lets repeated segments move with a wave-like timing pattern while still using one compact connection recipe. Mirrored copies reuse the same logical actuator order, amplitude, harmonic frequency multiplier, and phase. Their hinge axes and body orientations are reflected so equal controls produce mirror-symmetric motion.

With all three symmetry planes selected, one connection generates `2 ** 3 = 8` mirrored child subtrees. Copies are retained even when an attachment lies directly on a symmetry plane.

### `NodeGene`

A `NodeGene` describes one reusable body-part type. It includes:

- `name`: a symbolic name such as `"body"`, `"segment"`, or `"limb"`.
- `size`: local bounding half extents for this body part.
- `shape`: body geometry shape: `"box"`, `"ellipsoid"`, `"capsule"`, or `"cylinder"`. Capsules and cylinders use local X as the main axis and normalize Y/Z to the same circular radius.
- `joint_type`: `"free"` for the root body, or `"hinge"`, `"slide"`, or `"ball"` for articulated parts.
- `recursive_limit`: how many times this node type can appear along one recursive path.
- `child_connections`: ordered names resolved through `Genotype.connections`.
- `orientation`: Euler orientation in degrees. Only the root node's orientation is used as the generated root body's world-relative initial orientation.

Different nodes may reference the same connection name, intentionally sharing one mutable connection gene. A node may not repeat a name in its own list.

### `Genotype`

`Genotype` wraps the full creature recipe. `root` names the starting node, while `nodes` and `connections` are name-keyed active gene maps. Archived genes use the same map shape. `control_mode` selects the global controller family for the whole simulation: `"neural"` or `"sine"`.

### Genotype Mutation

`Genotype.mutation()` randomly changes mutable genotype parameters in place before the phenotype is built. It supports two modes:

- `num_mutations`: choose a fixed number of distinct mutable parameters uniformly at random.
- `mutation_rate`: independently mutate each mutable parameter with the given probability.

Mutable parameters include node fields such as `size`, `shape`, `joint_type`, `recursive_limit`, and `orientation`, plus connection fields such as `parent_face`, `surface_uv`, `orientation`, `symmetry`, `axis`, `motor_enabled`, `motor_gear`, `ctrlrange`, and the controller values. The genotype-level `control_mode` is not mutable during evolution. Orientation components mutate with normal angular noise and are wrapped into `[-180, 180]` degrees. If a connection `parent_face` or `orientation` mutation violates the child-normal constraint, the connection is repaired to a face-aligned orientation. Slide joints are disabled for random mutation by default; use `--allow-slide-joint` on mutation entrypoints to let mutations create them.

The method prints the mutation details as it applies them:

```text
Applying 2 genotype mutation(s):
  - connection 'segment' -> 'limb' #1.control_phase_order_scale: 0.0 -> -0.1330023623471946
  - connection 'segment' -> 'segment' #0.motor_gear: 2.0 -> 2.0837977275654938
```

For tuple or list fields, the output names the mutated element and also shows the final full field value:

```text
connection 'segment' -> 'segment' #0.surface_uv[0]: 0.0 -> 0.05 (final surface_uv: (0.05, 0.0))
```

Positive numeric fields are nudged by a small relative amount and clamped above zero. Boolean fields are flipped. Relative fields, such as surface coordinates, axes, phases, and control ranges, receive a random additive perturbation.

## Example Genotype

The `examples/example_genotype.json` file stores a simple creature recipe:

```text
body -> segment -> segment -> ... -> segment
              |          |              |
            limbs      limbs          limbs
```

JSON arrays are used for vector values such as `size`, `surface_uv`, `axis`, and `orientation`. Node `shape` defaults to `"box"` when omitted. Existing JSON files without `orientation` still load: missing root orientations default to identity, and missing connection orientations default to identity for `+x` attachments or to a compatible face-aligned orientation for non-`+x` attachments migrated from older schemas.

The file defines three node types:

- `body`: the root body, using a free joint so the whole creature can move in space.
- `segment`: a repeated hinge-connected body segment.
- `limb`: a side limb attached to each generated segment.

The key recursive detail is this connection:

```json
{
  "child": "segment",
  "axis": [0, 1, 0],
  "parent_face": "-x",
  "surface_uv": [0.0, 0.0],
  "orientation": [0.0, 0.0, 180.0]
}
```

Because `segment` connects to another `segment`, the genotype can expand into a chain. The `recursive_limit=10` on the `segment` node prevents that self-connection from expanding forever, producing up to ten segment bodies along that branch.

Each segment creates a limb at positive Y and mirrors its full subtree across the XZ plane to create the negative-Y limb:

```python
genotype.connections["segment_to_limb"] = ConnectionGene(
    name="segment_to_limb", child="limb", axis=(1, 0, 0),
    parent_face="+y", orientation=(0.0, 0.0, 90.0), symmetry=("xz",),
)
segment.child_connections.append("segment_to_limb")
```

## `PhenotypeBuilder`

`PhenotypeBuilder` is responsible for converting the genotype into MJCF XML.

```python
class PhenotypeBuilder:
    def __init__(self, genotype: Genotype):
        self.genotype = genotype
        self.body_counter = 0
        self.joint_counter = 0
        self.motor_counter = 0
        self.actuator_xml = None
```

The counters ensure that generated bodies, joints, and motors get unique names. This matters because MuJoCo expects named elements to be unique. `actuator_xml` stores the generated `<actuator>` element so recursive body-generation calls can add motors to the same actuator section.

### Name Helpers

The methods `new_body_name()`, `new_joint_name()`, and `new_motor_name()` increment counters and return names such as:

- `segment_2`
- `segment_joint_3`
- `segment_joint_3_motor_2`

These names make the generated XML easier to inspect and avoid collisions when the same genotype node is expanded many times.

## Building the MuJoCo XML

The `build()` method creates the top-level MJCF model:

```python
mujoco_xml = ET.Element("mujoco", model="genotype_creature")
```

It then adds standard MuJoCo sections:

- `<compiler>`: sets angles to degrees.
- `<option>`: sets simulation options. `swimming_x` and `swimming_away` use zero gravity with water-like fluid density and viscosity, `walking_x` and `walking_away` use Earth gravity without fluid drag, and `flying_x` and `flying_away` use Earth gravity with air-like MuJoCo fluid density, viscosity, and geom fluid coefficients.
- `<default>`: defines default geometry, joint, and motor properties.
- `<worldbody>`: contains the floor, light, and creature bodies.
- `<actuator>`: contains motors for motor-enabled hinge, slide, and ball joints.

The default geometry settings make body parts box-shaped, moderately dense, and frictional. Swimming tasks disable floor contacts unless self-collision is enabled; walking tasks enable floor contact and use land friction settings. A built-in checker texture and tiled material are assigned to the floor for visual scale and motion reference. The floor and light are added directly to the world:

```python
floor = ET.SubElement(worldbody, "geom")
floor.set("name", "floor")
floor.set("type", "plane")
floor.set("size", "5 5 0.1")
floor.set("pos", "0 0 -0.05")
```

The root body is created at position `0 0 0.6`, high enough to sit above the floor. Its `quat` attribute is derived from the root node's Euler `orientation`:

```python
root_body = self.create_body(
    worldbody,
    root_node,
    "0 0 0.6",
    quat=matrix_to_quat(euler_degrees_to_matrix(root_node.orientation)),
)
```

After creating the root body, `build()` calls `add_node_to_body()` to recursively fill in the body tree.

## Recursive Body Generation

The `add_node_to_body()` method is the heart of the script. It performs three jobs:

1. Add a joint for the current body part.
2. Add a box geometry for the current body part.
3. Recursively add child body parts.

### Adding Joints

If a node uses a free joint, the script adds a MuJoCo free joint:

```python
if node.joint_type == "free":
    joint = ET.SubElement(parent_xml, "joint")
    joint.set("type", "free")
    joint.set("name", self.new_joint_name(node.name))
```

A free joint gives the root body six degrees of freedom, allowing it to translate and rotate in the world.

If a node uses a hinge, slide, or ball joint, the script creates that joint and an optional motor. Hinge and slide joints use the connection axis directly. A ball joint has three rotational degrees of freedom and its single motor applies torque about the normalized connection axis:

```python
elif node.joint_type in {"hinge", "slide", "ball"}:
    joint_name = self.new_joint_name(node.name)
    joint = ET.SubElement(parent_xml, "joint")
    joint.set("type", node.joint_type)
    joint.set("name", joint_name)
    if node.joint_type != "ball":
        joint.set("axis", vec_to_str(joint_axis))

    motor = ET.SubElement(self.actuator_xml, "motor")
    motor.set("name", self.new_motor_name(joint_name))
    motor.set("joint", joint_name)
```

Each motor-enabled articulated joint receives one actuator. For ball joints, the motor gear is a three-component torque axis.

### Adding Geometry

Each body part receives a box-shaped geometry:

```python
geom = ET.SubElement(parent_xml, "geom")
geom.set("name", f"{parent_xml.get('name')}_geom")
geom.set("size", vec_to_str(node.size))
geom.set("rgba", "0.933333 0.603922 0.301961 1")
```

The default geometry type is already set to `box` in the `<default>` section, so this code only needs to set the size and color.

### Tracking Recursion Depth

`next_depths()` returns a copy of the `current_depths` dictionary with the current node's depth recorded:

```python
next_depths = dict(current_depths)
next_depths[node_name] = node_depth
```

This tracks how many times each node type has appeared along the current branch. Copying the dictionary is important because each recursive branch should track its own path independently.

Before adding a child node, the code checks the child's recursion limit:

```python
child_depth = current_depths.get(child_node.name, 0)
if child_depth >= child_node.recursive_limit:
    continue
```

For the self-recursive `segment` node, this check stops expansion after ten segments.

### Adding Children

For each allowed connection, the code creates a nested MuJoCo `<body>`:

```python
base_body_pos = _surface_attachment_position(
    parent_size=effective_size,
    parent_geom_center=geom_center,
    parent_face=conn.parent_face,
    surface_uv=conn.surface_uv,
)
base_rotation = euler_degrees_to_matrix(conn.orientation)
child_logical_path = (*logical_path, connection_index)
for local_reflection in _symmetry_reflections(conn.symmetry):
    child_reflection = _compose_reflections(reflection, local_reflection)
    child_body_pos = _reflect_point(base_body_pos, geom_center, child_reflection)
    child_rotation = _reflect_rotation_matrix(base_rotation, child_reflection)
    child_normal = _reflect_vector(FACE_NORMALS[conn.parent_face], child_reflection)
    child_geom_center = _child_geom_center_for_attachment(
        child_size,
        child_normal,
        child_rotation,
    )
    child_axis = _reflect_axial_vector(conn.axis, child_reflection)
    child_body = self.create_body(
        parent_xml,
        child_node,
        vec_to_str(child_body_pos),
        quat=matrix_to_quat(child_rotation),
    )
```

Then it recursively calls `add_node_to_body()` for that child:

```python
self.add_node_to_body(
    parent_xml=child_body,
    node=child_node,
    incoming_conn=conn,
    incoming_axis=child_axis,
    current_depths=current_depths,
    connection_orders=child_connection_orders,
    effective_size=child_size,
    geom_center=child_geom_center,
    reflection=child_reflection,
    logical_path=child_logical_path,
)
```

This recursive nesting turns the compact genotype graph into a full MuJoCo body hierarchy. The child body origin stays at the parent attachment point. The child body's `quat` carries the genetic orientation, and the child geom center is offset so the rotated box still touches the selected parent face. Reflection state propagates into descendants, while mirrored actuators share a logical path so their controller signals remain identical.

## Utility Function

```python
def vec_to_str(v):
    return " ".join(str(x) for x in v)
```

MuJoCo XML expects vectors as space-separated strings, such as `"0.25 0.15 0.1"`. This helper converts Python tuples into that XML-friendly format.

## Running a Creature

The `evc3d` command drives the whole pipeline. A typical session:

```bash
evc3d mutate --output mutated.json --mutations 3 --seed 42   # genotype -> mutated genotype
evc3d build --genotype mutated.json --task walking_x         # genotype -> MuJoCo XML
evc3d evaluate --genotype mutated.json --task walking_x      # simulate and score
evc3d view --genotype mutated.json --task walking_x          # watch it (window, or --mp4)
```

`Genotype.mutation()` prints which parameters changed. `PhenotypeBuilder(genotype, definition)` expands the genotype into MJCF XML for the task's environment and records one `ActuatorController` per articulated joint in `builder.actuator_controllers`.

During simulation, `control.apply_controller()` is the single entry point that writes motor commands each step. It looks up each controller's own motors by name, so a ball joint's three motors are all driven by that joint's controller:

- **Neural** (`control_mode = "neural"`, the default): a two-layer network per joint reads the joint's position and velocity plus the root body's orientation and velocity, and outputs one command per motor. The weights come from the connection gene, so they evolve with the body.
- **Sine** (`control_mode = "sine"`): each joint's motors follow `amp * sin(freq * t + phase)`, where `freq` is an angular frequency in rad/s: the genotype's `global_control_freq` times the connection's `control_freq` multiplier.

Evaluation (`evaluation.evaluate_task`) and viewing (`evc3d view`, via `scene.py`) build, place, and settle the creature the same way, so what you watch starts exactly like what was scored.

## Generated Output

`evc3d build` writes `generated_creature.xml` by default. For the example genotype it contains:

- One root body with a free joint.
- The recursive body plan generated from `examples/example_genotype.json` (or a mutated copy from `evc3d mutate`).
- Two limb bodies attached to each segment.
- One motor for every motor-enabled hinge, slide, or ball joint.
- A floor plane, a light, and the task's physics settings (gravity, fluid, collisions).

The XML describes the body and world only; the controller runs in Python, so opening the file in MuJoCo's standalone viewer shows a limp creature. Use `evc3d view` to watch it move.

## Extension Points

Natural places to grow:

- Add crossover operations to combine two genotypes.
- Record mutation history across multiple mutation steps if longer evolutionary traces are needed.
- Add or tune fitness functions in the task-specific modules under `evolution_tasks/`. Current CLI tasks are `swimming_x`, `walking_x`, `flying_x`, `swimming_away`, `walking_away`, and `flying_away`; the `_x` variants reward positive X-axis progress and the `_away` variants reward average speed away from the starting point. Flying tasks start above the floor, use horizontal speed, penalize center-of-mass height loss and earlier ground contact, and add a bonus when no ground contact occurs. A new task is a single module whose filename is its task name and which contains its config and result dataclasses, fitness and failure callbacks, rollout policy, and `TASK_DEFINITION`; lazy loading then supplies CLI discovery, shared evaluation dispatch, environment selection, and result presentation metadata without registration side effects.

## Summary

The `evc3d` subcommands demonstrate the core loop of genotype mutation, genotype-to-phenotype generation, and MuJoCo simulation for an evolutionary virtual creature. A small recursive graph describes reusable body parts and controller parameters, `Genotype.mutation()` randomly perturbs that recipe while printing what changed, `PhenotypeBuilder` expands the mutated graph into MJCF XML, and MuJoCo loads the result for simulation.
