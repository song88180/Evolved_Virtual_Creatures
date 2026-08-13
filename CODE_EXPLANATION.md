# Python Code Explanation

This project turns a small graph-like genotype into a MuJoCo creature model. The runnable entrypoint is `generate_model.py`, which loads the genotype, applies random mutations, writes the generated MJCF XML to `generated_creature.xml`, loads that XML into MuJoCo, and opens a viewer where the creature is animated with a simple sinusoidal controller.

The correct command is:

```bash
conda run -n mujoco --no-capture-output python generate_model.py
```

## Main Idea

The code separates a creature into two conceptual layers:

1. **Genotype**: a compact description of body-part types and how they connect.
2. **Phenotype**: the expanded MuJoCo XML model produced from that genotype.

This mirrors evolutionary robotics terminology. The genotype is the recipe, and the phenotype is the actual simulated body generated from that recipe.

## Module Layout

The code is split by responsibility:

- `generate_model.py`: command-line entrypoint for loading, mutating, building, saving, and viewing one creature.
- `evaluate.py`: command-line entrypoint for scoring one genotype on a locomotion task.
- `evolve.py`: command-line entrypoint for population-based mutation search.
- `evol_virtual_creature/genes.py`: `NodeGene` and `ConnectionGene` dataclasses plus validation helpers for attachment faces, symmetry, and orientation.
- `evol_virtual_creature/genotype.py`: top-level `Genotype` dataclass that combines active genes, archived genes, and mutation behavior.
- `evol_virtual_creature/genotype_io.py`: JSON genotype loading, old-schema migration, and serialization.
- `evol_virtual_creature/genotype_mutation.py`: in-place mutation operators for node fields, connection fields, and topology changes.
- `evol_virtual_creature/graph_analysis.py`: graph validation and node-count checks.
- `evol_virtual_creature/phenotype.py`: MJCF phenotype generation, including body quaternions and rotated attachment placement.
- `evol_virtual_creature/evaluation.py`: shared MuJoCo rollout mechanics and lazy task lookup.
- `evol_virtual_creature/evolution_tasks.py`: concrete task configs, evaluators, and registration metadata.
- `evol_virtual_creature/video.py`: optional evaluation video rendering helpers.
- `evol_virtual_creature/viewer.py`: MuJoCo simulation and viewer loop.

## Genotype Data Structures

The genotype section defines three dataclasses: `ConnectionGene`, `NodeGene`, and `Genotype`.

### `ConnectionGene`

```python
@dataclass
class ConnectionGene:
    name: str
    child: str
    axis: Tuple[float, float, float]
    parent_face: str = "+x"
    surface_uv: Tuple[float, float] = (0.0, 0.0)
    symmetry: Tuple[str, ...] = ()
    scale: float = 1.0
    terminal_only: bool = False
    motor_enabled: bool = True
    motor_gear: float = 2.0
    ctrlrange: Tuple[float, float] = (-1.0, 1.0)
    control_amp: float = 0.1
    control_freq: float = 0.1  # harmonic multiplier
    control_phase: float = 0.0
    control_phase_depth_scale: float = 0.0
    control_phase_order_scale: float = 0.0
    neural_w1: Tuple[Tuple[float, ...], ...] = ()
    neural_b1: Tuple[float, ...] = ()
    neural_w2: Tuple[Tuple[float, ...], ...] = ()
    neural_b2: Tuple[float, ...] = ()
    neural_output_axes: Tuple[Tuple[float, float, float], ...] = ()
    orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
```

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
- `control_amp`, `control_freq`, `control_phase`, `control_phase_depth_scale`, and `control_phase_order_scale`: open-loop sine controller parameters. `control_freq` is an actuator-specific harmonic multiplier applied to the genotype's global base frequency.
- `neural_w1`, `neural_b1`, `neural_w2`, and `neural_b2`: the two-layer neural controller tensors for this connection.
- `neural_output_axes`: signed physical axes associated with the neural output rows, allowing joint-type mutations to preserve an output's direction.

New neural connections receive Gaussian-random tensors. Joint-type mutation preserves compatible tensors, expands one-output hinge/slide networks when they become three-output ball networks, and keeps one randomly selected directional output when a ball network becomes hinge or slide. Fixed joints retain their tensors unchanged but exclude them from mutation until they become articulated again.

The connection orientation is constrained: after applying the Euler rotation, the child body's local `+X` axis must point within 90 degrees of the selected parent face normal. This prevents children from being genetically oriented back into or sideways across the parent attachment surface. The `phase_for()` helper combines the base phase with depth-based and actuator-order-based offsets. This lets repeated segments move with a wave-like timing pattern while still using one compact connection recipe. Mirrored copies reuse the same logical actuator order, amplitude, harmonic frequency multiplier, and phase. Their hinge axes and body orientations are reflected so equal controls produce mirror-symmetric motion.

With all three symmetry planes selected, one connection generates `2 ** 3 = 8` mirrored child subtrees. Copies are retained even when an attachment lies directly on a symmetry plane.

### `NodeGene`

```python
@dataclass
class NodeGene:
    name: str
    size: Tuple[float, float, float]
    joint_type: str = "hinge"
    recursive_limit: int = 1
    child_connections: List[str] = field(default_factory=list)
    orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    shape: str = "box"
```

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

```python
@dataclass
class Genotype:
    root: str
    nodes: Dict[str, NodeGene]
    connections: Dict[str, ConnectionGene]
    global_control_freq: float = 1.0
    control_mode: str = "neural"
```

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

The method copies and updates the `current_depths` dictionary:

```python
current_depths = dict(current_depths)
current_depths[node.name] = current_depths.get(node.name, 0) + 1
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

## Program Entry Point

The `main()` function runs the full pipeline:

```python
genotype = load_genotype_from_json(DEFAULT_GENOTYPE_PATH)
genotype.mutation(num_mutations=10)
print("Building MuJoCo organism from mutated genotype.")
builder = PhenotypeBuilder(genotype, max_node=MAX_N_NODES)
mjcf = builder.build()
```

It loads the genotype from JSON, applies random mutations, builds the MJCF XML from the mutated genotype, and stores the XML string in `mjcf`. Because `Genotype.mutation()` prints the mutation report itself, each run shows which genotype parameters changed before the MuJoCo organism is built.

Then it writes the XML to disk:

```python
with open("generated_creature.xml", "w") as f:
    f.write(mjcf)
```

After that, it loads the generated XML directly into MuJoCo:

```python
model = mujoco.MjModel.from_xml_string(mjcf)
data = mujoco.MjData(model)
```

Finally, it opens the MuJoCo viewer:

```python
with mujoco.viewer.launch_passive(model, data) as viewer:
```

Inside the viewer loop, the script drives every actuator with a sine wave:

```python
t = data.time
for actuator_id, controller in zip(actuator_ids, builder.actuator_controllers):
    data.ctrl[actuator_id] = controller.amp * math.sin(
        controller.freq * t + controller.phase
    )
```

`builder.actuator_controllers` stores the controller recipe for each generated motor. Each motor gets its own amplitude and phase from the connection gene that created it, while frequency is the genotype global base frequency multiplied by the connection's harmonic frequency multiplier. This makes the controller part of the genotype, so mutation can affect both body shape and motion.

Then the simulation advances one step and synchronizes the viewer:

```python
mujoco.mj_step(model, data)
viewer.sync()
```

## Generated Output

Running the script produces `generated_creature.xml`. That file contains:

- One root body with a free joint.
- A mutated recursive body plan generated from `examples/example_genotype.json`.
- Two limb bodies attached to each segment.
- One motor for every motor-enabled hinge, slide, or ball joint.
- A floor plane and a light.

The exact XML can differ from run to run because the genotype is randomly mutated before expansion. The resulting creature is not controlled by a learned policy. It uses deterministic sinusoidal controllers whose parameters may also be affected by mutation.

## Extension Points

The current script is intentionally minimal, but it leaves several natural places to grow:

- Add crossover operations to combine two genotypes.
- Record mutation history across multiple mutation steps if longer evolutionary traces are needed.
- Add or tune fitness functions in `evolution_tasks.py`. Current CLI tasks are `swimming_x`, `walking_x`, `flying_x`, `swimming_away`, `walking_away`, and `flying_away`; the `_x` variants reward positive X-axis progress and the `_away` variants reward average speed away from the starting point. Flying tasks start above the floor, use horizontal speed, penalize center-of-mass height loss and earlier ground contact, and add a bonus when no ground contact occurs. New tasks define a config dataclass and decorate their evaluator with `register_task(...)`; lazy loading then supplies CLI discovery, evaluator dispatch, environment selection, and result presentation metadata.
- Replace the open-loop sine controller with an evolved or learned controller.

## Summary

`generate_model.py` demonstrates the core loop of genotype mutation, genotype-to-phenotype generation, and MuJoCo simulation for an evolutionary virtual creature. A small recursive graph describes reusable body parts and controller parameters, `Genotype.mutation()` randomly perturbs that recipe while printing what changed, `PhenotypeBuilder` expands the mutated graph into MJCF XML, and MuJoCo loads the result for simulation.
