# Python Code Explanation

This project contains a single Python script, `generate_model.py`, that turns a small graph-like genotype into a MuJoCo creature model. It writes the generated MJCF XML to `generated_creature.xml`, loads that XML into MuJoCo, and opens a viewer where the creature is animated with a simple sinusoidal controller.

The script header currently says to run `python genotype_to_mujoco.py`, but the file in this repository is named `generate_model.py`. The correct command is:

```bash
python generate_model.py
```

## Main Idea

The code separates a creature into two conceptual layers:

1. **Genotype**: a compact description of body-part types and how they connect.
2. **Phenotype**: the expanded MuJoCo XML model produced from that genotype.

This mirrors evolutionary robotics terminology. The genotype is the recipe, and the phenotype is the actual simulated body generated from that recipe.

## Imports

```python
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
import mujoco
import mujoco.viewer
```

The script uses:

- `dataclasses` to define lightweight data containers for genotype information.
- `typing` to make the expected shapes of dictionaries, lists, and tuples clear.
- `xml.etree.ElementTree` to build MuJoCo XML programmatically.
- `mujoco` and `mujoco.viewer` to load, simulate, and display the generated creature.

## Genotype Data Structures

The genotype section defines three dataclasses: `ConnectionGene`, `NodeGene`, and `Genotype`.

### `ConnectionGene`

```python
@dataclass
class ConnectionGene:
    child: str
    pos: Tuple[float, float, float]
    axis: Tuple[float, float, float]
    scale: float = 1.0
    terminal_only: bool = False
```

A `ConnectionGene` describes how one body-part node connects to another. Its important fields are:

- `child`: the name of the child node to attach.
- `pos`: the child body's position relative to its parent.
- `axis`: an intended connection or joint axis.
- `scale`: a future extension point for resizing child parts.
- `terminal_only`: a future extension point for only adding a child at the end of a recursive chain.

In the current script, `child` and `pos` are actively used. `axis`, `scale`, and `terminal_only` are stored but not yet used during XML generation.

### `NodeGene`

```python
@dataclass
class NodeGene:
    name: str
    size: Tuple[float, float, float]
    joint_type: str = "hinge"
    joint_axis: Tuple[float, float, float] = (0, 1, 0)
    recursive_limit: int = 1
    children: List[ConnectionGene] = field(default_factory=list)
```

A `NodeGene` describes one reusable body-part type. It includes:

- `name`: a symbolic name such as `"body"`, `"segment"`, or `"limb"`.
- `size`: the MuJoCo box geometry size for this body part.
- `joint_type`: either `"free"` for the root body or `"hinge"` for articulated parts.
- `joint_axis`: the axis used for hinge joints.
- `recursive_limit`: how many times this node type can appear along one recursive path.
- `children`: connections from this node to other nodes.

The `field(default_factory=list)` call is important. It gives each `NodeGene` its own independent child list instead of accidentally sharing one list between all instances.

### `Genotype`

```python
@dataclass
class Genotype:
    root: str
    nodes: Dict[str, NodeGene]
```

`Genotype` wraps the full creature recipe. `root` names the starting node, and `nodes` maps node names to their `NodeGene` definitions.

## Example Genotype

The `make_example_genotype()` function builds a simple creature recipe:

```text
body -> segment -> segment -> segment -> segment
              |          |          |          |
            limbs      limbs      limbs      limbs
```

The function creates three node types:

- `body`: the root body, using a free joint so the whole creature can move in space.
- `segment`: a repeated hinge-connected body segment.
- `limb`: a side limb attached to each generated segment.

The key recursive detail is this connection:

```python
segment.children.append(
    ConnectionGene(
        child="segment",
        pos=(0.22, 0.0, 0.0),
        axis=(0, 1, 0),
    )
)
```

Because `segment` connects to another `segment`, the genotype can expand into a chain. The `recursive_limit=4` on the `segment` node prevents that self-connection from expanding forever.

Each segment also creates two limbs, one at positive Y and one at negative Y:

```python
segment.children.append(ConnectionGene(child="limb", pos=(0.0, 0.14, 0.0), axis=(1, 0, 0)))
segment.children.append(ConnectionGene(child="limb", pos=(0.0, -0.14, 0.0), axis=(1, 0, 0)))
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
```

The counters ensure that generated bodies, joints, and motors get unique names. This matters because MuJoCo expects named elements to be unique.

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
- `<option>`: sets gravity to Earth-like gravity.
- `<default>`: defines default geometry, joint, and motor properties.
- `<worldbody>`: contains the floor, light, and creature bodies.
- `<actuator>`: contains motors for hinge joints.

The floor and light are added directly to the world:

```python
floor = ET.SubElement(worldbody, "geom")
floor.set("name", "floor")
floor.set("type", "plane")
floor.set("size", "5 5 0.1")
floor.set("pos", "0 0 -0.05")
```

The root body is created at position `0 0 0.6`, high enough to sit above the floor:

```python
root_body = ET.SubElement(worldbody, "body")
root_body.set("name", self.new_body_name(root_node.name))
root_body.set("pos", "0 0 0.6")
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

If a node uses a hinge joint, the script creates both a hinge joint and a motor attached to that joint:

```python
elif node.joint_type == "hinge":
    joint_name = self.new_joint_name(node.name)
    joint = ET.SubElement(parent_xml, "joint")
    joint.set("type", "hinge")
    joint.set("name", joint_name)
    joint.set("axis", vec_to_str(node.joint_axis))

    motor = ET.SubElement(actuator_xml, "motor")
    motor.set("name", self.new_motor_name(joint_name))
    motor.set("joint", joint_name)
```

This is why the generated XML contains one actuator motor for every hinge joint.

### Adding Geometry

Each body part receives a box-shaped geometry:

```python
geom = ET.SubElement(parent_xml, "geom")
geom.set("name", f"{parent_xml.get('name')}_geom")
geom.set("size", vec_to_str(node.size))
geom.set("rgba", "0.6 0.7 0.9 1")
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

For the self-recursive `segment` node, this check stops expansion after four segments.

### Adding Children

For each allowed connection, the code creates a nested MuJoCo `<body>`:

```python
child_body = ET.SubElement(parent_xml, "body")
child_body.set("name", self.new_body_name(child_node.name))
child_body.set("pos", vec_to_str(conn.pos))
```

Then it recursively calls `add_node_to_body()` for that child:

```python
self.add_node_to_body(
    parent_xml=child_body,
    node=child_node,
    current_depths=current_depths,
    actuator_xml=actuator_xml,
)
```

This recursive nesting is what turns the compact genotype graph into a full MuJoCo body hierarchy.

## Utility Function

```python
def vec_to_str(v):
    return " ".join(str(x) for x in v)
```

MuJoCo XML expects vectors as space-separated strings, such as `"0.25 0.15 0.1"`. This helper converts Python tuples into that XML-friendly format.

## Program Entry Point

The `main()` function runs the full pipeline:

```python
genotype = make_example_genotype()
builder = PhenotypeBuilder(genotype)
mjcf = builder.build()
```

It creates the genotype, builds the MJCF XML, and stores the XML string in `mjcf`.

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
for i in range(model.nu):
    data.ctrl[i] = 0.5 * __import__("math").sin(4 * t + i)
```

`model.nu` is the number of control inputs, which matches the number of motors. Each motor gets a phase offset based on `i`, causing the joints to move in a wave-like open-loop pattern.

Then the simulation advances one step and synchronizes the viewer:

```python
mujoco.mj_step(model, data)
viewer.sync()
```

## Generated Output

Running the script produces `generated_creature.xml`. That file contains:

- One root body with a free joint.
- Four generated segment bodies.
- Two limb bodies attached to each segment.
- One motor for every hinge joint.
- A floor plane and a light.

The resulting creature is not controlled by a learned policy. It simply uses a deterministic sinusoidal controller to demonstrate that the generated joints and motors work.

## Extension Points

The current script is intentionally minimal, but it leaves several natural places to grow:

- Use `ConnectionGene.scale` to resize child body parts.
- Use `ConnectionGene.axis` to influence child orientation or joint axes.
- Implement `terminal_only` so some children appear only at the end of recursive chains.
- Add mutation and crossover operations to evolve genotypes.
- Add a fitness function, such as forward distance traveled.
- Replace the open-loop sine controller with an evolved or learned controller.
- Pretty-print the generated XML for easier reading.

## Summary

`generate_model.py` demonstrates the core loop of genotype-to-phenotype generation for an evolutionary MuJoCo creature. A small recursive graph describes reusable body parts, `PhenotypeBuilder` expands that graph into MJCF XML, and MuJoCo loads the result for simulation. The script is a compact starting point for evolving virtual creatures because the genotype is easy to mutate while the generated phenotype is directly runnable in MuJoCo.
