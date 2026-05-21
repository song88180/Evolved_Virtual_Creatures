"""
Minimal genotype -> phenotype example for MuJoCo.

Install:
    pip install mujoco

Run:
    python genotype_to_mujoco.py

This generates a simple articulated creature from a directed graph genotype
and opens it in the MuJoCo viewer.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Mapping, Optional, Tuple
import xml.etree.ElementTree as ET
import json
import math
import random
import mujoco
import mujoco.viewer


# -----------------------------
# 1. Define genotype structures
# -----------------------------

@dataclass
class ConnectionGene:
    child: str
    pos: Tuple[float, float, float]
    axis: Tuple[float, float, float]
    scale: float = 1.0
    terminal_only: bool = False
    motor_enabled: bool = True
    motor_gear: float = 2.0
    ctrlrange: Tuple[float, float] = (-1.0, 1.0)
    control_amp: float = 0.1
    control_freq: float = 0.1
    control_phase: float = 0.0
    control_phase_depth_scale: float = 0.0
    control_phase_order_scale: float = 0.0

    def phase_for(self, depth: int, order: int) -> float:
        return (
            self.control_phase
            + self.control_phase_depth_scale * (depth - 1)
            + self.control_phase_order_scale * order
        )


@dataclass
class NodeGene:
    name: str
    size: Tuple[float, float, float]
    joint_type: str = "hinge"
    joint_axis: Tuple[float, float, float] = (0, 1, 0)
    recursive_limit: int = 1
    children: List[ConnectionGene] = field(default_factory=list)


@dataclass
class _MutableParameter:
    name: str
    mutate: Callable[[random.Random], None]


@dataclass
class Genotype:
    root: str
    nodes: Dict[str, NodeGene]

    POS_MUTATION_RANGE: ClassVar[float] = 0.05
    AXIS_MUTATION_RANGE: ClassVar[float] = 0.1
    PHASE_MUTATION_RANGE: ClassVar[float] = 0.25
    CTRLRANGE_MUTATION_RANGE: ClassVar[float] = 0.1
    MIN_POSITIVE_VALUE: ClassVar[float] = 1e-6

    def mutation(
        self,
        num_mutations: int = 1,
        mutation_rate: Optional[float] = None,
        rng: Optional[random.Random] = None,
    ) -> "Genotype":
        """
        Randomly mutate mutable genotype parameters in place.

        If mutation_rate is provided, every mutable parameter is independently
        mutated with that probability and num_mutations is ignored. Otherwise,
        num_mutations distinct mutable parameters are chosen uniformly.
        """
        random_source = rng if rng is not None else random
        mutable_parameters = self._collect_mutable_parameters()
        if not mutable_parameters:
            return self

        if mutation_rate is not None:
            if not 0.0 <= mutation_rate <= 1.0:
                raise ValueError("mutation_rate must be between 0 and 1")
            selected_parameters = [
                parameter
                for parameter in mutable_parameters
                if random_source.random() < mutation_rate
            ]
        else:
            if num_mutations < 0:
                raise ValueError("num_mutations must be non-negative")
            selected_parameters = random_source.sample(
                mutable_parameters,
                k=min(num_mutations, len(mutable_parameters)),
            )

        for parameter in selected_parameters:
            parameter.mutate(random_source)

        return self

    def _collect_mutable_parameters(self) -> List[_MutableParameter]:
        mutable_parameters: List[_MutableParameter] = []

        for node_name, node in self.nodes.items():
            self._add_positive_tuple_parameters(
                mutable_parameters,
                owner=node,
                field_name="size",
                path=f"nodes.{node_name}.size",
            )
            self._add_relative_tuple_parameters(
                mutable_parameters,
                owner=node,
                field_name="joint_axis",
                path=f"nodes.{node_name}.joint_axis",
                mutation_range=self.AXIS_MUTATION_RANGE,
            )
            self._add_positive_scalar_parameter(
                mutable_parameters,
                owner=node,
                field_name="recursive_limit",
                path=f"nodes.{node_name}.recursive_limit",
            )

            for child_index, connection in enumerate(node.children):
                child_path = f"nodes.{node_name}.children[{child_index}]"
                self._add_relative_tuple_parameters(
                    mutable_parameters,
                    owner=connection,
                    field_name="pos",
                    path=f"{child_path}.pos",
                    mutation_range=self.POS_MUTATION_RANGE,
                )
                self._add_relative_tuple_parameters(
                    mutable_parameters,
                    owner=connection,
                    field_name="axis",
                    path=f"{child_path}.axis",
                    mutation_range=self.AXIS_MUTATION_RANGE,
                )
                self._add_positive_scalar_parameter(
                    mutable_parameters,
                    owner=connection,
                    field_name="scale",
                    path=f"{child_path}.scale",
                )
                self._add_bool_parameter(
                    mutable_parameters,
                    owner=connection,
                    field_name="terminal_only",
                    path=f"{child_path}.terminal_only",
                )
                self._add_bool_parameter(
                    mutable_parameters,
                    owner=connection,
                    field_name="motor_enabled",
                    path=f"{child_path}.motor_enabled",
                )
                self._add_positive_scalar_parameter(
                    mutable_parameters,
                    owner=connection,
                    field_name="motor_gear",
                    path=f"{child_path}.motor_gear",
                )
                self._add_ctrlrange_parameters(
                    mutable_parameters,
                    owner=connection,
                    field_name="ctrlrange",
                    path=f"{child_path}.ctrlrange",
                )
                self._add_positive_scalar_parameter(
                    mutable_parameters,
                    owner=connection,
                    field_name="control_amp",
                    path=f"{child_path}.control_amp",
                )
                self._add_positive_scalar_parameter(
                    mutable_parameters,
                    owner=connection,
                    field_name="control_freq",
                    path=f"{child_path}.control_freq",
                )
                self._add_relative_scalar_parameter(
                    mutable_parameters,
                    owner=connection,
                    field_name="control_phase",
                    path=f"{child_path}.control_phase",
                    mutation_range=self.PHASE_MUTATION_RANGE,
                )
                self._add_relative_scalar_parameter(
                    mutable_parameters,
                    owner=connection,
                    field_name="control_phase_depth_scale",
                    path=f"{child_path}.control_phase_depth_scale",
                    mutation_range=self.PHASE_MUTATION_RANGE,
                )
                self._add_relative_scalar_parameter(
                    mutable_parameters,
                    owner=connection,
                    field_name="control_phase_order_scale",
                    path=f"{child_path}.control_phase_order_scale",
                    mutation_range=self.PHASE_MUTATION_RANGE,
                )

        return mutable_parameters

    def _add_positive_scalar_parameter(
        self,
        mutable_parameters: List[_MutableParameter],
        owner: Any,
        field_name: str,
        path: str,
    ):
        def mutate(random_source: random.Random):
            original_value = getattr(owner, field_name)
            mutated_value = self._mutate_positive_value(original_value, random_source)
            setattr(owner, field_name, mutated_value)

        mutable_parameters.append(_MutableParameter(path, mutate))

    def _add_relative_scalar_parameter(
        self,
        mutable_parameters: List[_MutableParameter],
        owner: Any,
        field_name: str,
        path: str,
        mutation_range: float,
    ):
        def mutate(random_source: random.Random):
            original_value = getattr(owner, field_name)
            delta = random_source.uniform(-mutation_range, mutation_range)
            setattr(owner, field_name, original_value + delta)

        mutable_parameters.append(_MutableParameter(path, mutate))

    def _add_bool_parameter(
        self,
        mutable_parameters: List[_MutableParameter],
        owner: Any,
        field_name: str,
        path: str,
    ):
        def mutate(random_source: random.Random):
            del random_source
            setattr(owner, field_name, not getattr(owner, field_name))

        mutable_parameters.append(_MutableParameter(path, mutate))

    def _add_positive_tuple_parameters(
        self,
        mutable_parameters: List[_MutableParameter],
        owner: Any,
        field_name: str,
        path: str,
    ):
        for index in range(len(getattr(owner, field_name))):
            def mutate(random_source: random.Random, index: int = index):
                values = list(getattr(owner, field_name))
                values[index] = self._mutate_positive_value(values[index], random_source)
                setattr(owner, field_name, tuple(values))

            mutable_parameters.append(_MutableParameter(f"{path}[{index}]", mutate))

    def _add_relative_tuple_parameters(
        self,
        mutable_parameters: List[_MutableParameter],
        owner: Any,
        field_name: str,
        path: str,
        mutation_range: float,
    ):
        for index in range(len(getattr(owner, field_name))):
            def mutate(random_source: random.Random, index: int = index):
                values = list(getattr(owner, field_name))
                delta = random_source.uniform(-mutation_range, mutation_range)
                values[index] += delta
                setattr(owner, field_name, tuple(values))

            mutable_parameters.append(_MutableParameter(f"{path}[{index}]", mutate))

    def _add_ctrlrange_parameters(
        self,
        mutable_parameters: List[_MutableParameter],
        owner: Any,
        field_name: str,
        path: str,
    ):
        for index in range(len(getattr(owner, field_name))):
            def mutate(random_source: random.Random, index: int = index):
                values = list(getattr(owner, field_name))
                delta = random_source.uniform(
                    -self.CTRLRANGE_MUTATION_RANGE,
                    self.CTRLRANGE_MUTATION_RANGE,
                )
                values[index] += delta
                values = sorted(values)
                if values[0] == values[1]:
                    values[0] -= self.CTRLRANGE_MUTATION_RANGE
                    values[1] += self.CTRLRANGE_MUTATION_RANGE
                setattr(owner, field_name, tuple(values))

            mutable_parameters.append(_MutableParameter(f"{path}[{index}]", mutate))

    def _mutate_positive_value(self, value: Any, random_source: random.Random) -> Any:
        value_as_float = float(value)
        delta_range = abs(value_as_float) * 0.05
        delta = random_source.uniform(-delta_range, delta_range)
        mutated_value = max(self.MIN_POSITIVE_VALUE, value_as_float + delta)

        if isinstance(value, int):
            return max(1, int(round(mutated_value)))
        return mutated_value


@dataclass
class ActuatorController:
    motor_name: str
    amp: float
    freq: float
    phase: float


GenotypeSpec = Mapping[str, Mapping[str, Any]]
DEFAULT_GENOTYPE_PATH = Path(__file__).with_name("example_genotype.json")


def build_genotype(root: str, spec: GenotypeSpec) -> Genotype:
    """
    Build a concrete genotype from a compact declarative recipe.

    Each spec entry describes one reusable node type. The optional "children"
    list contains connection dictionaries that are passed to ConnectionGene.
    """
    nodes: Dict[str, NodeGene] = {}

    for node_name, node_spec in spec.items():
        node_kwargs = {
            key: value
            for key, value in node_spec.items()
            if key != "children"
        }
        nodes[node_name] = NodeGene(name=node_name, **node_kwargs)

    if root not in nodes:
        raise KeyError(f"Unknown genotype root: {root}")

    for node_name, node_spec in spec.items():
        children = node_spec.get("children", [])
        for connection_spec in children:
            child_name = connection_spec["child"]
            if child_name not in nodes:
                raise KeyError(
                    f"Node '{node_name}' connects to unknown child '{child_name}'"
                )
            nodes[node_name].children.append(ConnectionGene(**connection_spec))

    return Genotype(root=root, nodes=nodes)


def load_genotype_from_json(path: str | Path) -> Genotype:
    """Load a recursive genotype recipe from a JSON file."""
    with Path(path).open() as f:
        genotype_data = json.load(f)

    return build_genotype(
        root=genotype_data["root"],
        spec=genotype_data["nodes"],
    )


# -----------------------------
# 2. Genotype -> MJCF phenotype
# -----------------------------

class PhenotypeBuilder:
    def __init__(self, genotype: Genotype):
        self.genotype = genotype
        self.body_counter = 0
        self.joint_counter = 0
        self.motor_counter = 0
        self.mujoco_xml: Optional[ET.Element] = None
        self.actuator_xml: Optional[ET.Element] = None
        self.actuator_controllers: List[ActuatorController] = []

    def new_body_name(self, node_name: str) -> str:
        self.body_counter += 1
        return f"{node_name}_{self.body_counter}"

    def new_joint_name(self, node_name: str) -> str:
        self.joint_counter += 1
        return f"{node_name}_joint_{self.joint_counter}"

    def new_motor_name(self, joint_name: str) -> str:
        self.motor_counter += 1
        return f"{joint_name}_motor_{self.motor_counter}"

    def build(self) -> str:
        self.actuator_controllers = []
        self.mujoco_xml = ET.Element("mujoco", model="genotype_creature")
        self.configure_model()

        worldbody = ET.SubElement(self.mujoco_xml, "worldbody")
        self.add_world_elements(worldbody)
        self.actuator_xml = ET.SubElement(self.mujoco_xml, "actuator")

        root_node = self.genotype.nodes[self.genotype.root]
        root_body = self.create_body(worldbody, root_node, "0 0 0.6")

        self.add_node_to_body(
            parent_xml=root_body,
            node=root_node,
            incoming_conn=None,
            current_depths={},
        )

        ET.indent(self.mujoco_xml, space="  ")
        return ET.tostring(self.mujoco_xml, encoding="unicode")

    def configure_model(self):
        if self.mujoco_xml is None:
            raise RuntimeError("mujoco_xml must be initialized before configuring the model")

        compiler = ET.SubElement(self.mujoco_xml, "compiler")
        compiler.set("angle", "degree")

        option = ET.SubElement(self.mujoco_xml, "option")
        option.set("gravity", "0 0 0")
        option.set("density", "1000")
        option.set("viscosity", "0.001")

        self.add_defaults()

    def add_defaults(self):
        if self.mujoco_xml is None:
            raise RuntimeError("mujoco_xml must be initialized before adding defaults")

        default = ET.SubElement(self.mujoco_xml, "default")

        geom_default = ET.SubElement(default, "geom")
        geom_default.set("type", "box")
        geom_default.set("density", "500")
        geom_default.set("friction", "1.0 0.5 0.5")
        geom_default.set("contype", "0")
        geom_default.set("conaffinity", "0")

        joint_default = ET.SubElement(default, "joint")
        joint_default.set("limited", "true")
        joint_default.set("range", "-45 45")
        joint_default.set("damping", "2.0")

        motor_default = ET.SubElement(default, "motor")
        motor_default.set("ctrllimited", "true")

    def add_world_elements(self, worldbody: ET.Element):
        floor = ET.SubElement(worldbody, "geom")
        floor.set("name", "floor")
        floor.set("type", "plane")
        floor.set("size", "5 5 0.1")
        floor.set("pos", "0 0 -0.05")

        light = ET.SubElement(worldbody, "light")
        light.set("pos", "0 0 3")

    def create_body(
        self,
        parent_xml: ET.Element,
        node: NodeGene,
        pos: str,
    ) -> ET.Element:
        body = ET.SubElement(parent_xml, "body")
        body.set("name", self.new_body_name(node.name))
        body.set("pos", pos)
        return body

    def add_node_to_body(
        self,
        parent_xml: ET.Element,
        node: NodeGene,
        incoming_conn: Optional[ConnectionGene],
        current_depths: Dict[str, int],
    ):
        """
        Add one phenotype body part from one genotype node.
        Then recursively add children according to connections.
        """
        node_depth = current_depths.get(node.name, 0) + 1

        self.add_joint(parent_xml, node, incoming_conn, node_depth)
        self.add_geom(parent_xml, node)

        next_depths = self.next_depths(current_depths, node.name, node_depth)
        self.add_children(parent_xml, node, next_depths)

    def add_joint(
        self,
        parent_xml: ET.Element,
        node: NodeGene,
        incoming_conn: Optional[ConnectionGene],
        node_depth: int,
    ):
        if node.joint_type == "free":
            joint = ET.SubElement(parent_xml, "joint")
            joint.set("type", "free")
            joint.set("name", self.new_joint_name(node.name))
        elif node.joint_type == "hinge":
            self.add_hinge_joint(parent_xml, node, incoming_conn, node_depth)

    def add_hinge_joint(
        self,
        parent_xml: ET.Element,
        node: NodeGene,
        incoming_conn: Optional[ConnectionGene],
        node_depth: int,
    ):
        joint_name = self.new_joint_name(node.name)
        joint = ET.SubElement(parent_xml, "joint")
        joint.set("type", "hinge")
        joint.set("name", joint_name)
        joint_axis = incoming_conn.axis if incoming_conn else node.joint_axis
        joint.set("axis", vec_to_str(joint_axis))

        if incoming_conn is None or incoming_conn.motor_enabled:
            self.add_motor(joint_name, incoming_conn, node_depth)

    def add_motor(
        self,
        joint_name: str,
        incoming_conn: Optional[ConnectionGene],
        node_depth: int,
    ):
        if self.actuator_xml is None:
            raise RuntimeError("actuator_xml must be initialized before adding motors")

        motor_gear = incoming_conn.motor_gear if incoming_conn else 10.0
        ctrlrange = incoming_conn.ctrlrange if incoming_conn else (-1.0, 1.0)
        control_amp = incoming_conn.control_amp if incoming_conn else 0.5
        control_freq = incoming_conn.control_freq if incoming_conn else 4.0
        control_order = len(self.actuator_controllers)
        control_phase = (
            incoming_conn.phase_for(node_depth, control_order)
            if incoming_conn
            else 0.0
        )

        motor_name = self.new_motor_name(joint_name)
        motor = ET.SubElement(self.actuator_xml, "motor")
        motor.set("name", motor_name)
        motor.set("joint", joint_name)
        motor.set("gear", str(motor_gear))
        motor.set("ctrlrange", vec_to_str(ctrlrange))

        self.actuator_controllers.append(
            ActuatorController(
                motor_name=motor_name,
                amp=control_amp,
                freq=control_freq,
                phase=control_phase,
            )
        )

    def add_geom(self, parent_xml: ET.Element, node: NodeGene):
        geom = ET.SubElement(parent_xml, "geom")
        geom.set("name", f"{parent_xml.get('name')}_geom")
        geom.set("size", vec_to_str(node.size))
        geom.set("rgba", "0.6 0.7 0.9 1")

    def next_depths(
        self,
        current_depths: Dict[str, int],
        node_name: str,
        node_depth: int,
    ) -> Dict[str, int]:
        next_depths = dict(current_depths)
        next_depths[node_name] = node_depth
        return next_depths

    def add_children(
        self,
        parent_xml: ET.Element,
        node: NodeGene,
        current_depths: Dict[str, int],
    ):
        for conn in node.children:
            child_node = self.genotype.nodes[conn.child]
            if not self.can_add_child(child_node, current_depths):
                continue

            child_body = self.create_body(
                parent_xml,
                child_node,
                vec_to_str(conn.pos),
            )

            self.add_node_to_body(
                parent_xml=child_body,
                node=child_node,
                incoming_conn=conn,
                current_depths=current_depths,
            )

    def can_add_child(
        self,
        child_node: NodeGene,
        current_depths: Dict[str, int],
    ) -> bool:
        child_depth = current_depths.get(child_node.name, 0)
        return child_depth < child_node.recursive_limit


def vec_to_str(v):
    return " ".join(str(x) for x in v)


# -----------------------------
# 3. Show phenotype in MuJoCo
# -----------------------------

def main():
    genotype = load_genotype_from_json(DEFAULT_GENOTYPE_PATH)

    builder = PhenotypeBuilder(genotype)
    mjcf = builder.build()

    with open("generated_creature.xml", "w") as f:
        f.write(mjcf)

    print("Generated MuJoCo model saved to generated_creature.xml")

    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)
    actuator_ids = [
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            controller.motor_name,
        )
        for controller in builder.actuator_controllers
    ]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Simple open-loop controller:
            # drive each motor with its connection gene's control rule.
            t = data.time
            for actuator_id, controller in zip(
                actuator_ids,
                builder.actuator_controllers,
            ):
                data.ctrl[actuator_id] = controller.amp * math.sin(
                    controller.freq * t + controller.phase
                )

            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
