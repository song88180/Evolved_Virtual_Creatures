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
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
import math
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
    motor_gear: float = 10.0
    ctrlrange: Tuple[float, float] = (-1.0, 1.0)
    control_amp: float = 0.5
    control_freq: float = 4.0
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
class Genotype:
    root: str
    nodes: Dict[str, NodeGene]


@dataclass
class ActuatorController:
    motor_name: str
    amp: float
    freq: float
    phase: float


# -----------------------------
# 2. Example genotype
# -----------------------------

def make_example_genotype() -> Genotype:
    """
    A simple recursive genotype:

        body -> segment -> segment -> segment
                      |
                    limb

    The segment node points to itself, so it generates a repeated chain.
    """

    body = NodeGene(
        name="body",
        size=(0.25, 0.15, 0.10),
        joint_type="free",
        recursive_limit=1,
    )

    segment = NodeGene(
        name="segment",
        size=(0.18, 0.08, 0.08),
        joint_type="hinge",
        joint_axis=(0, 1, 0),
        recursive_limit=10,
    )

    limb = NodeGene(
        name="limb",
        size=(0.06, 0.08, 0.06),
        joint_type="hinge",
        joint_axis=(1, 0, 0),
        recursive_limit=1,
    )

    body.children.append(
        ConnectionGene(
            child="segment",
            pos=(0.28, 0.0, 0.0),
            axis=(0, 1, 0),
            control_phase=0.0,
        )
    )

    segment.children.append(
        ConnectionGene(
            child="segment",
            pos=(0.22, 0.0, 0.0),
            axis=(0, 1, 0),
            control_phase_depth_scale=0.7,
        )
    )

    segment.children.append(
        ConnectionGene(
            child="limb",
            pos=(0.0, 0.14, 0.0),
            axis=(1, 0, 0),
            control_phase=1.4,
        )
    )

    segment.children.append(
        ConnectionGene(
            child="limb",
            pos=(0.0, -0.14, 0.0),
            axis=(1, 0, 0),
            control_phase=-1.4,
        )
    )

    return Genotype(
        root="body",
        nodes={
            "body": body,
            "segment": segment,
            "limb": limb,
        },
    )


# -----------------------------
# 3. Genotype -> MJCF phenotype
# -----------------------------

class PhenotypeBuilder:
    def __init__(self, genotype: Genotype):
        self.genotype = genotype
        self.body_counter = 0
        self.joint_counter = 0
        self.motor_counter = 0
        self.actuator_xml = None
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
        mujoco_xml = ET.Element("mujoco", model="genotype_creature")

        compiler = ET.SubElement(mujoco_xml, "compiler")
        compiler.set("angle", "degree")

        option = ET.SubElement(mujoco_xml, "option")
        option.set("gravity", "0 0 0")
        option.set("density", "1000")
        option.set("viscosity", "0.001")

        default = ET.SubElement(mujoco_xml, "default")

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

        worldbody = ET.SubElement(mujoco_xml, "worldbody")

        floor = ET.SubElement(worldbody, "geom")
        floor.set("name", "floor")
        floor.set("type", "plane")
        floor.set("size", "5 5 0.1")
        floor.set("pos", "0 0 -0.05")

        light = ET.SubElement(worldbody, "light")
        light.set("pos", "0 0 3")

        self.actuator_xml = ET.SubElement(mujoco_xml, "actuator")

        root_node = self.genotype.nodes[self.genotype.root]

        root_body = ET.SubElement(worldbody, "body")
        root_body.set("name", self.new_body_name(root_node.name))
        root_body.set("pos", "0 0 0.6")

        self.add_node_to_body(
            parent_xml=root_body,
            node=root_node,
            incoming_conn=None,
            current_depths={},
        )

        ET.indent(mujoco_xml, space="  ")
        return ET.tostring(mujoco_xml, encoding="unicode")

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

        # Add joint
        if node.joint_type == "free":
            joint = ET.SubElement(parent_xml, "joint")
            joint.set("type", "free")
            joint.set("name", self.new_joint_name(node.name))
        elif node.joint_type == "hinge":
            joint_name = self.new_joint_name(node.name)
            joint = ET.SubElement(parent_xml, "joint")
            joint.set("type", "hinge")
            joint.set("name", joint_name)
            joint_axis = incoming_conn.axis if incoming_conn else node.joint_axis
            joint.set("axis", vec_to_str(joint_axis))

            if incoming_conn is None or incoming_conn.motor_enabled:
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

        # Add body geometry
        geom = ET.SubElement(parent_xml, "geom")
        geom.set("name", f"{parent_xml.get('name')}_geom")
        geom.set("size", vec_to_str(node.size))
        geom.set("rgba", "0.6 0.7 0.9 1")

        # Track recursion depth for this genotype node
        current_depths = dict(current_depths)
        current_depths[node.name] = node_depth

        for conn in node.children:
            child_node = self.genotype.nodes[conn.child]

            # Recursive limit: prevent infinite graph expansion
            child_depth = current_depths.get(child_node.name, 0)
            if child_depth >= child_node.recursive_limit:
                continue

            child_body = ET.SubElement(parent_xml, "body")
            child_body.set("name", self.new_body_name(child_node.name))
            child_body.set("pos", vec_to_str(conn.pos))

            self.add_node_to_body(
                parent_xml=child_body,
                node=child_node,
                incoming_conn=conn,
                current_depths=current_depths,
            )


def vec_to_str(v):
    return " ".join(str(x) for x in v)


# -----------------------------
# 4. Show phenotype in MuJoCo
# -----------------------------

def main():
    genotype = make_example_genotype()

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
