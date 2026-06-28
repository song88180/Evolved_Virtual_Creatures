from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

from .genes import ATTACHMENT_FACES, SYMMETRY_PLANES
from .genotype import ConnectionGene, Genotype, NodeGene
from .graph_analysis import (
    GenotypeGraphAnalyzer,
    PhenotypeNodeLimitExceeded,
)


FACE_NORMALS = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}

PLANE_REFLECTIONS = {
    "xy": (1.0, 1.0, -1.0),
    "xz": (1.0, -1.0, 1.0),
    "yz": (-1.0, 1.0, 1.0),
}
IDENTITY_REFLECTION = (1.0, 1.0, 1.0)
DEFAULT_ARTICULATED_ROOT_AXIS = (0.0, 1.0, 0.0)


@dataclass
class ActuatorController:
    motor_name: str
    amp: float
    freq: float
    phase: float


# -----------------------------
# 2. Genotype -> MJCF phenotype
# -----------------------------

class PhenotypeBuilder:
    def __init__(
        self,
        genotype: Genotype,
        max_node: int,
        task: str = "swimming_x",
        self_collision: bool = False,
    ):
        if max_node < 1:
            raise ValueError("max_node must be at least 1")
        if task not in {
            "swimming_x",
            "swimming_away",
            "walking_x",
            "walking_away",
            "flying_x",
            "flying_away",
        }:
            raise ValueError(
                "task must be 'swimming_x', 'swimming_away', "
                "'walking_x', 'walking_away', 'flying_x', or 'flying_away'"
            )

        self.genotype = genotype
        self.max_node = max_node
        self.task = task
        self.self_collision = self_collision
        self.body_counter = 0
        self.joint_counter = 0
        self.motor_counter = 0
        self.mujoco_xml: Optional[ET.Element] = None
        self.actuator_xml: Optional[ET.Element] = None
        self.actuator_controllers: List[ActuatorController] = []
        self.control_orders: Dict[Tuple[int, ...], int] = {}

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
        GenotypeGraphAnalyzer(self.genotype, self.max_node).validate()
        self.actuator_controllers = []
        self.control_orders = {}
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
            incoming_axis=None,
            current_depths={},
            connection_orders={},
            effective_size=root_node.size,
            geom_center=(0.0, 0.0, 0.0),
            reflection=IDENTITY_REFLECTION,
            logical_path=(),
        )

        ET.indent(self.mujoco_xml, space="  ")
        return ET.tostring(self.mujoco_xml, encoding="unicode")

    def configure_model(self):
        if self.mujoco_xml is None:
            raise RuntimeError("mujoco_xml must be initialized before configuring the model")

        compiler = ET.SubElement(self.mujoco_xml, "compiler")
        compiler.set("angle", "degree")

        option = ET.SubElement(self.mujoco_xml, "option")
        if self.task in {"swimming_x", "swimming_away"}:
            option.set("gravity", "0 0 0")
            option.set("density", "1000")
            option.set("viscosity", "0.001")
        else:
            option.set("gravity", "0 0 -9.81")
            option.set("density", "0")
            option.set("viscosity", "0")
        option.set("timestep", "0.01")

        self.add_assets()
        self.add_defaults()

    def add_assets(self):
        if self.mujoco_xml is None:
            raise RuntimeError("mujoco_xml must be initialized before adding assets")

        asset = ET.SubElement(self.mujoco_xml, "asset")
        texture = ET.SubElement(asset, "texture")
        texture.set("name", "floor_checker")
        texture.set("type", "2d")
        texture.set("builtin", "checker")
        texture.set("rgb1", "0.172549 0.286275 0.521569")
        texture.set("rgb2", "0.250980 0.458824 0.784314")
        texture.set("width", "512")
        texture.set("height", "512")

        material = ET.SubElement(asset, "material")
        material.set("name", "floor_material")
        material.set("texture", "floor_checker")
        material.set("texrepeat", "10 10")
        material.set("reflectance", "0")


    def add_defaults(self):
        if self.mujoco_xml is None:
            raise RuntimeError("mujoco_xml must be initialized before adding defaults")

        default = ET.SubElement(self.mujoco_xml, "default")

        geom_default = ET.SubElement(default, "geom")
        geom_default.set("type", "box")
        geom_default.set("density", "500")
        if self.task in {"swimming_x", "swimming_away"}:
            geom_default.set("friction", "1.0 0.5 0.5")
            collision_mask = "2" if self.self_collision else "0"
            geom_default.set("contype", collision_mask)
            geom_default.set("conaffinity", collision_mask)
        else:
            geom_default.set("friction", "1 0.005 0.0001")
            geom_default.set("contype", "2")
            geom_default.set(
                "conaffinity", "3" if self.self_collision else "1"
            )

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
        floor.set("material", "floor_material")
        if self.task in {"walking_x", "walking_away", "flying_x", "flying_away"}:
            floor.set("contype", "1")
            floor.set("conaffinity", "2")
        else:
            floor.set("contype", "0")
            floor.set("conaffinity", "0")

        light = ET.SubElement(worldbody, "light")
        light.set("pos", "0 0 3")

    def create_body(
        self,
        parent_xml: ET.Element,
        node: NodeGene,
        pos: str,
    ) -> ET.Element:
        if self.body_counter >= self.max_node:
            raise PhenotypeNodeLimitExceeded(
                "Maximum allowed number of phenotype nodes exceeded while "
                f"building XML: max_node is {self.max_node}."
            )

        body = ET.SubElement(parent_xml, "body")
        body.set("name", self.new_body_name(node.name))
        body.set("pos", pos)
        return body

    def add_node_to_body(
        self,
        parent_xml: ET.Element,
        node: NodeGene,
        incoming_conn: Optional[ConnectionGene],
        incoming_axis: Optional[Tuple[float, float, float]],
        current_depths: Dict[str, int],
        connection_orders: Dict[int, int],
        effective_size: Tuple[float, float, float],
        geom_center: tuple[float, float, float],
        reflection: Tuple[float, float, float],
        logical_path: Tuple[int, ...],
    ):
        """
        Add one phenotype body part from one genotype node.
        Then recursively add children according to connections.
        """
        node_depth = current_depths.get(node.name, 0) + 1

        self.add_joint(
            parent_xml,
            node,
            incoming_conn,
            incoming_axis,
            node_depth,
            logical_path,
        )
        self.add_geom(parent_xml, effective_size, geom_center)

        next_depths = self.next_depths(current_depths, node.name, node_depth)
        self.add_children(
            parent_xml,
            node,
            next_depths,
            connection_orders,
            effective_size,
            geom_center,
            reflection,
            logical_path,
        )

    def add_joint(
        self,
        parent_xml: ET.Element,
        node: NodeGene,
        incoming_conn: Optional[ConnectionGene],
        incoming_axis: Optional[Tuple[float, float, float]],
        node_depth: int,
        logical_path: Tuple[int, ...],
    ):
        if node.joint_type == "free":
            joint = ET.SubElement(parent_xml, "joint")
            joint.set("type", "free")
            joint.set("name", self.new_joint_name(node.name))
        elif node.joint_type in {"hinge", "slide", "ball"}:
            self.add_articulated_joint(
                parent_xml,
                node,
                incoming_conn,
                incoming_axis,
                node_depth,
                logical_path,
            )

    def add_articulated_joint(
        self,
        parent_xml: ET.Element,
        node: NodeGene,
        incoming_conn: Optional[ConnectionGene],
        incoming_axis: Optional[Tuple[float, float, float]],
        node_depth: int,
        logical_path: Tuple[int, ...],
    ):
        joint_name = self.new_joint_name(node.name)
        joint = ET.SubElement(parent_xml, "joint")
        joint.set("type", node.joint_type)
        joint.set("name", joint_name)
        joint.set("pos", "0 0 0")
        joint_axis = (
            incoming_axis
            if incoming_axis is not None
            else DEFAULT_ARTICULATED_ROOT_AXIS
        )
        if node.joint_type != "ball":
            joint.set("axis", vec_to_str(joint_axis))
        if node.joint_type == "slide":
            joint.set("range", "-0.5 0.5")
        else:
            joint.set("range", "0 45")

        if incoming_conn is None or incoming_conn.motor_enabled:
            self.add_motor(
                joint_name=joint_name,
                joint_type=node.joint_type,
                joint_axis=joint_axis,
                incoming_conn=incoming_conn,
                node_depth=node_depth,
                logical_path=logical_path,
            )

    def add_motor(
        self,
        joint_name: str,
        joint_type: str,
        joint_axis: Sequence[float],
        incoming_conn: Optional[ConnectionGene],
        node_depth: int,
        logical_path: Tuple[int, ...],
    ):
        if self.actuator_xml is None:
            raise RuntimeError("actuator_xml must be initialized before adding motors")

        motor_gear = incoming_conn.motor_gear if incoming_conn else 10.0
        ctrlrange = incoming_conn.ctrlrange if incoming_conn else (-1.0, 1.0)
        control_amp = incoming_conn.control_amp if incoming_conn else 0.5
        control_freq = incoming_conn.control_freq if incoming_conn else 4.0
        control_order = self.control_orders.setdefault(
            logical_path,
            len(self.control_orders),
        )
        control_phase = (
            incoming_conn.phase_for(node_depth, control_order)
            if incoming_conn
            else 0.0
        )

        motor_name = self.new_motor_name(joint_name)
        motor = ET.SubElement(self.actuator_xml, "motor")
        motor.set("name", motor_name)
        motor.set("joint", joint_name)
        if joint_type == "ball":
            normalized_axis = _normalized_vector(joint_axis)
            motor.set(
                "gear",
                vec_to_str(component * motor_gear for component in normalized_axis),
            )
        else:
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

    def add_geom(
        self,
        parent_xml: ET.Element,
        size: Sequence[float],
        geom_center: tuple[float, float, float],
    ):
        geom = ET.SubElement(parent_xml, "geom")
        geom.set("name", f"{parent_xml.get('name')}_geom")
        geom.set("size", vec_to_str(size))
        geom.set("pos", vec_to_str(geom_center))
        geom.set("rgba", "0.933333 0.603922 0.301961 1")

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
        connection_orders: Dict[int, int],
        effective_size: Tuple[float, float, float],
        geom_center: tuple[float, float, float],
        reflection: Tuple[float, float, float],
        logical_path: Tuple[int, ...],
    ):
        for connection_index, conn in enumerate(node.children):
            if conn.terminal_only and not self.is_terminal_node(node, current_depths):
                continue

            child_node = self.genotype.nodes[conn.child]
            if not self.can_add_child(child_node, current_depths):
                continue

            connection_key = id(conn)
            connection_order = connection_orders.get(connection_key, 0) + 1
            child_size = tuple(
                dimension * conn.scale ** (connection_order - 1)
                for dimension in child_node.size
            )
            child_connection_orders = dict(connection_orders)
            child_connection_orders[connection_key] = connection_order
            base_body_pos, base_geom_center = _surface_attachment_transform(
                parent_size=effective_size,
                parent_geom_center=geom_center,
                child_size=child_size,
                parent_face=conn.parent_face,
                surface_uv=conn.surface_uv,
            )
            child_logical_path = (*logical_path, connection_index)
            for local_reflection in _symmetry_reflections(conn.symmetry):
                child_reflection = _compose_reflections(
                    reflection,
                    local_reflection,
                )
                child_body_pos = _reflect_point(
                    base_body_pos,
                    geom_center,
                    child_reflection,
                )
                child_geom_center = _reflect_vector(
                    base_geom_center,
                    child_reflection,
                )
                child_axis = (
                    _reflect_vector(conn.axis, child_reflection)
                    if child_node.joint_type == "slide"
                    else _reflect_axial_vector(conn.axis, child_reflection)
                )
                child_body = self.create_body(
                    parent_xml,
                    child_node,
                    vec_to_str(child_body_pos),
                )

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

    def is_terminal_node(
        self,
        node: NodeGene,
        current_depths: Dict[str, int],
    ) -> bool:
        return current_depths.get(node.name, 0) >= node.recursive_limit

    def can_add_child(
        self,
        child_node: NodeGene,
        current_depths: Dict[str, int],
    ) -> bool:
        child_depth = current_depths.get(child_node.name, 0)
        return child_depth < child_node.recursive_limit


def _symmetry_reflections(
    symmetry: Sequence[str],
) -> List[Tuple[float, float, float]]:
    reflections = [IDENTITY_REFLECTION]
    for plane in SYMMETRY_PLANES:
        if plane not in symmetry:
            continue
        plane_reflection = PLANE_REFLECTIONS[plane]
        reflections += [
            _compose_reflections(reflection, plane_reflection)
            for reflection in reflections
        ]
    return reflections


def _compose_reflections(
    first: Sequence[float],
    second: Sequence[float],
) -> Tuple[float, float, float]:
    return tuple(a * b for a, b in zip(first, second))


def _reflect_point(
    point: Sequence[float],
    origin: Sequence[float],
    reflection: Sequence[float],
) -> Tuple[float, float, float]:
    return tuple(
        center + sign * (coordinate - center)
        for coordinate, center, sign in zip(point, origin, reflection)
    )


def _reflect_vector(
    vector: Sequence[float],
    reflection: Sequence[float],
) -> Tuple[float, float, float]:
    return tuple(component * sign for component, sign in zip(vector, reflection))


def _reflect_axial_vector(
    vector: Sequence[float],
    reflection: Sequence[float],
) -> Tuple[float, float, float]:
    determinant = reflection[0] * reflection[1] * reflection[2]
    return tuple(
        determinant * component * sign
        for component, sign in zip(vector, reflection)
    )


def _surface_attachment_transform(
    parent_size: Sequence[float],
    parent_geom_center: Sequence[float],
    child_size: Sequence[float],
    parent_face: str,
    surface_uv: Sequence[float],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    try:
        normal = FACE_NORMALS[parent_face]
    except KeyError as exc:
        valid_faces = ", ".join(ATTACHMENT_FACES)
        raise ValueError(
            f"Unknown parent face {parent_face!r}; expected one of {valid_faces}"
        ) from exc

    normal_axis = next(index for index, value in enumerate(normal) if value)
    tangent_axes = [axis for axis in range(3) if axis != normal_axis]
    joint_pos = list(parent_geom_center)
    joint_pos[normal_axis] += normal[normal_axis] * parent_size[normal_axis]
    for coordinate, axis in zip(surface_uv, tangent_axes):
        joint_pos[axis] += coordinate * parent_size[axis]

    child_geom_center = tuple(
        normal[axis] * child_size[axis]
        for axis in range(3)
    )
    return tuple(joint_pos), child_geom_center


def _normalized_vector(vector: Sequence[float]) -> Tuple[float, float, float]:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        return (0.0, 1.0, 0.0)
    return tuple(component / norm for component in vector)

def vec_to_str(v):
    return " ".join(str(x) for x in v)
