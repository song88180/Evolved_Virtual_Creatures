"""Shared simulation setup for watching a creature (MP4 video or live viewer).

``prepare_scene`` builds a genotype into a MuJoCo model for a task, places and
settles it exactly as the scored rollout does (``simulation.start_episode``), and tunes
lighting and the floor for display. Scoring rules are not applied: the task's
start rules and volume limits decide fitness, not whether a creature can be
shown, so any creature that can be built is viewable. ``step_scene`` advances one controlled
physics step with the same failure checks as evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import mujoco

from .control import BrainRuntime, apply_controller, create_brain_runtime
from .controller_spec import ActuatorController
from .evolution_tasks.shared import DISALLOWED_COLLISION_REASON, TaskDefinition
from .genotype import Genotype
from .graph_analysis import PhenotypeBuildAbort
from .phenotype import PhenotypeBuilder
from .simulation import (
    has_nonparent_self_collision,
    simulation_failure_reason,
    start_episode,
)

SUN_LIGHT_POS = (0.0, 0.0, 5.0)
SUN_LIGHT_DIR = (0.3, 0.2, -1.0)
SUN_LIGHT_DIFFUSE = (0.9, 0.85, 0.75)
SUN_LIGHT_AMBIENT = (0.15, 0.15, 0.18)
FLOOR_TEXTURE_SCALE = 0.25


class SceneBuildError(RuntimeError):
    """The phenotype could not be built for display."""


class SceneInitializationError(RuntimeError):
    """The task's initialization rejected the creature."""


@dataclass
class Scene:
    model: mujoco.MjModel
    data: mujoco.MjData
    controllers: list[ActuatorController]
    root_body_id: int
    floor_geom_id: int
    brain: BrainRuntime | None = None


def prepare_scene(
    genotype: Genotype,
    definition: TaskDefinition,
    *,
    shadowsize: int = 4096,
    spotlight: bool = False,
    offscreen_size: tuple[int, int] | None = None,
) -> Scene:
    """Build, light, and place a creature exactly as the scored rollout starts."""
    config = definition.config
    environment = definition.environment
    try:
        builder = PhenotypeBuilder(genotype, definition)
        mjcf = builder.build()
    except PhenotypeBuildAbort as error:
        raise SceneBuildError(str(error)) from error

    root_body_name = builder.root_body_name
    mjcf = _track_light(mjcf, root_body_name, spotlight)
    model = mujoco.MjModel.from_xml_string(mjcf)
    _configure_light(model, shadowsize, spotlight)
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_geom_id >= 0:
        floor_material_id = model.geom_matid[floor_geom_id]
        if floor_material_id >= 0:
            model.mat_texrepeat[floor_material_id] *= FLOOR_TEXTURE_SCALE
        # Zero plane extents make MuJoCo render the floor to the horizon.
        # Plane contacts are unbounded either way, so physics is unchanged.
        model.geom_size[floor_geom_id, :2] = 0.0
    if offscreen_size is not None:
        width, height = offscreen_size
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)

    data = mujoco.MjData(model)
    failure = start_episode(model, data, config, environment)
    if failure is not None:
        raise SceneInitializationError(failure)

    # Match the scored rollout's initialization before the controller observes
    # derived state such as body transforms and joint-related quantities.  A
    # fresh MjData does not have all of that state populated until mj_forward
    # (or a step) runs; applying neural controls first can therefore put the
    # display on a different, potentially unstable trajectory from evaluation.
    mujoco.mj_forward(model, data)

    root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
    if root_body_id < 0:
        raise SceneBuildError(f"{root_body_name!r} was not found.")
    brain = create_brain_runtime(
        model, builder.brain_spec, environment.physics_substeps
    )
    return Scene(
        model,
        data,
        builder.actuator_controllers,
        root_body_id,
        floor_geom_id,
        brain,
    )


def step_scene(scene: Scene, definition: TaskDefinition) -> str | None:
    """Apply the controller and advance one control step; return a failure reason, if any.

    A control step is ``environment.physics_substeps`` MuJoCo steps, matching
    how the scored rollout updates controllers.
    """
    config = definition.config
    apply_controller(scene.model, scene.data, scene.controllers, scene.brain)
    for _ in range(definition.environment.physics_substeps):
        previous_time = float(scene.data.time)
        mujoco.mj_step(scene.model, scene.data)
        if config.disallow_collision and has_nonparent_self_collision(
            scene.model, scene.data
        ):
            return DISALLOWED_COLLISION_REASON
        failure = simulation_failure_reason(scene.data, previous_time, config)
        if failure is not None:
            return failure
    return None


def track_root(camera: mujoco.MjvCamera, scene: Scene) -> None:
    """Point a camera at the creature's root body and follow it."""
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = scene.root_body_id


def _configure_light(model: mujoco.MjModel, shadowsize: int, spotlight: bool) -> None:
    if model.nlight < 1:
        return
    light_id = 0
    model.light_type[light_id] = (
        mujoco.mjtLightType.mjLIGHT_SPOT
        if spotlight
        else mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    )
    model.light_castshadow[light_id] = 1
    model.light_pos[light_id] = SUN_LIGHT_POS
    model.light_dir[light_id] = SUN_LIGHT_DIR
    model.light_diffuse[light_id] = SUN_LIGHT_DIFFUSE
    model.light_ambient[light_id] = SUN_LIGHT_AMBIENT
    model.vis.quality.shadowsize = shadowsize


def _track_light(mjcf: str, root_body_name: str, spotlight: bool) -> str:
    """Attach the light to the root while keeping its direction global."""
    xml_root = ET.fromstring(mjcf)
    worldbody = xml_root.find("worldbody")
    root_body = xml_root.find(f"./worldbody/body[@name='{root_body_name}']")
    light = worldbody.find("light") if worldbody is not None else None
    if worldbody is None or root_body is None or light is None:
        return mjcf

    worldbody.remove(light)
    light.set("mode", "track")
    light.set("directional", "false" if spotlight else "true")
    light.set("castshadow", "true")
    light.set("pos", " ".join(map(str, SUN_LIGHT_POS)))
    light.set("dir", " ".join(map(str, SUN_LIGHT_DIR)))
    root_body.insert(0, light)
    return ET.tostring(xml_root, encoding="unicode")
