from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from .evaluation import (
    _has_nonparent_self_collision,
    initialize_model,
    simulation_failure_reason,
)
from .evolution_tasks.shared import (
    DISALLOWED_COLLISION_REASON,
    EnvironmentFamily,
    EvaluationConfig,
    TaskDefinition,
)
from .evolution_tasks.swimming_x import (
    SwimmingEvaluationConfig,
    TASK_DEFINITION as SWIMMING_X_TASK,
)
from .genotype import Genotype
from .graph_analysis import PhenotypeBuildAbort
from .control import (
    ActuatorController,
    actuator_ids_for_controllers,
    apply_controller,
    apply_open_loop_controller as _apply_open_loop_controller,
)
from .phenotype import PhenotypeBuilder

_VIDEO_SUN_LIGHT_POS = (0.0, 0.0, 5.0)
_VIDEO_SUN_LIGHT_DIR = (0.3, 0.2, -1.0)
_VIDEO_SUN_LIGHT_DIFFUSE = (0.9, 0.85, 0.75)
_VIDEO_SUN_LIGHT_AMBIENT = (0.15, 0.15, 0.18)
_VIDEO_FLOOR_TEXTURE_SCALE = 0.25


def _configure_video_light(
    model: mujoco.MjModel,
    shadowsize: int,
    spotlight: bool,
) -> None:
    if model.nlight < 1:
        return

    light_id = 0
    model.light_type[light_id] = (
        mujoco.mjtLightType.mjLIGHT_SPOT
        if spotlight
        else mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    )
    model.light_castshadow[light_id] = 1
    model.light_pos[light_id] = _VIDEO_SUN_LIGHT_POS
    model.light_dir[light_id] = _VIDEO_SUN_LIGHT_DIR
    model.light_diffuse[light_id] = _VIDEO_SUN_LIGHT_DIFFUSE
    model.light_ambient[light_id] = _VIDEO_SUN_LIGHT_AMBIENT
    model.vis.quality.shadowsize = shadowsize


def _track_video_light(
    mjcf: str,
    root_body_name: str,
    spotlight: bool,
) -> str:
    """Attach the video light to the root while keeping its direction global."""
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
    light.set("pos", " ".join(map(str, _VIDEO_SUN_LIGHT_POS)))
    light.set("dir", " ".join(map(str, _VIDEO_SUN_LIGHT_DIR)))
    root_body.insert(0, light)
    return ET.tostring(xml_root, encoding="unicode")


def _prevent_floor_shadow_casting(
    scene: mujoco.MjvScene,
    floor_geom_id: int,
) -> None:
    """Exclude the rendered floor from the shadow-map casting pass."""
    for scene_geom_id in range(scene.ngeom):
        scene_geom = scene.geoms[scene_geom_id]
        if (
            scene_geom.objtype == mujoco.mjtObj.mjOBJ_GEOM
            and scene_geom.objid == floor_geom_id
        ):
            # This changes only the temporary visual scene. The floor remains
            # unchanged in MjModel, so walking contacts and friction still work.
            scene_geom.category = mujoco.mjtCatBit.mjCAT_DECOR
            return


def save_x_axis_video(
    genotype: Genotype,
    output_path: Path,
    definition: TaskDefinition,
    config: EvaluationConfig,
    fps: int,
    width: int,
    height: int,
    track_root: bool = False,
    speed: float = 1.0,
    shadowsize: int = 4096,
    spotlight: bool = False,
    camera_circle_around: bool = False,
    environment: EnvironmentFamily | None = None,
):
    """Render a selected swimming, walking, or flying task episode to MP4."""
    environment = definition.environment if environment is None else environment
    try:
        import imageio.v3 as iio
    except ImportError as error:
        raise RuntimeError(
            "Saving MP4 video requires imageio and imageio-ffmpeg. Install them with:\n"
            "conda run -n mujoco --no-capture-output python -m pip install imageio imageio-ffmpeg"
        ) from error

    try:
        builder = PhenotypeBuilder(
            genotype,
            max_node=config.max_node,
            environment=environment,
            self_collision=(
                config.self_collision or config.disallow_collision
            ),
        )
        mjcf = builder.build()
    except PhenotypeBuildAbort as error:
        raise RuntimeError(
            f"Cannot record video because phenotype build failed: {error}"
        ) from error

    root_body_name = f"{genotype.root}_1"
    mjcf = _track_video_light(mjcf, root_body_name, spotlight)
    model = mujoco.MjModel.from_xml_string(mjcf)
    _configure_video_light(model, shadowsize, spotlight)
    floor_geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "floor",
    )
    if floor_geom_id >= 0:
        floor_material_id = model.geom_matid[floor_geom_id]
        if floor_material_id >= 0:
            model.mat_texrepeat[floor_material_id] *= _VIDEO_FLOOR_TEXTURE_SCALE
        # Zero plane extents make MuJoCo render the floor to the horizon.
        # This model exists only for video generation.
        model.geom_size[floor_geom_id, :2] = 0.0
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)
    failure = initialize_model(model, data, config, environment)
    if failure is not None:
        raise RuntimeError(f"Cannot record video: {failure}")

    # Match the scored rollout's initialization before the controller observes
    # derived state such as body transforms and joint-related quantities.  A
    # fresh MjData does not have all of that state populated until mj_forward
    # (or a step) runs; applying neural controls first can therefore put the
    # video on a different, potentially unstable trajectory from evaluation.
    mujoco.mj_forward(model, data)

    actuator_ids = actuator_ids_for_controllers(model, builder.actuator_controllers)
    frames = []
    next_frame_time = 0.0
    frame_interval = speed / fps
    camera = -1

    root_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        root_body_name,
    )
    if root_body_id < 0:
        raise RuntimeError(
            f"Cannot render video because {root_body_name!r} was not found."
        )

    if track_root or camera_circle_around:
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = root_body_id
        initial_camera_azimuth = float(camera.azimuth)

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        while data.time < config.episode_seconds:
            previous_time = float(data.time)
            apply_controller(model, data, builder.actuator_controllers)
            mujoco.mj_step(model, data)
            if (
                config.disallow_collision
                and _has_nonparent_self_collision(model, data)
            ):
                raise RuntimeError(DISALLOWED_COLLISION_REASON)
            failure = simulation_failure_reason(data, previous_time, config)
            if failure is not None:
                raise RuntimeError(f"Cannot record video: {failure}")
            if data.time >= next_frame_time:
                if camera_circle_around:
                    camera.azimuth = initial_camera_azimuth + (
                        360.0 * data.time / config.episode_seconds
                    )
                renderer.update_scene(data, camera=camera)
                if floor_geom_id >= 0:
                    _prevent_floor_shadow_casting(
                        renderer.scene,
                        floor_geom_id,
                    )
                frames.append(renderer.render())
                next_frame_time += frame_interval

    try:
        iio.imwrite(output_path, frames, fps=fps)
    except OSError as error:
        raise RuntimeError(
            "Saving MP4 video requires imageio and imageio-ffmpeg. Install them with:\n"
            "conda run -n mujoco --no-capture-output python -m pip install imageio imageio-ffmpeg"
        ) from error


def save_x_axis_swimming_video(
    genotype: Genotype,
    output_path: Path,
    config: SwimmingEvaluationConfig,
    fps: int,
    width: int,
    height: int,
    track_root: bool = False,
    speed: float = 1.0,
    shadowsize: int = 4096,
    spotlight: bool = False,
    camera_circle_around: bool = False,
):
    """Compatibility wrapper for callers that explicitly render swimming."""
    return save_x_axis_video(
        genotype,
        output_path,
        SWIMMING_X_TASK,
        config,
        fps,
        width,
        height,
        track_root,
        speed,
        shadowsize,
        spotlight,
        camera_circle_around,
    )


def apply_open_loop_controller(
    data: mujoco.MjData,
    actuator_ids: list[int],
    actuator_controllers: list[ActuatorController],
):
    _apply_open_loop_controller(data, actuator_ids, actuator_controllers)
