from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mujoco

from .evolution_tasks.shared import (
    DISALLOWED_COLLISION_REASON,
    TaskDefinition,
)
from .evolution_tasks.swimming_x import (
    SwimmingEvaluationConfig,
    TASK_DEFINITION as SWIMMING_X_TASK,
)
from .genotype import Genotype
from .control import (
    ActuatorController,
    apply_open_loop_controller as _apply_open_loop_controller,
)
from .scene import (
    SceneBuildError,
    SceneInitializationError,
    prepare_scene,
    step_scene,
    track_root as _track_root_camera,
)

_IMAGEIO_REQUIRED = (
    "Saving MP4 video requires imageio and imageio-ffmpeg. Install them with:\n"
    "conda run -n mujoco --no-capture-output python -m pip install imageio imageio-ffmpeg"
)


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
    fps: int,
    width: int,
    height: int,
    track_root: bool = False,
    speed: float = 1.0,
    shadowsize: int = 4096,
    spotlight: bool = False,
    camera_circle_around: bool = False,
):
    """Render a selected swimming, walking, or flying task episode to MP4.

    The recording lasts ``definition.config.episode_seconds`` of simulated time.
    """
    config = definition.config
    try:
        import imageio.v3 as iio
    except ImportError as error:
        raise RuntimeError(_IMAGEIO_REQUIRED) from error

    try:
        scene = prepare_scene(
            genotype,
            definition,
            shadowsize=shadowsize,
            spotlight=spotlight,
            offscreen_size=(width, height),
        )
    except SceneBuildError as error:
        raise RuntimeError(
            f"Cannot record video because phenotype build failed: {error}"
        ) from error
    except SceneInitializationError as error:
        raise RuntimeError(f"Cannot record video: {error}") from error

    frames = []
    next_frame_time = 0.0
    frame_interval = speed / fps
    camera = -1
    if track_root or camera_circle_around:
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(scene.model, camera)
        _track_root_camera(camera, scene)
        initial_camera_azimuth = float(camera.azimuth)

    data = scene.data
    with mujoco.Renderer(scene.model, height=height, width=width) as renderer:
        while data.time < config.episode_seconds:
            failure = step_scene(scene, definition)
            if failure == DISALLOWED_COLLISION_REASON:
                raise RuntimeError(failure)
            if failure is not None:
                raise RuntimeError(f"Cannot record video: {failure}")
            if data.time >= next_frame_time:
                if camera_circle_around:
                    camera.azimuth = initial_camera_azimuth + (
                        360.0 * data.time / config.episode_seconds
                    )
                renderer.update_scene(data, camera=camera)
                if scene.floor_geom_id >= 0:
                    _prevent_floor_shadow_casting(
                        renderer.scene,
                        scene.floor_geom_id,
                    )
                frames.append(renderer.render())
                next_frame_time += frame_interval

    try:
        iio.imwrite(output_path, frames, fps=fps)
    except OSError as error:
        raise RuntimeError(_IMAGEIO_REQUIRED) from error


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
        replace(SWIMMING_X_TASK, config=config),
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
