from __future__ import annotations

import math
from pathlib import Path

import mujoco

from .evaluation import (
    EvaluationConfig,
    SwimmingEvaluationConfig,
    WalkingEvaluationConfig,
    DISALLOWED_COLLISION_REASON,
    _has_nonparent_self_collision,
    settle_walking_model,
    task_for_config,
)
from .genotype import Genotype
from .graph_analysis import PhenotypeBuildAbort
from .phenotype import ActuatorController, PhenotypeBuilder


def save_x_axis_video(
    genotype: Genotype,
    output_path: Path,
    config: EvaluationConfig,
    fps: int,
    width: int,
    height: int,
    track_root: bool = False,
    speed: float = 1.0,
):
    """Render a swimming or walking evaluation episode to MP4."""
    try:
        import imageio.v3 as iio
    except ImportError as error:
        raise RuntimeError(
            "Saving MP4 video requires imageio and imageio-ffmpeg. Install them with:\n"
            "conda run -n mujoco --no-capture-output python -m pip install imageio imageio-ffmpeg"
        ) from error

    task = task_for_config(config)
    try:
        builder = PhenotypeBuilder(
            genotype,
            max_node=config.max_node,
            task=task,
            self_collision=(
                config.self_collision or config.disallow_collision
            ),
        )
        mjcf = builder.build()
    except PhenotypeBuildAbort as error:
        raise RuntimeError(
            f"Cannot record video because phenotype build failed: {error}"
        ) from error

    model = mujoco.MjModel.from_xml_string(mjcf)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)
    if isinstance(config, WalkingEvaluationConfig):
        failure = settle_walking_model(model, data, config)
        if failure is not None:
            raise RuntimeError(f"Cannot record video: {failure}")
        data.time = 0.0

    actuator_ids = actuator_ids_for_controllers(model, builder.actuator_controllers)
    frames = []
    next_frame_time = 0.0
    frame_interval = speed / fps
    camera = -1

    if track_root:
        root_body_name = f"{genotype.root}_1"
        root_body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            root_body_name,
        )
        if root_body_id < 0:
            raise RuntimeError(
                f"Cannot track root body because {root_body_name!r} was not found."
            )
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = root_body_id

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        while data.time < config.episode_seconds:
            apply_open_loop_controller(
                data=data,
                actuator_ids=actuator_ids,
                actuator_controllers=builder.actuator_controllers,
            )
            mujoco.mj_step(model, data)
            if (
                config.disallow_collision
                and _has_nonparent_self_collision(model, data)
            ):
                raise RuntimeError(DISALLOWED_COLLISION_REASON)
            if data.time >= next_frame_time:
                renderer.update_scene(data, camera=camera)
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
):
    """Compatibility wrapper for callers that explicitly render swimming."""
    return save_x_axis_video(
        genotype, output_path, config, fps, width, height, track_root, speed
    )


def actuator_ids_for_controllers(
    model: mujoco.MjModel,
    actuator_controllers: list[ActuatorController],
) -> list[int]:
    return [
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            controller.motor_name,
        )
        for controller in actuator_controllers
    ]


def apply_open_loop_controller(
    data: mujoco.MjData,
    actuator_ids: list[int],
    actuator_controllers: list[ActuatorController],
):
    for actuator_id, controller in zip(actuator_ids, actuator_controllers):
        data.ctrl[actuator_id] = controller.amp * math.sin(
            controller.freq * data.time + controller.phase
        )
