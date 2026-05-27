from __future__ import annotations

import math
from pathlib import Path

import mujoco

from .evaluation import SwimmingEvaluationConfig
from .genotype import Genotype
from .graph_analysis import PhenotypeBuildAbort
from .phenotype import ActuatorController, PhenotypeBuilder


def save_x_axis_swimming_video(
    genotype: Genotype,
    output_path: Path,
    config: SwimmingEvaluationConfig,
    fps: int,
    width: int,
    height: int,
):
    try:
        import imageio.v3 as iio
    except ImportError as error:
        raise RuntimeError(
            "Saving MP4 video requires imageio and imageio-ffmpeg. Install them with:\n"
            "conda run -n mujoco --no-capture-output python -m pip install imageio imageio-ffmpeg"
        ) from error

    try:
        builder = PhenotypeBuilder(genotype, max_node=config.max_node)
        mjcf = builder.build()
    except PhenotypeBuildAbort as error:
        raise RuntimeError(f"Cannot record video because phenotype build failed: {error}") from error

    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)
    actuator_ids = actuator_ids_for_controllers(model, builder.actuator_controllers)
    frames = []
    next_frame_time = 0.0
    frame_interval = 1.0 / fps

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        while data.time < config.episode_seconds:
            apply_open_loop_controller(
                data=data,
                actuator_ids=actuator_ids,
                actuator_controllers=builder.actuator_controllers,
            )
            mujoco.mj_step(model, data)

            if data.time >= next_frame_time:
                renderer.update_scene(data)
                frames.append(renderer.render())
                next_frame_time += frame_interval

    iio.imwrite(output_path, frames, fps=fps)


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
