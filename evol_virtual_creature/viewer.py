from __future__ import annotations

import math

import mujoco
import mujoco.viewer

from .phenotype import ActuatorController


def launch_viewer(mjcf: str, actuator_controllers: list[ActuatorController]):
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)
    actuator_ids = [
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            controller.motor_name,
        )
        for controller in actuator_controllers
    ]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Simple open-loop controller:
            # drive each motor with its connection gene's control rule.
            t = data.time
            for actuator_id, controller in zip(
                actuator_ids,
                actuator_controllers,
            ):
                data.ctrl[actuator_id] = controller.amp * math.sin(
                    controller.freq * t + controller.phase
                )

            mujoco.mj_step(model, data)
            viewer.sync()
