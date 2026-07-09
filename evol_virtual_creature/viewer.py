from __future__ import annotations

import mujoco
import mujoco.viewer

from .control import ActuatorController, actuator_ids_for_controllers, apply_controller


def launch_viewer(mjcf: str, actuator_controllers: list[ActuatorController]):
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)
    actuator_ids_for_controllers(model, actuator_controllers)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            apply_controller(model, data, actuator_controllers)
            mujoco.mj_step(model, data)
            viewer.sync()
