"""Interactive MuJoCo viewer for watching a creature perform a task."""

from __future__ import annotations

import time

import mujoco
import mujoco.viewer

from .evolution_tasks.shared import TaskDefinition
from .genotype import Genotype
from .scene import prepare_scene, step_scene, track_root as _track_root_camera


def launch_viewer(
    genotype: Genotype,
    definition: TaskDefinition,
    *,
    track_root: bool = False,
    speed: float = 1.0,
    shadowsize: int = 4096,
    spotlight: bool = False,
) -> None:
    """Simulate a genotype in its task environment until the window is closed.

    The scene starts exactly like the scored rollout (task initialization and
    controller), runs past the task's episode length, and is paced so that
    simulated time advances at ``speed`` times real time. If the simulation
    fails (e.g. becomes numerically unstable), stepping stops, the reason is
    printed, and the window stays open on the last valid state.
    """
    scene = prepare_scene(
        genotype,
        definition,
        shadowsize=shadowsize,
        spotlight=spotlight,
    )
    data = scene.data

    with mujoco.viewer.launch_passive(scene.model, data) as viewer:
        if track_root:
            _track_root_camera(viewer.cam, scene)
        wall_start = time.perf_counter()
        sim_start = float(data.time)
        stopped = False
        while viewer.is_running():
            if not stopped:
                failure = step_scene(scene, definition)
                if failure is not None:
                    print(f"Simulation stopped at t={data.time:.2f}s: {failure}")
                    stopped = True
            viewer.sync()
            if stopped:
                time.sleep(0.02)
                continue
            ahead = (data.time - sim_start) / speed - (time.perf_counter() - wall_start)
            if ahead > 0.0:
                time.sleep(ahead)
