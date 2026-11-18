"""Tests for evc3d view, evc3d.viewer, and evc3d.scene: interactive viewing and MP4 option handling."""

from pathlib import Path

import mujoco
import pytest

from evc3d import cli
from evc3d.cli import evolve as evolve_cli, view as view_cli
from evc3d.evaluation import MAXIMUM_CREATURE_HEIGHT_REASON, evaluate_task
import evc3d.evaluation as evaluation
from evc3d.evolution_tasks import task_definition
from evc3d.evolution_tasks.swimming_x import SwimmingEvaluationConfig
from evc3d.genotype_io import load_genotype_from_json
from helpers import make_definition, GENOTYPE_PATH


class _FakeViewer:
    """Stands in for mujoco.viewer.launch_passive; runs a fixed number of frames."""

    def __init__(self, frames):
        self.frames = frames
        self.syncs = 0
        self.cam = mujoco.MjvCamera()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def is_running(self):
        return self.syncs < self.frames

    def sync(self):
        self.syncs += 1


def test_view_accepts_shadow_rendering_options():
    args = view_cli.parse_args(
        ["--mp4", "--shadowsize", "8192", "--spotlight", "--camera-circle-around"]
    )
    assert args.shadowsize == 8192
    assert args.spotlight
    assert args.camera_circle_around


def test_view_rejects_nonpositive_shadowsize(capsys):
    with pytest.raises(SystemExit, match="2"):
        view_cli.parse_args(["--shadowsize", "0"])
    assert "--shadowsize must be greater than zero" in capsys.readouterr().err


def test_view_defaults_to_interactive_with_mp4_defaults_unset():
    args = view_cli.parse_args([])
    assert args.mp4 is None
    assert args.duration is None and args.fps is None


@pytest.mark.parametrize(
    "argv",
    [["--duration", "3"], ["--fps", "10"], ["--width", "100"], ["--camera-circle-around"]],
)
def test_view_rejects_mp4_options_without_mp4(argv, capsys):
    with pytest.raises(SystemExit, match="2"):
        view_cli.parse_args(argv)
    assert "only applies with --mp4" in capsys.readouterr().err


def test_view_mp4_fills_video_defaults():
    args = view_cli.parse_args(["--mp4", "out.mp4"])
    assert args.mp4 == Path("out.mp4")
    assert (args.duration, args.fps, args.width, args.height) == (10.0, 30, 960, 544)


def test_view_mp4_without_path_uses_task_default(monkeypatch, tmp_path):
    recorded = {}

    def fake_video(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr("evc3d.video.save_x_axis_video", fake_video)
    monkeypatch.chdir(tmp_path)
    cli.main(["view", "--task", "walking_x", "--mp4", "--duration", "2"])

    assert recorded["output_path"] == Path("example_walking_x.mp4")
    assert recorded["definition"].config.episode_seconds == 2.0


def test_interactive_view_steps_past_episode_and_tracks_root(monkeypatch):
    import evc3d.viewer as viewer_module

    fake = _FakeViewer(frames=5)
    started = {}

    def fake_launch(model, data):
        started["time"] = float(data.time)
        started["data"] = data
        return fake

    monkeypatch.setattr(viewer_module.mujoco.viewer, "launch_passive", fake_launch)
    monkeypatch.setattr(viewer_module.time, "sleep", lambda _seconds: None)
    genotype = evolve_cli.load_genotype_from_json(evolve_cli.DEFAULT_GENOTYPE_PATH)
    definition = task_definition("swimming_x")
    definition.config = SwimmingEvaluationConfig(episode_seconds=0.02)

    viewer_module.launch_viewer(genotype, definition, track_root=True, speed=1000.0)

    assert fake.syncs == 5
    # Five control steps are taken even though the episode is only two steps long.
    timestep = definition.environment.timestep
    assert started["data"].time == pytest.approx(started["time"] + 5 * timestep)
    assert fake.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING
    assert fake.cam.trackbodyid >= 1


def test_interactive_view_reports_unbuildable_creature(monkeypatch, capsys):
    from evc3d.scene import SceneBuildError

    def fail(*_args, **_kwargs):
        raise SceneBuildError("too many nodes")

    monkeypatch.setattr("evc3d.viewer.launch_viewer", fail)
    with pytest.raises(SystemExit, match="1"):
        cli.main(["view"])
    assert "Cannot view creature: too many nodes" in capsys.readouterr().out


def test_view_shows_creature_that_scoring_rejects_at_start():
    from evc3d.scene import prepare_scene

    genotype = load_genotype_from_json(GENOTYPE_PATH)
    definition = make_definition("walking_x", max_creature_height=0.01)

    result = evaluate_task(genotype, definition)
    assert result.build_failed
    assert result.failure_reason.startswith(evaluation.MAXIMUM_CREATURE_HEIGHT_REASON)

    scene = prepare_scene(genotype, definition)
    assert scene.data.time == 0.0
