"""Tests for evc3d.simulation: episode start (placement and settling) and shared mechanics."""

from dataclasses import replace
from functools import partial

import mujoco
import numpy as np
import pytest

from evc3d.evaluation import start_rule_failure
from evc3d.evolution_tasks.flying_x import (
    FlyingEvaluationConfig,
    TASK_ENVIRONMENT as FLYING_X_ENVIRONMENT,
)
from evc3d.evolution_tasks.walking_x import (
    WalkingEvaluationConfig,
    TASK_ENVIRONMENT as WALKING_X_ENVIRONMENT,
)
from evc3d.genotype_io import build_genotype
from evc3d.phenotype import PhenotypeBuilder
from evc3d.simulation import start_episode
import evc3d.simulation as simulation
from helpers import make_definition, creature_geom_ids, placed_walker


def test_flying_initialization_raises_low_creature_above_floor():
    genotype = build_genotype(
        root="body",
        spec={
            "body": {
                "size": (0.2, 0.2, 0.8),
                "joint_type": "free",
            }
        },
    )
    model = mujoco.MjModel.from_xml_string(
        PhenotypeBuilder(
            genotype,
            make_definition(
                max_body_count=500, environment=FLYING_X_ENVIRONMENT,
            ),
        ).build()
    )
    data = mujoco.MjData(model)

    assert start_episode(
        model, data, FlyingEvaluationConfig(), FLYING_X_ENVIRONMENT,
        start_check=start_rule_failure,
    ) is None

    floor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
    )
    body_geom_id = creature_geom_ids(model)[0]
    lowest_z = data.geom_xpos[body_geom_id, 2] - model.geom_size[body_geom_id, 2]
    assert lowest_z == pytest.approx(data.geom_xpos[floor_id, 2] + 5.0)


def test_walking_initialization_raises_low_creature_above_floor():
    genotype = build_genotype(
        root="body",
        spec={
            "body": {
                "size": (0.2, 0.2, 0.8),
                "joint_type": "free",
            }
        },
    )
    model = mujoco.MjModel.from_xml_string(
        PhenotypeBuilder(
            genotype,
            make_definition(
                max_body_count=500, environment=WALKING_X_ENVIRONMENT,
            ),
        ).build()
    )
    data = mujoco.MjData(model)

    assert start_episode(
        model, data, WalkingEvaluationConfig(), WALKING_X_ENVIRONMENT,
        start_check=start_rule_failure,
    ) is None

    floor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
    )
    body_geom_id = creature_geom_ids(model)[0]
    lowest_z = data.geom_xpos[body_geom_id, 2] - model.geom_size[body_geom_id, 2]
    # Generic initialization applies clearance and then the configured settling pass.
    assert lowest_z == pytest.approx(data.geom_xpos[floor_id, 2], abs=2e-4)
    assert all(float(contact.dist) >= -2e-4 for contact in data.contact)


def test_skipping_start_rules_does_not_change_the_start_state():
    definition = make_definition("walking_x")
    states = []
    scored_start = partial(start_episode, start_check=start_rule_failure)
    for start in (scored_start, start_episode):  # scored start vs viewing start
        model, data = placed_walker(definition)
        assert start(model, data, definition.config, definition.environment) is None
        states.append((data.time, data.qpos.copy(), data.qvel.copy()))

    assert states[0][0] == states[1][0]
    np.testing.assert_array_equal(states[0][1], states[1][1])
    np.testing.assert_array_equal(states[0][2], states[1][2])


def test_floor_clearance_without_free_root_joint_reports_reason():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="floor" type="plane" size="1 1 0.1"/>
            <body name="body_1" pos="0 0 0.5"><geom type="box" size="0.1 0.1 0.1"/></body>
          </worldbody>
        </mujoco>
        """
    )
    environment = replace(WALKING_X_ENVIRONMENT, initial_floor_clearance=0.05)

    failure = simulation.place_creature(
        model, mujoco.MjData(model), WalkingEvaluationConfig(), environment
    )

    assert failure == "walking_x creature has no free root joint."
