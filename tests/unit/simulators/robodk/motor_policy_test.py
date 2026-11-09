# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tbp.monty.frameworks.actions.action_samplers import ConstantSampler
from tbp.monty.frameworks.actions.actions import (
    MoveForward,
    OrientHorizontal,
    OrientVertical,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.motor_system_state import MotorSystemState
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.simulators.robodk.motor_policy import (
    RoboDKSurfacePolicyCurvatureInformed,
)


def make_policy(max_consecutive_rejections=3):
    return RoboDKSurfacePolicyCurvatureInformed(
        agent_id=AgentID("agent_id_0"),
        sensor_id=SensorID("patch"),
        action_sampler=ConstantSampler(
            actions=[MoveForward, OrientHorizontal, OrientVertical],
            rotation_degrees=5.0,
            translation_distance=0.004,
        ),
        use_goal_driven_actions=True,
        desired_object_distance=0.025,
        alpha=0.1,
        pc_alpha=0.5,
        max_pc_bias_steps=32,
        min_general_steps=8,
        min_heading_steps=12,
        max_consecutive_rejections=max_consecutive_rejections,
    )


def rejected_observations(action_type, action_signature, reason="unreachable"):
    return {
        AgentID("agent_id_0"): {
            SensorID("patch"): {
                "robodk_action_feedback": {
                    "status": "rejected",
                    "reason": reason,
                    "action_type": action_type,
                    "action_signature": action_signature,
                    "message": "test rejection",
                }
            }
        }
    }


def test_rejection_selects_a_different_action_and_leaves_goal_jump_mode():
    policy = make_policy()
    rng = np.random.RandomState(0)
    action_signature = "SetAgentPose(location=(0.1, 0.2, 0.3))"

    result = policy(
        SimpleNamespace(rng=rng),
        rejected_observations("SetAgentPose", action_signature),
        MotorSystemState(),
        percept=None,
        goal=None,
    )

    assert len(result.actions) == 1
    assert isinstance(result.actions[0], OrientHorizontal)
    assert repr(result.actions[0]) != action_signature
    assert policy._recovery_mode
    assert not policy._is_jumping


def test_repeated_rejections_use_distinct_actions_then_stop_episode():
    policy = make_policy(max_consecutive_rejections=3)
    rng = np.random.RandomState(0)
    rejected = "SetAgentPose(location=(0.1, 0.2, 0.3))"
    attempted_types = ["SetAgentPose"]
    attempted_signatures = [rejected]

    for _ in range(2):
        result = policy(
            SimpleNamespace(rng=rng),
            rejected_observations(attempted_types[-1], attempted_signatures[-1]),
            MotorSystemState(),
            percept=None,
            goal=None,
        )
        action = result.actions[0]
        assert repr(action) not in attempted_signatures
        attempted_types.append(type(action).__name__)
        attempted_signatures.append(repr(action))

    with pytest.raises(StopIteration):
        policy(
            SimpleNamespace(rng=rng),
            rejected_observations(attempted_types[-1], attempted_signatures[-1]),
            MotorSystemState(),
            percept=None,
            goal=None,
        )

    assert policy._consecutive_rejections == 3
