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

from tbp.monty.frameworks.actions.action_samplers import ConstantSampler
from tbp.monty.frameworks.actions.actions import (
    MoveForward,
    OrientHorizontal,
    OrientVertical,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.motor_policies import (
    MotorPolicyResult,
    SurfacePolicyCurvatureInformed,
)
from tbp.monty.frameworks.models.motor_system_state import MotorSystemState
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.simulators.robodk.motor_policy import (
    RoboDKSurfacePolicyCurvatureInformed,
)


def make_policy():
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
    )


def rejected_observations(
    action_type,
    action_signature,
    reason="unreachable",
    failure_type="path",
):
    return {
        AgentID("agent_id_0"): {
            SensorID("patch"): {
                "robodk_action_feedback": {
                    "status": "rejected",
                    "reason": reason,
                    "failure_type": failure_type,
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


def test_collision_feedback_requests_a_different_action():
    policy = make_policy()
    rng = np.random.RandomState(0)
    action_signature = "MoveForward(agent_id='agent_id_0', distance=0.004)"

    result = policy(
        SimpleNamespace(rng=rng),
        rejected_observations(
            "MoveForward",
            action_signature,
            reason="unsafe",
            failure_type="collision",
        ),
        MotorSystemState(),
        percept=None,
        goal=None,
    )

    assert len(result.actions) == 1
    assert not isinstance(result.actions[0], MoveForward)
    assert repr(result.actions[0]) != action_signature
    assert policy._recovery_mode


def test_repeated_rejections_keep_requesting_distinct_alternatives():
    policy = make_policy()
    rng = np.random.RandomState(0)
    rejected = "SetAgentPose(location=(0.1, 0.2, 0.3))"
    attempted_types = ["SetAgentPose"]
    attempted_signatures = [rejected]

    for _ in range(12):
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
    assert policy._consecutive_rejections == 12


def test_accepted_action_clears_collision_recovery_state(monkeypatch):
    policy = make_policy()
    rng = np.random.RandomState(0)
    policy(
        SimpleNamespace(rng=rng),
        rejected_observations(
            "MoveForward",
            "MoveForward(agent_id='agent_id_0', distance=0.004)",
            reason="unsafe",
            failure_type="collision",
        ),
        MotorSystemState(),
        percept=None,
        goal=None,
    )

    observations = {
        AgentID("agent_id_0"): {
            SensorID("patch"): {"robodk_action_feedback": {"status": "accepted"}}
        }
    }
    monkeypatch.setattr(
        SurfacePolicyCurvatureInformed,
        "__call__",
        lambda *_args, **_kwargs: MotorPolicyResult(),
    )
    policy(
        SimpleNamespace(rng=rng),
        observations,
        MotorSystemState(),
        percept=None,
        goal=None,
    )

    assert policy._consecutive_rejections == 0
    assert not policy._rejected_action_signatures
    assert not policy._recovery_mode
