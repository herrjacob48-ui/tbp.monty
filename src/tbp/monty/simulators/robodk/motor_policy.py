# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import logging
from typing import Any, Protocol, cast

from numpy.random import RandomState

from tbp.monty.cmp import Goal, Message
from tbp.monty.context import RuntimeContext
from tbp.monty.experiment.motor_system import ExperimentMotorSystem
from tbp.monty.frameworks.actions.actions import (
    Action,
    MoveForward,
    OrientHorizontal,
    OrientVertical,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_policies import (
    MotorPolicyResult,
    SurfacePolicyCurvatureInformed,
    SurfacePolicyTelemetry,
)
from tbp.monty.frameworks.models.motor_system_state import MotorSystemState
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.memento import Memento

logger = logging.getLogger(__name__)


class RoboDKActionSampler(Protocol):
    def sample_move_forward(
        self, agent_id: AgentID, rng: RandomState
    ) -> MoveForward: ...

    def sample_orient_horizontal(
        self, agent_id: AgentID, rng: RandomState
    ) -> OrientHorizontal: ...

    def sample_orient_vertical(
        self, agent_id: AgentID, rng: RandomState
    ) -> OrientVertical: ...


class RoboDKSurfacePolicyCurvatureInformed(SurfacePolicyCurvatureInformed):
    """Surface policy that keeps proposing alternatives after RoboDK rejections."""

    def __init__(
        self,
        sensor_id: SensorID | str = "patch",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.use_goal_driven_actions: bool = bool(
            kwargs.get("use_goal_driven_actions", False)
        )
        self.sensor_id = SensorID(sensor_id)
        self._consecutive_rejections = 0
        self._rejected_action_signatures: set[str] = set()
        self._recovery_mode = False
        self._last_rejected_action_type = ""

    def reset(self, motor_system: ExperimentMotorSystem) -> None:
        self._consecutive_rejections = 0
        self._rejected_action_signatures.clear()
        self._recovery_mode = False
        self._last_rejected_action_type = ""
        super().reset(motor_system)

    def state_dict(self) -> Memento:
        return {
            **super().state_dict(),
            "consecutive_rejections": self._consecutive_rejections,
            "rejected_action_signatures": list(self._rejected_action_signatures),
            "recovery_mode": self._recovery_mode,
            "last_rejected_action_type": self._last_rejected_action_type,
        }

    def load_state_dict(self, memento: Memento) -> None:
        super().load_state_dict(memento)
        self._consecutive_rejections = memento.get("consecutive_rejections", 0)
        self._rejected_action_signatures = set(
            memento.get("rejected_action_signatures", [])
        )
        self._recovery_mode = memento.get("recovery_mode", False)
        self._last_rejected_action_type = memento.get("last_rejected_action_type", "")

    def __call__(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        state: MotorSystemState,
        percept: Message,
        goal: Goal | None,
    ) -> MotorPolicyResult:
        feedback = observations[self.agent_id][self.sensor_id].get(
            "robodk_action_feedback"
        )
        if feedback is not None and feedback.get("status") == "rejected":
            return self._recover_from_rejection(ctx, feedback)
        if feedback is not None and feedback.get("status") == "accepted":
            self._consecutive_rejections = 0
            self._rejected_action_signatures.clear()
            self._recovery_mode = False
            self._last_rejected_action_type = ""
        elif feedback is not None:
            raise ValueError(
                f"Unknown RoboDK action feedback status: {feedback.get('status')!r}"
            )

        if not self._recovery_mode:
            return super().__call__(ctx, observations, state, percept, goal)

        goal_driven_actions = self.use_goal_driven_actions
        self.use_goal_driven_actions = False
        try:
            result = super().__call__(ctx, observations, state, percept, None)
        finally:
            self.use_goal_driven_actions = goal_driven_actions

        repeated_actions = [
            action
            for action in result.actions
            if repr(action) in self._rejected_action_signatures
        ]
        if repeated_actions:
            repeated_signature = repr(repeated_actions[0])
            action = self._sample_untried_alternative(
                ctx,
                self._last_rejected_action_type,
                exclude_signature=repeated_signature,
            )
            if action is None:
                self._rejected_action_signatures.clear()
                action = self._sample_untried_alternative(
                    ctx,
                    self._last_rejected_action_type,
                    exclude_signature=repeated_signature,
                )
            if action is None:
                logger.warning(
                    "No new RoboDK recovery action is available; returning a "
                    "no-motion step and continuing recovery"
                )
                result.actions = []
                return result
            result.actions = [action]
            result.motor_only_step = action.name != OrientVertical.action_name()
            self.last_surface_policy_action = action
        return result

    def _recover_from_rejection(
        self,
        ctx: RuntimeContext,
        feedback: dict[str, str],
    ) -> MotorPolicyResult:
        action_signature = feedback.get("action_signature")
        action_type = feedback.get("action_type", "")
        if action_signature is None:
            raise ValueError("RoboDK rejection feedback is missing action_signature")

        self._consecutive_rejections += 1
        self._rejected_action_signatures.add(action_signature)
        self._last_rejected_action_type = action_type
        self._recovery_mode = True
        self._reset_jump_state()
        self._undo_action = None

        action = self._sample_untried_alternative(
            ctx, action_type, exclude_signature=action_signature
        )
        if action is None:
            self._rejected_action_signatures.clear()
            action = self._sample_untried_alternative(
                ctx, action_type, exclude_signature=action_signature
            )
        if action is None:
            logger.warning(
                "No new RoboDK recovery action is available; returning a "
                "no-motion step and continuing recovery"
            )
            self._telemetry = SurfacePolicyTelemetry()
            return MotorPolicyResult(actions=[], telemetry=self._telemetry)

        logger.warning(
            "RoboDK rejected %s (%s); trying a different surface action: %s",
            action_type,
            feedback.get("reason", "unknown"),
            action,
        )
        self.last_surface_policy_action = action
        self.attempting_to_find_object = False
        self._telemetry = SurfacePolicyTelemetry()
        return MotorPolicyResult(
            actions=[action],
            motor_only_step=action.name != OrientVertical.action_name(),
            telemetry=self._telemetry,
        )

    def _sample_untried_alternative(
        self,
        ctx: RuntimeContext,
        rejected_action_type: str,
        exclude_signature: str | None = None,
    ) -> Action | None:
        action_sampler = cast("RoboDKActionSampler", self.action_sampler)
        action_factories = {
            "MoveForward": lambda: action_sampler.sample_move_forward(
                self.agent_id, ctx.rng
            ),
            "OrientHorizontal": lambda: action_sampler.sample_orient_horizontal(
                self.agent_id, ctx.rng
            ),
            "OrientVertical": lambda: action_sampler.sample_orient_vertical(
                self.agent_id, ctx.rng
            ),
        }
        order = {
            "MoveForward": ("OrientHorizontal", "OrientVertical"),
            "OrientHorizontal": ("OrientVertical", "MoveForward"),
            "OrientVertical": ("OrientHorizontal", "MoveForward"),
        }.get(
            rejected_action_type,
            ("OrientHorizontal", "MoveForward", "OrientVertical"),
        )
        for action_type in order:
            action = action_factories[action_type]()
            if (
                repr(action) not in self._rejected_action_signatures
                and repr(action) != exclude_signature
            ):
                return action

        scale = 0.5 ** ((self._consecutive_rejections + 2) // 3)
        reverse = -1.0 if (self._consecutive_rejections // 6) % 2 else 1.0
        for action_type in order:
            action = self._scaled_action(
                action_factories[action_type](), scale, reverse
            )
            if (
                repr(action) not in self._rejected_action_signatures
                and repr(action) != exclude_signature
            ):
                logger.info(
                    "All standard RoboDK recovery actions have been rejected; "
                    "trying a scaled %s action (scale %.6g)",
                    action_type,
                    scale,
                )
                return action
        return None

    @staticmethod
    def _scaled_action(action: Action, scale: float, sign: float) -> Action:
        if isinstance(action, MoveForward):
            return MoveForward(action.agent_id, action.distance * scale * sign)
        if isinstance(action, OrientHorizontal):
            return OrientHorizontal(
                action.agent_id,
                action.rotation_degrees * scale * sign,
                action.left_distance * scale * sign,
                action.forward_distance * scale * sign,
            )
        if isinstance(action, OrientVertical):
            return OrientVertical(
                action.agent_id,
                action.rotation_degrees * scale * sign,
                action.down_distance * scale * sign,
                action.forward_distance * scale * sign,
            )
        raise TypeError(
            f"Cannot scale unsupported RoboDK recovery action {type(action).__name__}"
        )
