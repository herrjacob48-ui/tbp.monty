# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import io

import numpy as np
import pytest
import quaternion as qt
from PIL import Image
from robodk.robolink import TargetReachError
from robodk.robomath import rotx, roty, transl

from tbp.monty.frameworks.actions.actions import (
    MoveForward,
    MoveTangentially,
    OrientHorizontal,
    OrientVertical,
    SetAgentPose,
    SetSensorRotation,
)
from tbp.monty.simulators.robodk.environment import (
    RoboDKFanucEnvironment,
    UnsafeMoveError,
)


class FakeRobot:
    def __init__(self):
        self.pose = transl(500, 0, 200)
        self.base_pose = transl(0, 0, 0)
        self.moved_pose = None
        self.move_test_result = None
        self.move_test_calls = 0
        self.move_error = None
        self.move_calls = 0

    def Pose(self):  # noqa: N802
        return self.pose

    def PoseAbs(self):  # noqa: N802
        return self.base_pose

    def Joints(self):  # noqa: N802
        return []

    def MoveL_Test(self, _joints, _pose):  # noqa: N802
        self.move_test_calls += 1
        if self.move_test_result is not None:
            return self.move_test_result
        return 0

    def MoveL(self, pose):  # noqa: N802
        self.move_calls += 1
        if self.move_error is not None:
            raise self.move_error
        self.pose = pose
        self.moved_pose = pose

    def MoveJ(self, _joints):  # noqa: N802
        return None


class FakeRoboDK:
    def __init__(self):
        image = Image.fromarray(np.full((12, 16, 3), 127, dtype=np.uint8))
        image_buffer = io.BytesIO()
        image.save(image_buffer, format="PNG")
        self.image_bytes = image_buffer.getvalue()
        self.depth_bytes = None

    def Cam2D_Snapshot(self, _path, _camera, params=""):  # noqa: N802
        if params == "Depth":
            return self.depth_bytes
        return self.image_bytes


class FakeTableReference:
    def __init__(self, height_mm=0.0):
        self.pose = transl(0.0, 0.0, height_mm)

    def PoseAbs(self):  # noqa: N802
        return self.pose


def make_environment(
    depth_provider=None,
    observation_output_dir=None,
    table_height_mm=0.0,
    check_motion_safety=True,
):
    env = RoboDKFanucEnvironment.__new__(RoboDKFanucEnvironment)
    env.resolution = (8, 6)
    env.agent_id = "agent_id_0"
    env.sensor_id = "patch"
    env.sensor_ids = ["patch", "view_finder"]
    env.camera_dimensions_mm = np.asarray((100.0, 60.0, 50.0))
    env.table_clearance_mm = 10.0
    env.table_height_mm = table_height_mm
    env.check_motion_safety = check_motion_safety
    env.max_step_mm = 50.0
    env.depth_provider = depth_provider
    env.require_depth = False
    env.observation_output_dir = observation_output_dir
    env._observation_count = 0
    env._last_action_feedback = None
    env._depth_far_length_mm = None
    env.use_real_robot = False
    env._agent_frame_flip = rotx(np.pi)
    env._depth_camera = None
    env.object_names = ()
    env._object_items = {}
    env._initial_object_visibility = {}
    env.rdk = FakeRoboDK()
    env.camera = object()
    env.robot = FakeRobot()
    env.table_reference = FakeTableReference(table_height_mm)
    env.home_joints = []
    return env


def test_reset_and_step_return_monty_observations_and_state():
    env = make_environment()

    observations, state = env.reset()
    observation = observations["agent_id_0"]["patch"]

    assert observation["rgba"].shape == (6, 8, 4)
    assert "depth" not in observation
    assert state["agent_id_0"].position == (0.5, 0.0, 0.2)
    np.testing.assert_allclose(
        qt.as_float_array(state["agent_id_0"].rotation),
        (0.0, 1.0, 0.0, 0.0),
        atol=1e-12,
    )
    assert observations["agent_id_0"]["view_finder"]["rgba"].shape == (6, 8, 4)
    assert "view_finder" in state["agent_id_0"].sensors

    observations, state = env.step([MoveForward(agent_id="agent_id_0", distance=0.01)])

    assert (
        observations["agent_id_0"]["patch"]["robodk_action_feedback"]["status"]
        == "accepted"
    )
    np.testing.assert_allclose(
        env.robot.moved_pose.Pos(), (500.0, 0.0, 210.0), atol=1e-12
    )
    assert observations["agent_id_0"]["patch"]["rgba"].shape == (6, 8, 4)
    np.testing.assert_allclose(
        state["agent_id_0"].position, (0.5, 0.0, 0.21), atol=1e-12
    )


def test_depth_provider_is_shape_checked_and_included_in_observation():
    env = make_environment(depth_provider=lambda _camera, shape: np.full(shape, 0.25))

    observations = env.get_observation()

    depth = observations["agent_id_0"]["patch"]["depth"]
    assert depth.shape == (6, 8)
    assert depth.dtype == np.float32
    assert np.all(depth == 0.25)


def test_observation_output_saves_rgb_and_depth(tmp_path):
    env = make_environment(
        depth_provider=lambda _camera, shape: np.full(shape, 0.25),
        observation_output_dir=tmp_path,
    )

    env.get_observation()

    rgb = np.asarray(Image.open(tmp_path / "frame_000000_rgb.png"))
    depth = np.load(tmp_path / "frame_000000_depth_m.npy", allow_pickle=False)
    assert rgb.shape == (6, 8, 4)
    assert depth.shape == (6, 8)
    assert depth.dtype == np.float32
    assert np.all(depth == 0.25)


def test_observation_output_requires_depth(tmp_path):
    env = make_environment(observation_output_dir=tmp_path)

    with pytest.raises(RuntimeError, match="requires RGB-aligned depth"):
        env.get_observation()


def test_invalid_depth_shape_raises_clear_error():
    env = make_environment(depth_provider=lambda _camera, _shape: np.zeros((2, 2)))

    with pytest.raises(ValueError, match="aligned to the RGB image"):
        env.get_observation()


def test_required_depth_fails_explicitly_when_not_configured():
    env = make_environment()
    env.require_depth = True

    with pytest.raises(RuntimeError, match="requires real, RGB-aligned depth"):
        env.get_observation()


def test_robodk_depth_buffer_is_scaled_to_meters_and_resized():
    env = make_environment()
    env._depth_camera = object()
    env._depth_far_length_mm = 1000.0
    values = np.full((2, 3), np.iinfo(np.uint32).max // 4, dtype=">u4")
    header = np.asarray([3, 2], dtype=">u4").tobytes()
    env.rdk.depth_bytes = header + values.tobytes()

    observations = env.get_observation()

    depth = observations["agent_id_0"]["patch"]["depth"]
    assert depth.shape == (6, 8)
    assert depth.dtype == np.float32
    assert np.all(depth == 0.25)


def test_robodk_depth_buffer_is_big_endian_and_vertically_flipped():
    max_depth = np.iinfo(np.uint32).max
    values = np.asarray(
        [[0, max_depth], [max_depth // 4, max_depth // 2]], dtype=">u4"
    )
    depth_bytes = (
        np.asarray([2, 2], dtype=">u4").tobytes() + values.tobytes()
    )

    depth = RoboDKFanucEnvironment._decode_robodk_depth(depth_bytes, 1000.0)

    np.testing.assert_allclose(
        depth,
        np.asarray([[0.25, 0.5], [0.0, 1.0]], dtype=np.float32),
    )


def test_camera_far_length_is_read_from_settings():
    assert RoboDKFanucEnvironment._camera_far_length_mm(
        "FOV=30.000 FOCAL_LENGTH=5.00 FAR_LENGTH=2005.00"
    ) == 2005.0


def test_unsafe_absolute_motion_is_logged_and_returns_current_state(caplog):
    env = make_environment()

    observations, state = env.step(
        [
            SetAgentPose(
                agent_id="agent_id_0",
                location=(0.6, 0.0, 0.2),
                rotation_quat=(1.0, 0.0, 0.0, 0.0),
            ),
            MoveForward(agent_id="agent_id_0", distance=0.01),
        ]
    )

    assert env.robot.moved_pose is None
    assert state["agent_id_0"].position == (0.5, 0.0, 0.2)
    assert observations["agent_id_0"]["patch"]["rgba"].shape == (6, 8, 4)
    assert "maximum is" in caplog.text
    assert "robot did not move" in caplog.text


def test_ik_rejected_motion_is_logged_and_returns_current_state(caplog):
    env = make_environment()
    env.robot.move_test_result = -1

    observations, state = env.step(
        [MoveForward(agent_id="agent_id_0", distance=0.01)]
    )

    assert env.robot.moved_pose is None
    assert state["agent_id_0"].position == (0.5, 0.0, 0.2)
    assert observations["agent_id_0"]["patch"]["rgba"].shape == (6, 8, 4)
    assert "RoboDK rejected the motion safety check" in caplog.text
    assert "robot did not move" in caplog.text
    feedback = observations["agent_id_0"]["patch"]["robodk_action_feedback"]
    assert feedback["status"] == "rejected"
    assert feedback["reason"] == "unsafe"


def test_unreachable_target_is_logged_and_does_not_abort_step(caplog):
    env = make_environment(check_motion_safety=False)
    env.robot.move_error = TargetReachError("Target cannot be reached")

    observations, state = env.step(
        [
            MoveForward(agent_id="agent_id_0", distance=0.01),
            MoveForward(agent_id="agent_id_0", distance=0.01),
        ]
    )

    assert env.robot.moved_pose is None
    assert env.robot.move_calls == 1
    assert state["agent_id_0"].position == (0.5, 0.0, 0.2)
    assert observations["agent_id_0"]["patch"]["rgba"].shape == (6, 8, 4)
    assert "RoboDK could not reach the target" in caplog.text
    assert "Target cannot be reached" in caplog.text
    feedback = observations["agent_id_0"]["patch"]["robodk_action_feedback"]
    assert feedback["status"] == "rejected"
    assert feedback["reason"] == "unreachable"


def test_unexpected_move_error_still_propagates():
    env = make_environment(check_motion_safety=False)
    env.robot.move_error = RuntimeError("unexpected failure")

    with pytest.raises(RuntimeError, match="unexpected failure"):
        env.step([MoveForward(agent_id="agent_id_0", distance=0.01)])


def test_direct_unsafe_move_raises_detailed_error():
    env = make_environment()

    with pytest.raises(UnsafeMoveError, match="maximum is"):
        env._move_to_pose(transl(600, 0, 200))

    assert env.robot.moved_pose is None


def test_disabling_motion_safety_bypasses_distance_table_and_ik_checks():
    env = make_environment(table_height_mm=100.0, check_motion_safety=False)
    env.robot.move_test_result = -1

    env._move_to_pose(transl(600.0, 0.0, 50.0))

    assert tuple(env.robot.moved_pose.Pos()) == (600.0, 0.0, 50.0)
    assert env.robot.move_test_calls == 0


def test_physical_robot_mode_cannot_disable_motion_safety():
    with pytest.raises(
        ValueError, match="cannot be disabled in physical-robot mode"
    ):
        RoboDKFanucEnvironment(use_real_robot=True, check_motion_safety=False)


def test_camera_box_allows_tcp_below_legacy_minimum_when_clear_of_table():
    env = make_environment()
    env.robot.pose = transl(500, 0, 120)

    env._move_to_pose(transl(500, 0, 80))

    assert tuple(env.robot.moved_pose.Pos()) == (500.0, 0.0, 80.0)


def test_table_height_is_converted_from_station_to_robot_base_frame():
    robot = FakeRobot()
    robot.base_pose = transl(100.0, 20.0, 30.0) * roty(np.pi / 2)
    table_reference = FakeTableReference()
    table_reference.pose = robot.base_pose * transl(25.0, 0.0, 80.0)

    table_height_mm = RoboDKFanucEnvironment._table_height_in_robot_base_mm(
        robot, table_reference
    )

    assert table_height_mm == pytest.approx(80.0)


def test_camera_box_must_clear_table_by_configured_margin():
    env = make_environment(table_height_mm=100.0)
    env.max_step_mm = 100.0

    with pytest.raises(
        UnsafeMoveError, match=r"table Z 100\.0 mm plus 10\.0 mm"
    ):
        env._move_to_pose(transl(500, 0, 130))

    assert env.robot.moved_pose is None


def test_camera_rotation_is_checked_along_the_move_not_only_at_endpoints():
    env = make_environment()
    env.robot.pose = transl(500, 0, 63)
    target_pose = transl(500, 0, 63) * roty(np.pi / 2)

    assert env._camera_bottom_z_mm(env.robot.pose) >= 10.0
    assert env._camera_bottom_z_mm(target_pose) >= 10.0
    with pytest.raises(UnsafeMoveError, match="minimum Z"):
        env._move_to_pose(target_pose)

    assert env.robot.moved_pose is None


def test_motion_limit_allows_floating_point_roundoff_at_boundary():
    env = make_environment()

    env._move_to_pose(transl(550.0000000000001, 0.0, 200.0))

    assert tuple(env.robot.moved_pose.Pos()) == (550.0000000000001, 0.0, 200.0)


@pytest.mark.parametrize(
    "action",
    [
        MoveTangentially(
            agent_id="agent_id_0", distance=0.005, direction=(1.0, 1.0, 0.0)
        ),
        OrientHorizontal(
            agent_id="agent_id_0",
            rotation_degrees=5.0,
            left_distance=0.005,
            forward_distance=0.005,
        ),
        OrientVertical(
            agent_id="agent_id_0",
            rotation_degrees=5.0,
            down_distance=0.005,
            forward_distance=0.005,
        ),
    ],
)
def test_combined_relative_action_translation_is_capped(action):
    env = make_environment()
    env.max_step_mm = 5.0
    start_position = np.asarray(env.robot.pose.Pos())

    env.step([action])

    displacement = np.linalg.norm(
        np.asarray(env.robot.moved_pose.Pos()) - start_position
    )
    np.testing.assert_allclose(displacement, 5.0, atol=1e-12)


def test_add_object_selects_registered_station_item():
    class FakeItem:
        def __init__(self):
            self.visible = False

        def setVisible(self, visible):  # noqa: N802
            self.visible = visible

        def Visible(self):  # noqa: N802
            return self.visible

    env = make_environment()
    first = FakeItem()
    second = FakeItem()
    env.object_names = ("cup", "bowl")
    env._object_items = {"cup": first, "bowl": second}
    env._initial_object_visibility = {"cup": False, "bowl": False}

    info = env.add_object("bowl", semantic_id=4)

    assert info.object_id == 2
    assert info.semantic_id == 4
    assert first.visible is False
    assert second.visible is True


def test_unknown_station_item_is_reported():
    env = make_environment()

    with pytest.raises(LookupError, match="not in configured RoboDK object_names"):
        env.add_object("unknown")


def test_set_agent_pose_accepts_quaternion_object():
    env = make_environment()
    env.step(
        [
            SetAgentPose(
                agent_id="agent_id_0",
                location=(0.5, 0.0, 0.2),
                rotation_quat=qt.one,
            )
        ]
    )

    assert tuple(env.robot.moved_pose.Pos()) == (500.0, 0.0, 200.0)


def test_fixed_camera_accepts_identity_sensor_rotation_as_noop():
    env = make_environment()

    observations, _state = env.step(
        [
            SetSensorRotation(
                agent_id="agent_id_0",
                rotation_quat=(1.0, 0.0, 0.0, 0.0),
            )
        ]
    )

    assert env.robot.moved_pose is None
    assert (
        observations["agent_id_0"]["patch"]["robodk_action_feedback"]["status"]
        == "accepted"
    )


def test_fixed_camera_rejects_nonidentity_sensor_rotation():
    env = make_environment()

    with pytest.raises(NotImplementedError, match="tool-mounted camera"):
        env.step(
            [
                SetSensorRotation(
                    agent_id="agent_id_0",
                    rotation_quat=(0.0, 0.0, 0.0, 1.0),
                )
            ]
        )
