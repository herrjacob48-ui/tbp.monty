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
    PRESCAN_AABB_BELOW_BASE_MM,
    PRESCAN_AABB_SAFETY_MARGIN_MM,
    RoboDKFanucEnvironment,
    UnsafeMoveError,
)


class FakeIKSolutions:
    def __init__(self, rows):
        self.rows = rows


class FakeCollisionItem:
    def __init__(self):
        self.valid = True
        self.pose = None
        self.mesh = None
        self.color = None

    def Valid(self):  # noqa: N802
        return self.valid

    def setPoseAbs(self, pose):  # noqa: N802
        self.pose = pose

    def setColor(self, color):  # noqa: N802
        self.color = color

    def Delete(self):  # noqa: N802
        self.valid = False


class FakeRobot:
    def __init__(self):
        self.pose = transl(500, 0, 200)
        self.base_pose = transl(0, 0, 0)
        self.moved_pose = None
        self.move_test_result = None
        self.move_test_calls = 0
        self.move_test_joint_starts = []
        self.move_j_test_result = 0
        self.move_j_test_calls = 0
        self.move_j_test_joint_starts = []
        self.set_joints_calls = []
        self.ik_poses = {}
        self.move_error = None
        self.move_calls = 0
        self.move_j_calls = 0
        self.move_j_targets = []
        self.tool_item = FakeCollisionItem()

    def Pose(self):  # noqa: N802
        return self.pose

    def PoseAbs(self):  # noqa: N802
        return self.base_pose

    def Joints(self):  # noqa: N802
        return list(self.pose.Pos())

    def setJoints(self, joints):  # noqa: N802
        self.set_joints_calls.append(list(joints))
        if hasattr(self, "_pose_before_move_test") and joints == list(
            self._pose_before_move_test.Pos()
        ):
            self.pose = self._pose_before_move_test
            del self._pose_before_move_test
        elif len(joints) == 3:
            self.pose = transl(*joints)

    def getLink(self, _item_type):  # noqa: N802
        return self.tool_item

    def PoseTool(self):  # noqa: N802
        return transl(0.0, 0.0, 0.0)

    def PoseFrame(self):  # noqa: N802
        return transl(0.0, 0.0, 0.0)

    def MoveL_Test(self, joints, pose):  # noqa: N802
        self.move_test_calls += 1
        self.move_test_joint_starts.append(list(joints))
        self._pose_before_move_test = self.pose
        self.pose = pose
        if self.move_test_result is not None:
            return self.move_test_result
        return 0

    def SolveIK_All(self, pose, tool=None, reference=None):  # noqa: N802
        del tool, reference
        joints = [float(value) for value in pose.Pos()]
        self.ik_poses[tuple(joints)] = pose
        return FakeIKSolutions(np.asarray(joints, dtype=float).reshape(3, 1))

    def MoveJ_Test(self, start_joints, target_joints):  # noqa: N802
        self.move_j_test_calls += 1
        self.move_j_test_joint_starts.append((list(start_joints), list(target_joints)))
        return self.move_j_test_result

    def MoveL(self, pose):  # noqa: N802
        self.move_calls += 1
        if self.move_error is not None:
            raise self.move_error
        self.pose = pose
        self.moved_pose = pose

    def MoveJ(self, joints):  # noqa: N802
        self.move_j_calls += 1
        self.move_j_targets.append(joints)
        if hasattr(joints, "pose"):
            self.pose = joints.pose
            return
        target_pose = self.ik_poses.get(tuple(joints))
        if target_pose is not None:
            self.pose = target_pose
        elif len(joints) == 3:
            self.pose = transl(*joints)


class FakeTarget:
    def __init__(self, pose):
        self.pose = pose
        self.valid = True

    def Valid(self):  # noqa: N802
        return self.valid

    def PoseAbs(self):  # noqa: N802
        return self.pose

    def Joints(self):  # noqa: N802
        return list(self.pose.Pos())

    def Parent(self):  # noqa: N802
        return object()

    def setPoseAbs(self, pose):  # noqa: N802
        self.pose = pose

    def Delete(self):  # noqa: N802
        self.valid = False


class FakeRoboDK:
    def __init__(self):
        image = Image.fromarray(np.full((12, 16, 3), 127, dtype=np.uint8))
        image_buffer = io.BytesIO()
        image.save(image_buffer, format="PNG")
        self.image_bytes = image_buffer.getvalue()
        self.depth_bytes = None
        self.collision_enabled = False
        self.collision_pairs = []
        self.collision_shapes = []

    def AddShape(self, mesh):  # noqa: N802
        item = FakeCollisionItem()
        item.mesh = mesh
        self.collision_shapes.append(item)
        return item

    def setCollisionActive(self, state):  # noqa: N802
        self.collision_enabled = bool(state)
        return 0

    def setCollisionActivePair(self, state, item1, item2, id1=0, id2=0):  # noqa: N802
        self.collision_pairs.append((state, item1, item2, id1, id2))
        return 1

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
    env.prescan_enabled = False
    env.prescan_target_names = tuple(
        f"Monty Prescan View {index}" if index > 1 else "Monty Prescan View"
        for index in range(1, 7)
    )
    env.prescan_targets = []
    env.prescan_center_xy_mm = None
    env.prescan_object_bottom_z_mm = 0.0
    env.prescan_max_object_dimensions_mm = np.asarray((400.0, 400.0, 400.0))
    env.prescan_center_tolerance_mm = 100.0
    env.prescan_output_path = None
    env.prescan_geometry = None
    env._prescan_aabb_item = None
    env._prescan_aabb_link_ids = ()
    env._prescan_aabb_tool = None
    env._prescanned_object_key = None
    env._camera_hfov_degrees = None
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
    values = np.asarray([[0, max_depth], [max_depth // 4, max_depth // 2]], dtype=">u4")
    depth_bytes = np.asarray([2, 2], dtype=">u4").tobytes() + values.tobytes()

    depth = RoboDKFanucEnvironment._decode_robodk_depth(depth_bytes, 1000.0)

    np.testing.assert_allclose(
        depth,
        np.asarray([[0.25, 0.5], [0.0, 1.0]], dtype=np.float32),
    )


def test_camera_far_length_is_read_from_settings():
    assert (
        RoboDKFanucEnvironment._camera_far_length_mm(
            "FOV=30.000 FOCAL_LENGTH=5.00 FAR_LENGTH=2005.00"
        )
        == 2005.0
    )


def test_camera_horizontal_fov_is_read_from_settings():
    assert (
        RoboDKFanucEnvironment._camera_hfov_degrees_from_settings(
            "FOV=30.000 FOCAL_LENGTH=5.00 FAR_LENGTH=2005.00"
        )
        == 30.0
    )


def test_prescan_captures_depth_estimate_and_saves_metadata(tmp_path):
    env = make_environment(
        depth_provider=lambda _camera, shape: np.full(shape, 0.01, dtype=np.float32),
        check_motion_safety=False,
        table_height_mm=-100.0,
    )
    targets = [FakeTarget(transl(100.0, 10.0 * index, 100.0)) for index in range(6)]
    env.prescan_targets = targets
    env.prescan_enabled = True
    env.robot.pose = targets[0].pose
    env.camera_dimensions_mm = np.asarray((10.0, 10.0, 10.0))
    env.prescan_center_xy_mm = np.asarray((100.0, 25.0))
    env.prescan_max_object_dimensions_mm = np.asarray((20.0, 20.0, 20.0))
    env.prescan_center_tolerance_mm = 100.0
    env._camera_hfov_degrees = 90.0
    env._depth_far_length_mm = 2000.0
    env.prescan_output_path = tmp_path / "prescan.npz"
    env._run_prescan(("cube", ()))

    assert env.robot.move_j_calls == 6
    assert env.robot.move_j_test_calls == 0
    assert env.robot.move_j_targets == targets
    assert env.prescan_geometry is not None
    assert env.prescan_geometry["point_count"] > 10
    assert env.prescan_geometry["scan_view_count"] == 6
    assert env.prescan_geometry["geometry_is_complete"] is False
    assert env.prescan_geometry["bounds_min_mm"][2] == (
        -PRESCAN_AABB_BELOW_BASE_MM - PRESCAN_AABB_SAFETY_MARGIN_MM
    )
    assert env.prescan_geometry["dimensions_mm"][2] == pytest.approx(
        env.prescan_geometry["bounds_max_mm"][2]
        + PRESCAN_AABB_BELOW_BASE_MM
        + PRESCAN_AABB_SAFETY_MARGIN_MM
    )
    np.testing.assert_allclose(
        np.asarray(env.prescan_geometry["bounds_min_mm"][:2]),
        np.asarray(env.prescan_geometry["point_filter_min_mm"][:2])
        - PRESCAN_AABB_SAFETY_MARGIN_MM,
    )
    np.testing.assert_allclose(
        np.asarray(env.prescan_geometry["bounds_max_mm"][:2]),
        np.asarray(env.prescan_geometry["point_filter_max_mm"][:2])
        + PRESCAN_AABB_SAFETY_MARGIN_MM,
    )
    assert env._prescan_aabb_item is not None
    with np.load(env.prescan_output_path, allow_pickle=False) as output:
        assert output["points_mm"].shape[1] == 3
        assert output["points_mm"].shape[0] > 10
        assert "point_filter_min_mm" in str(output["metadata_json"])


def test_prescan_uses_predefined_station_targets_in_order():
    env = make_environment()
    targets = [FakeTarget(transl(500.0, 50.0 * index, 400.0)) for index in range(6)]
    env.prescan_targets = targets
    env.prescan_enabled = True
    env.prescan_target_names = tuple(f"view-{index}" for index in range(6))
    env.prescan_center_xy_mm = np.asarray((650.0, -30.0))
    captures = []

    def capture(_depth):
        position = tuple(env.robot.Pose().Pos())
        captures.append(position)
        return np.asarray([position])

    env.get_observation = lambda: {
        "agent_id_0": {"patch": {"depth": np.ones((1, 1), dtype=np.float32)}}
    }
    env._depth_to_robot_base_points = capture

    clouds = env._capture_prescan_views()

    assert env.robot.move_j_targets == targets
    assert len(captures) == 6
    assert captures == [tuple(target.pose.Pos()) for target in targets]
    assert len(clouds) == 6


def test_prescan_moves_to_targets_without_a_temporary_collision_aabb():
    env = make_environment()
    targets = [FakeTarget(transl(500.0, 50.0 * index, 400.0)) for index in range(6)]
    env.prescan_enabled = True
    env.prescan_targets = targets
    env.prescan_center_xy_mm = np.asarray((650.0, -30.0))
    env.robot.move_j_test_result = 1
    captures = []

    def capture(_depth):
        captures.append(tuple(env.robot.Pose().Pos()))
        return np.asarray([env.robot.Pose().Pos()])

    env.get_observation = lambda: {
        "agent_id_0": {"patch": {"depth": np.ones((1, 1), dtype=np.float32)}}
    }
    env._depth_to_robot_base_points = capture

    clouds = env._capture_prescan_views()

    assert env.robot.move_j_calls == 6
    assert env.robot.move_j_test_calls == 0
    assert env._prescan_aabb_item is None
    assert captures == [tuple(target.pose.Pos()) for target in targets]
    assert len(clouds) == 6


def test_prescan_requires_a_tool_collision_model():
    env = make_environment()
    env.robot.tool_item.valid = False

    with pytest.raises(RuntimeError, match="active RoboDK tool item"):
        env._set_prescan_collision_aabb(
            np.asarray((600.0, -50.0, 100.0)),
            np.asarray((700.0, 50.0, 300.0)),
        )

    assert env.rdk.collision_shapes == []


def test_prescan_aabb_checks_tool_and_robot_links():
    env = make_environment()
    env.robot.tool_item.valid = True
    bounds_min = np.asarray((10.0, 20.0, 30.0))
    bounds_max = np.asarray((40.0, 60.0, 90.0))

    env._set_prescan_collision_aabb(bounds_min, bounds_max)

    assert env._prescan_aabb_item is not None
    assert env._prescan_aabb_item.color[3] == 0.0
    mesh_points = np.asarray(env._prescan_aabb_item.mesh.rows).T
    np.testing.assert_allclose(mesh_points.min(axis=0), (-15.0, -20.0, -30.0))
    np.testing.assert_allclose(mesh_points.max(axis=0), (15.0, 20.0, 30.0))
    np.testing.assert_allclose(env._prescan_aabb_item.pose.Pos(), (25.0, 40.0, 60.0))
    enabled_link_ids = [
        pair[3]
        for pair in env.rdk.collision_pairs
        if pair[0] == 1 and pair[1] is env.robot
    ]
    assert enabled_link_ids == [0, 1, 2, 3]
    assert any(
        pair[0] == 1 and pair[1] is env.robot.tool_item
        for pair in env.rdk.collision_pairs
    )


def test_prescan_aabb_rejects_regular_moves_even_without_other_motion_checks():
    env = make_environment(check_motion_safety=False)
    env._set_prescan_collision_aabb(
        np.asarray((600.0, -50.0, 100.0)),
        np.asarray((700.0, 50.0, 300.0)),
    )
    env.robot.move_test_result = 1

    with pytest.raises(UnsafeMoveError, match="collision pair"):
        env._move_to_pose(transl(510.0, 0.0, 200.0))

    assert env.robot.move_test_calls == 1
    assert env.robot.move_calls == 0
    assert env.robot.set_joints_calls == [[500.0, 0.0, 200.0]]


def test_prescan_depth_points_are_projected_to_robot_base_coordinates():
    env = make_environment()
    env._camera_hfov_degrees = 60.0
    depth = np.ones((2, 2), dtype=np.float32)

    points = env._depth_to_robot_base_points(depth)

    assert points.shape == (4, 3)
    np.testing.assert_allclose(points[:, 2], 1200.0, atol=1e-8)
    np.testing.assert_allclose(
        np.sort(points[:, 0]),
        (
            500.0 - 500 / np.sqrt(3),
            500.0 - 500 / np.sqrt(3),
            500.0 + 500 / np.sqrt(3),
            500.0 + 500 / np.sqrt(3),
        ),
    )


def test_reset_runs_prescan_once_for_each_visible_object_pose():
    class FakeItem:
        def Visible(self):  # noqa: N802
            return True

        def PoseAbs(self):  # noqa: N802
            return transl(650.0, -30.0, 0.0)

    env = make_environment()
    env.prescan_enabled = True
    env.object_names = ("cube",)
    env._object_items = {"cube": FakeItem()}
    run_keys = []

    def run_prescan(object_key):
        run_keys.append(object_key)
        env._prescanned_object_key = object_key

    env._run_prescan = run_prescan

    env.reset()
    env.reset()

    assert len(run_keys) == 1
    assert env.robot.move_j_calls == 0


def test_prescan_requires_a_depth_source():
    with pytest.raises(ValueError, match="requires require_depth=True"):
        RoboDKFanucEnvironment(
            prescan_enabled=True,
            prescan_center_xy_mm=(650.0, -30.0),
            prescan_object_bottom_z_mm=0.0,
            object_names=("cube",),
        )


def test_prescan_is_rejected_in_physical_robot_mode():
    with pytest.raises(ValueError, match="simulation-only"):
        RoboDKFanucEnvironment(
            use_real_robot=True,
            prescan_enabled=True,
            prescan_center_xy_mm=(650.0, -30.0),
            prescan_object_bottom_z_mm=0.0,
            object_names=("cube",),
            require_depth=True,
        )


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

    observations, state = env.step([MoveForward(agent_id="agent_id_0", distance=0.01)])

    assert env.robot.moved_pose is None
    assert state["agent_id_0"].position == (0.5, 0.0, 0.2)
    assert observations["agent_id_0"]["patch"]["rgba"].shape == (6, 8, 4)
    assert "RoboDK rejected the proposed path" in caplog.text
    assert "robot did not move" in caplog.text
    feedback = observations["agent_id_0"]["patch"]["robodk_action_feedback"]
    assert feedback["status"] == "rejected"
    assert feedback["reason"] == "unsafe"
    assert feedback["failure_type"] == "path"


def test_collision_rejection_is_reported_as_unsafe_with_retry_instruction(caplog):
    env = make_environment()
    env.robot.move_test_result = 2

    observations, state = env.step([MoveForward(agent_id="agent_id_0", distance=0.01)])

    feedback = observations["agent_id_0"]["patch"]["robodk_action_feedback"]
    assert feedback["status"] == "rejected"
    assert feedback["reason"] == "unsafe"
    assert feedback["failure_type"] == "collision"
    assert "choose a different action" in feedback["message"]
    assert "2 collision pair(s)" in caplog.text
    assert env.robot.moved_pose is None
    assert state["agent_id_0"].position == (0.5, 0.0, 0.2)


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
    with pytest.raises(ValueError, match="cannot be disabled in physical-robot mode"):
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

    with pytest.raises(UnsafeMoveError, match=r"table Z 100\.0 mm plus 10\.0 mm"):
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
