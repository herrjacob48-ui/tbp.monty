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
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import numpy.typing as npt
import quaternion as qt
from PIL import Image
from robodk.robolink import (
    ITEM_TYPE_CAMERA,
    ITEM_TYPE_OBJECT,
    ITEM_TYPE_ROBOT,
    RUNMODE_RUN_ROBOT,
    RUNMODE_SIMULATE,
    Item,
    Robolink,
)
from robodk.robomath import Mat, rotx, roty, transl
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.actions.actions import (
    Action,
    MoveForward,
    MoveTangentially,
    OrientHorizontal,
    OrientVertical,
    SetAgentPose,
    SetSensorRotation,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environments.environment import (
    ObjectID,
    ObjectInfo,
    SemanticID,
)
from tbp.monty.frameworks.models.abstract_monty_classes import (
    AgentObservations,
    Observations,
    SensorObservation,
)
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    ProprioceptiveState,
    SensorState,
)
from tbp.monty.frameworks.sensors import SensorID

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

DepthProvider = Callable[[object, tuple], npt.ArrayLike]


class UnsafeMoveError(RuntimeError):
    """Raised when a requested robot motion fails the configured safety checks."""


class RoboDKFanucEnvironment:
    """Monty environment adapter for a robot and a RoboDK camera.

    The camera is assumed co-located with the active TCP. The reported agent frame is
    rotated so Monty's -Z forward axis aligns with the camera's TCP +Z viewing axis.
    RoboDK simulated depth is scaled using the camera's FAR_LENGTH setting.
    Physical-robot mode requires a separate aligned RGB-D depth provider.
    """

    def __init__(
        self,
        robot_name: str = "Fanuc CRX-10iA",
        camera_name: str = "Camera 1",
        resolution: tuple[int, int] = (64, 64),
        agent_id: str = "agent_id_0",
        sensor_id: str = "patch",
        view_finder_sensor_id: str | None = "view_finder",
        object_names: tuple[str, ...] | None = None,
        use_real_robot: bool = False,
        min_safe_z_mm: float = 100.0,
        max_step_mm: float = 50.0,
        depth_provider: DepthProvider | None = None,
        require_depth: bool = False,
        observation_output_dir: str | Path | None = None,
    ):
        if len(resolution) != 2 or any(size <= 0 for size in resolution):
            raise ValueError("resolution must be a positive (width, height) pair")
        if min_safe_z_mm < 0 or max_step_mm <= 0:
            raise ValueError("safety limits must be positive")

        self.resolution = tuple(resolution)
        self.agent_id = AgentID(agent_id)
        self.sensor_id = SensorID(sensor_id)
        self.sensor_ids = [self.sensor_id]
        if view_finder_sensor_id is not None:
            view_finder_id = SensorID(view_finder_sensor_id)
            if view_finder_id == self.sensor_id:
                raise ValueError("patch and view-finder sensor IDs must be different")
            self.sensor_ids.append(view_finder_id)
        self.min_safe_z_mm = min_safe_z_mm
        self.max_step_mm = max_step_mm
        self.depth_provider = depth_provider
        self.require_depth = require_depth
        self.observation_output_dir = (
            Path(observation_output_dir) if observation_output_dir is not None else None
        )
        self._observation_count = 0
        if self.observation_output_dir is not None:
            self.observation_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Saving RoboDK camera observations to %s",
                self.observation_output_dir,
            )
        self.use_real_robot = use_real_robot
        # Monty uses -Z as forward; this camera faces along the TCP's +Z axis.
        self._agent_frame_flip = rotx(np.pi)
        self.object_names = tuple(object_names or ())
        if object_names is not None and not object_names:
            raise ValueError(
                "object_names must list RoboDK station object names for training "
                "or evaluation"
            )
        if require_depth and depth_provider is None and use_real_robot:
            raise ValueError(
                "RoboDK simulated depth is unavailable in physical-robot mode; "
                "provide an RGB-D depth_provider"
            )

        self.rdk = self._connect_robodk()
        self.rdk.setRunMode(RUNMODE_RUN_ROBOT if use_real_robot else RUNMODE_SIMULATE)

        self.robot = self.rdk.Item(robot_name, ITEM_TYPE_ROBOT)
        if not self.robot.Valid():
            raise LookupError(f"RoboDK robot {robot_name!r} was not found")

        self.camera = self.rdk.Item(camera_name, ITEM_TYPE_CAMERA)
        if not self.camera.Valid():
            self.camera = self.rdk.Item("Camera", ITEM_TYPE_CAMERA)
        if not self.camera.Valid():
            raise LookupError(f"RoboDK camera {camera_name!r} was not found")

        self.robot.setSpeedJoints(15 if use_real_robot else 45)
        self.robot.setSpeed(30 if use_real_robot else 100)
        self.home_joints = self.robot.Joints()

        self._object_items: dict[str, Item] = {}
        self._initial_object_visibility: dict[str, bool] = {}
        for object_name in self.object_names:
            item = self.rdk.Item(object_name, ITEM_TYPE_OBJECT)
            if not item.Valid():
                raise LookupError(
                    f"RoboDK station object {object_name!r} was not found"
                )
            if object_name in self._object_items:
                raise ValueError(f"Duplicate RoboDK object name: {object_name!r}")
            self._object_items[object_name] = item
            self._initial_object_visibility[object_name] = item.Visible()

        self._depth_camera: Item | None = None
        self._depth_far_length_mm: float | None = None
        if require_depth and depth_provider is None:
            width, height = self.resolution
            self._depth_camera = self.rdk.Cam2D_Add(
                camera_item=self.camera,
                cam_params=f"SIZE={width}x{height} MINIMIZED NO_TASKBAR",
            )
            self._depth_far_length_mm = self._camera_far_length_mm(
                self.camera.setParam("Settings")
            )

    def _connect_robodk(self) -> Robolink:
        candidates = ["127.0.0.1"]
        try:
            with Path("/etc/resolv.conf").open() as resolv_conf:
                resolver_lines = resolv_conf.readlines()
            for line in resolver_lines:
                fields = line.split()
                if fields[:1] == ["nameserver"] and not fields[1].startswith("127."):
                    if fields[1] not in candidates:
                        candidates.insert(0, fields[1])
                    break
        except OSError as error:
            logger.debug("Could not read the configured DNS server: %s", error)

        for ip in candidates:
            rdk = Robolink(robodk_ip=ip, port=20500)
            if rdk.Connect():
                return rdk

        raise ConnectionError(
            f"Could not connect to RoboDK on port 20500 (tried {candidates})"
        )

    @staticmethod
    def _camera_far_length_mm(settings: str) -> float:
        for setting in settings.split():
            key, separator, value = setting.partition("=")
            if separator and key.upper() == "FAR_LENGTH":
                far_length_mm = float(value)
                if np.isfinite(far_length_mm) and far_length_mm > 0:
                    return far_length_mm
                break
        raise ValueError(
            "RoboDK camera settings must provide a positive FAR_LENGTH in millimeters"
        )

    @staticmethod
    def _decode_robodk_depth(
        depth_bytes: bytes, far_length_mm: float
    ) -> npt.NDArray[np.float32]:
        if not np.isfinite(far_length_mm) or far_length_mm <= 0:
            raise ValueError("far_length_mm must be positive and finite")
        if not isinstance(depth_bytes, bytes) or len(depth_bytes) < 8:
            raise ValueError("RoboDK returned an invalid depth snapshot")

        depth_words = np.frombuffer(depth_bytes, dtype=">u4")
        width, height = (int(value) for value in depth_words[:2])
        expected_words = 2 + width * height
        if (
            width <= 0
            or height <= 0
            or depth_words.size != expected_words
        ):
            raise ValueError(
                "RoboDK depth snapshot has unexpected dimensions or payload length: "
                f"{width}x{height}, {len(depth_bytes)} bytes"
            )

        depth_mm = np.flipud(depth_words[2:].reshape(height, width)).astype(
            np.float32
        )
        return depth_mm * (
            far_length_mm / (np.iinfo(np.uint32).max * 1000.0)
        )

    @staticmethod
    def _pose_quaternion(pose_mat: npt.ArrayLike) -> qt.quaternion:
        xyzw = Rotation.from_matrix(np.asarray(pose_mat)[:3, :3]).as_quat()
        return qt.quaternion(xyzw[3], xyzw[0], xyzw[1], xyzw[2])

    def _state(self) -> ProprioceptiveState:
        pose_mat = np.asarray(self.robot.Pose().rows)
        position = tuple(pose_mat[:3, 3] / 1000.0)
        agent_pose = pose_mat.copy()
        agent_pose[:3, :3] = (
            pose_mat[:3, :3]
            @ np.asarray(self._agent_frame_flip.rows)[:3, :3]
        )
        rotation = self._pose_quaternion(agent_pose)
        return ProprioceptiveState(
            {
                self.agent_id: AgentState(
                    position=position,
                    rotation=rotation,
                    sensors={
                        self.sensor_id: SensorState(
                            position=(0.0, 0.0, 0.0),
                            rotation=qt.quaternion(1.0, 0.0, 0.0, 0.0),
                        ),
                        **{
                            sensor_id: SensorState(
                                position=(0.0, 0.0, 0.0),
                                rotation=qt.quaternion(1.0, 0.0, 0.0, 0.0),
                            )
                            for sensor_id in self.sensor_ids
                            if sensor_id != self.sensor_id
                        },
                    },
                )
            }
        )

    def get_observation(self) -> Observations:
        if self.require_depth and self.depth_provider is None:
            if self._depth_camera is None:
                raise RuntimeError(
                    "This Monty experiment requires real, RGB-aligned depth in "
                    "meters. Configure a depth_provider or RoboDK depth camera."
                )
            image_bytes = self.rdk.Cam2D_Snapshot("", self._depth_camera, "Color")
        else:
            image_bytes = self.rdk.Cam2D_Snapshot("", self.camera)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        rgba = np.asarray(
            image.resize(self.resolution, Image.Resampling.BILINEAR), dtype=np.uint8
        )
        sensor_observation: SensorObservation = {"rgba": rgba}

        if self.depth_provider is not None:
            image_shape = (rgba.shape[0], rgba.shape[1])
            depth = np.asarray(self.depth_provider(self.camera, image_shape))
            if depth.shape != rgba.shape[:2]:
                raise ValueError(
                    "depth provider must return an image aligned to the RGB image "
                    f"with shape {rgba.shape[:2]}, got {depth.shape}"
                )
            if not np.issubdtype(depth.dtype, np.number):
                raise TypeError("depth provider must return a numeric array in meters")
            if not np.isfinite(depth).all() or np.any(depth < 0):
                raise ValueError("depth values must be finite and non-negative")
            sensor_observation["depth"] = depth.astype(np.float32, copy=False)
        elif self._depth_camera is not None:
            if self._depth_far_length_mm is None:
                raise RuntimeError("RoboDK camera FAR_LENGTH is unavailable")
            depth_bytes = self.rdk.Cam2D_Snapshot("", self._depth_camera, "Depth")
            depth_m = self._decode_robodk_depth(
                depth_bytes, self._depth_far_length_mm
            )
            depth_image = Image.fromarray(depth_m.astype(np.float32), mode="F")
            depth_resized = depth_image.resize(
                self.resolution, Image.Resampling.NEAREST
            )
            sensor_observation["depth"] = np.asarray(depth_resized, dtype=np.float32)

        if self.observation_output_dir is not None:
            self._save_observation(rgba, sensor_observation.get("depth"))

        return Observations(
            {
                self.agent_id: AgentObservations(
                    {
                        sensor_id: {
                            key: (
                                value.copy() if isinstance(value, np.ndarray) else value
                            )
                            for key, value in sensor_observation.items()
                        }
                        for sensor_id in self.sensor_ids
                    }
                )
            }
        )

    def _save_observation(
        self,
        rgba: npt.NDArray[np.uint8],
        depth: npt.NDArray[np.float32] | None,
    ) -> None:
        output_dir = self.observation_output_dir
        if output_dir is None:
            raise RuntimeError("RoboDK observation output directory is not configured")
        if depth is None:
            raise RuntimeError(
                "RoboDK observation output requires RGB-aligned depth data"
            )

        frame_name = f"frame_{self._observation_count:06d}"
        Image.fromarray(rgba).save(output_dir / f"{frame_name}_rgb.png")
        np.save(
            output_dir / f"{frame_name}_depth_m.npy",
            depth,
            allow_pickle=False,
        )
        self._observation_count += 1

    def remove_all_objects(self) -> None:
        for item in self._object_items.values():
            item.setVisible(False)

    def add_object(
        self,
        name: str,
        position=(0.0, 0.0, 0.0),
        rotation=(1.0, 0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        semantic_id: SemanticID | None = None,
        primary_target_object: ObjectID | None = None,
    ) -> ObjectInfo:
        del position, rotation, scale, primary_target_object
        try:
            item_index = self.object_names.index(name)
            item = self._object_items[name]
        except (ValueError, KeyError) as error:
            raise LookupError(
                f"Object {name!r} is not in configured RoboDK object_names "
                f"{self.object_names!r}"
            ) from error

        self.remove_all_objects()
        item.setVisible(True)
        return ObjectInfo(
            object_id=ObjectID(item_index + 1),
            semantic_id=semantic_id,
        )

    def is_move_safe(self, target_pose: Mat) -> bool:
        _, _, z = target_pose.Pos()
        if z < self.min_safe_z_mm:
            return False
        return self.robot.MoveL_Test(self.robot.Joints(), target_pose) == 0

    def _move_to_pose(self, target_pose: Mat) -> None:
        current_position = np.asarray(self.robot.Pose().Pos(), dtype=float)
        target_position = np.asarray(target_pose.Pos(), dtype=float)
        displacement_mm = np.linalg.norm(target_position - current_position)
        if displacement_mm > self.max_step_mm + 1e-6:
            raise UnsafeMoveError(
                f"Requested move is {displacement_mm:.1f} mm; "
                f"maximum is {self.max_step_mm:.1f} mm"
            )
        _, _, target_z_mm = target_pose.Pos()
        if target_z_mm < self.min_safe_z_mm:
            raise UnsafeMoveError(
                f"Target Z is {target_z_mm:.1f} mm; "
                f"minimum safe Z is {self.min_safe_z_mm:.1f} mm"
            )
        if self.robot.MoveL_Test(self.robot.Joints(), target_pose) != 0:
            raise UnsafeMoveError("RoboDK rejected the motion safety check")
        self.robot.MoveL(target_pose)

    @staticmethod
    def _rotation_matrix(rotation_quat_wxyz: Sequence[float]) -> Mat:
        if isinstance(rotation_quat_wxyz, qt.quaternion):
            quaternion_wxyz = qt.as_float_array(rotation_quat_wxyz)
        else:
            quaternion_wxyz = np.asarray(rotation_quat_wxyz, dtype=float)
        if (
            quaternion_wxyz.shape != (4,)
            or not np.isfinite(quaternion_wxyz).all()
            or np.linalg.norm(quaternion_wxyz) == 0
        ):
            raise ValueError(
                "Monty rotations must be finite, non-zero WXYZ quaternions"
            )
        w, x, y, z = quaternion_wxyz
        rotation = Rotation.from_quat((x, y, z, w)).as_matrix()
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        return Mat(matrix.tolist())

    def _relative_move(self, delta: Mat) -> None:
        delta_matrix = np.asarray(delta.rows, dtype=float)
        translation = delta_matrix[:3, 3]
        displacement_mm = np.linalg.norm(translation)
        if not np.isfinite(delta_matrix).all():
            raise ValueError("Relative motion pose must contain finite values")
        if displacement_mm > self.max_step_mm:
            delta_matrix[:3, 3] = translation * (self.max_step_mm / displacement_mm)
            delta = Mat(delta_matrix.tolist())

        frame_flip = self._agent_frame_flip
        target_pose = self.robot.Pose() * frame_flip * delta * frame_flip
        self._move_to_pose(target_pose)

    def step(
        self, actions: Sequence[Action]
    ) -> tuple[Observations, ProprioceptiveState]:
        """Apply actions, treating safety-rejected moves as no-ops.

        A rejected action is logged and ends the current action sequence. The
        environment then returns a fresh observation and the actual robot state so
        Monty can choose what to do next.

        Returns:
            The next sensor observations and measured robot state.

        Raises:
            ValueError: If an action targets a different agent.
            NotImplementedError: If the action type is unsupported by this adapter.
        """
        supported_actions = (
            MoveForward,
            MoveTangentially,
            OrientHorizontal,
            OrientVertical,
            SetAgentPose,
            SetSensorRotation,
        )
        for action in actions:
            if action.agent_id != self.agent_id:
                raise ValueError(
                    f"Action agent {action.agent_id!r} does not match "
                    f"environment agent {self.agent_id!r}"
                )
            if not isinstance(action, supported_actions):
                raise NotImplementedError(
                    f"RoboDK adapter does not support {type(action).__name__}"
                )
            if isinstance(action, SetSensorRotation):
                raise NotImplementedError(
                    "A fixed, tool-mounted camera cannot independently execute "
                    "SetSensorRotation; calibrate a gimbal or remove this action "
                    "from the motor policy"
                )
            try:
                action.act(self)
            except UnsafeMoveError as error:
                logger.warning(
                    "Rejected unsafe %s action; robot did not move: %s",
                    type(action).__name__,
                    error,
                )
                break

        return self.get_observation(), self._state()

    def actuate_move_forward(self, action: MoveForward) -> None:
        distance_mm = float(
            np.clip(action.distance * 1000.0, -self.max_step_mm, self.max_step_mm)
        )
        self._relative_move(transl(0.0, 0.0, -distance_mm))

    def actuate_move_tangentially(self, action: MoveTangentially) -> None:
        direction = np.asarray(action.direction, dtype=float)
        if direction.shape != (3,) or not np.isfinite(direction).all():
            raise ValueError("MoveTangentially direction must be a finite XYZ vector")
        distance_mm = float(
            np.clip(action.distance * 1000.0, -self.max_step_mm, self.max_step_mm)
        )
        self._relative_move(transl(*(direction * distance_mm)))

    def actuate_orient_horizontal(self, action: OrientHorizontal) -> None:
        left_mm = float(
            np.clip(action.left_distance * 1000.0, -self.max_step_mm, self.max_step_mm)
        )
        forward_mm = float(
            np.clip(
                action.forward_distance * 1000.0,
                -self.max_step_mm,
                self.max_step_mm,
            )
        )
        rotation_degrees = float(np.clip(action.rotation_degrees, -15.0, 15.0))
        self._relative_move(
            transl(-left_mm, 0.0, 0.0)
            * roty(np.deg2rad(-rotation_degrees))
            * transl(0.0, 0.0, -forward_mm)
        )

    def actuate_orient_vertical(self, action: OrientVertical) -> None:
        down_mm = float(
            np.clip(action.down_distance * 1000.0, -self.max_step_mm, self.max_step_mm)
        )
        forward_mm = float(
            np.clip(
                action.forward_distance * 1000.0,
                -self.max_step_mm,
                self.max_step_mm,
            )
        )
        rotation_degrees = float(np.clip(action.rotation_degrees, -15.0, 15.0))
        self._relative_move(
            transl(0.0, -down_mm, 0.0)
            * rotx(np.deg2rad(rotation_degrees))
            * transl(0.0, 0.0, -forward_mm)
        )

    def actuate_set_agent_pose(self, action: SetAgentPose) -> None:
        position = np.asarray(action.location, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("SetAgentPose location must be a finite XYZ vector")
        target_pose = (
            transl(*(position * 1000.0))
            * self._rotation_matrix(action.rotation_quat)
            * self._agent_frame_flip
        )
        self._move_to_pose(target_pose)

    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        self.robot.MoveJ(self.home_joints)
        return self.get_observation(), self._state()

    def close(self) -> None:
        for name, item in self._object_items.items():
            item.setVisible(self._initial_object_visibility[name])
        if self._depth_camera is not None:
            self.rdk.Cam2D_Close(self._depth_camera)
        self.rdk.Disconnect()
