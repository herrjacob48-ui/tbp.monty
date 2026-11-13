# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import copy
import io
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import numpy.typing as npt
import quaternion as qt
from PIL import Image
from robodk.robolink import (
    COLLISION_OFF,
    COLLISION_ON,
    ITEM_TYPE_CAMERA,
    ITEM_TYPE_OBJECT,
    ITEM_TYPE_ROBOT,
    ITEM_TYPE_TARGET,
    ITEM_TYPE_TOOL,
    RUNMODE_RUN_ROBOT,
    RUNMODE_SIMULATE,
    Item,
    Robolink,
    TargetReachError,
)
from robodk.robomath import Mat, rotx, roty, transl
from scipy.spatial.transform import Rotation, Slerp

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
PRESCAN_AABB_SAFETY_MARGIN_MM = 25.0
PRESCAN_AABB_BELOW_BASE_MM = 300.0


class UnsafeMoveError(RuntimeError):
    """Raised when a requested robot motion fails the configured safety checks."""

    def __init__(self, message: str, failure_type: str = "safety"):
        super().__init__(message)
        self.failure_type = failure_type


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
        check_motion_safety: bool = True,
        table_reference_name: str = "Object Inspection",
        camera_dimensions_mm: tuple[float, float, float] = (100.0, 60.0, 50.0),
        table_clearance_mm: float = 10.0,
        max_step_mm: float = 50.0,
        prescan_enabled: bool = False,
        prescan_target_names: tuple[str, ...] = (
            "Monty Prescan View",
            "Monty Prescan View 2",
            "Monty Prescan View 3",
            "Monty Prescan View 4",
            "Monty Prescan View 5",
            "Monty Prescan View 6",
        ),
        prescan_center_xy_mm: tuple[float, float] | None = None,
        prescan_object_bottom_z_mm: float | None = None,
        prescan_max_object_dimensions_mm: tuple[float, float, float] = (
            400.0,
            400.0,
            400.0,
        ),
        prescan_center_tolerance_mm: float = 100.0,
        prescan_output_path: str | Path | None = None,
        depth_provider: DepthProvider | None = None,
        require_depth: bool = False,
        observation_output_dir: str | Path | None = None,
    ):
        self._validate_motion_safety_mode(use_real_robot, check_motion_safety)
        resolution = self._validate_resolution(resolution)
        camera_dimensions = np.asarray(camera_dimensions_mm, dtype=float)
        if (
            camera_dimensions.shape != (3,)
            or not np.isfinite(camera_dimensions).all()
            or np.any(camera_dimensions <= 0)
        ):
            raise ValueError(
                "camera_dimensions_mm must contain three positive finite "
                "dimensions in TCP-local X, Y, Z order"
            )
        if not np.isfinite(table_clearance_mm) or table_clearance_mm < 0:
            raise ValueError("table_clearance_mm must be finite and non-negative")
        if not np.isfinite(max_step_mm) or max_step_mm <= 0:
            raise ValueError("max_step_mm must be positive and finite")
        object_dimensions = self._validate_prescan_configuration(
            prescan_enabled,
            prescan_target_names,
            prescan_center_xy_mm,
            prescan_object_bottom_z_mm,
            prescan_max_object_dimensions_mm,
            prescan_center_tolerance_mm,
            use_real_robot,
            depth_provider,
            require_depth,
        )

        self.resolution = tuple(resolution)
        self.agent_id = AgentID(agent_id)
        self.sensor_id = SensorID(sensor_id)
        self.sensor_ids = [self.sensor_id]
        if view_finder_sensor_id is not None:
            view_finder_id = SensorID(view_finder_sensor_id)
            if view_finder_id == self.sensor_id:
                raise ValueError("patch and view-finder sensor IDs must be different")
            self.sensor_ids.append(view_finder_id)
        self.table_reference_name = table_reference_name
        self.camera_dimensions_mm = camera_dimensions
        self.table_clearance_mm = table_clearance_mm
        self.check_motion_safety = check_motion_safety
        self.max_step_mm = max_step_mm
        self.prescan_enabled = prescan_enabled
        self.prescan_target_names = tuple(prescan_target_names)
        self.prescan_center_xy_mm = (
            np.asarray(prescan_center_xy_mm, dtype=float)
            if prescan_center_xy_mm is not None
            else None
        )
        self.prescan_targets: list[Item] = []
        self.prescan_object_bottom_z_mm = prescan_object_bottom_z_mm
        self.prescan_max_object_dimensions_mm = object_dimensions
        self.prescan_center_tolerance_mm = prescan_center_tolerance_mm
        self.prescan_output_path = (
            Path(prescan_output_path) if prescan_output_path is not None else None
        )
        self.prescan_geometry: dict[str, object] | None = None
        self._prescanned_object_key: tuple[str, tuple[float, ...]] | None = None
        self._prescan_aabb_item: Item | None = None
        self._prescan_aabb_link_ids: tuple[int, ...] = ()
        self._prescan_aabb_tool: Item | None = None
        self.depth_provider = depth_provider
        self.require_depth = require_depth
        self.observation_output_dir = (
            Path(observation_output_dir) if observation_output_dir is not None else None
        )
        self._observation_count = 0
        self._last_action_feedback: dict[str, str] | None = None
        if self.observation_output_dir is not None:
            self.observation_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Saving RoboDK camera observations to %s",
                self.observation_output_dir,
            )
        self.use_real_robot = use_real_robot
        # Monty uses -Z as forward; this camera faces along the TCP's +Z axis.
        self._agent_frame_flip = rotx(np.pi)
        self.object_names = self._validate_object_names(object_names, prescan_enabled)
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
        if prescan_enabled:
            self.prescan_targets = self._find_prescan_targets(self.prescan_target_names)

        self._initialize_table_clearance(
            check_motion_safety,
            table_reference_name,
            camera_dimensions,
            table_clearance_mm,
            prescan_enabled,
        )

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
        self._camera_hfov_degrees: float | None = None
        if prescan_enabled:
            self._camera_hfov_degrees = self._camera_hfov_degrees_from_settings(
                self.camera.setParam("Settings")
            )

    @staticmethod
    def _validate_motion_safety_mode(
        use_real_robot: bool, check_motion_safety: bool
    ) -> None:
        if use_real_robot and not check_motion_safety:
            raise ValueError(
                "check_motion_safety cannot be disabled in physical-robot mode"
            )

    @staticmethod
    def _validate_resolution(resolution: tuple[int, int]) -> tuple[int, int]:
        if len(resolution) != 2 or any(size <= 0 for size in resolution):
            raise ValueError("resolution must be a positive (width, height) pair")
        return resolution

    @staticmethod
    def _validate_object_names(
        object_names: tuple[str, ...] | None, prescan_enabled: bool
    ) -> tuple[str, ...]:
        if object_names is not None and not object_names:
            raise ValueError(
                "object_names must list RoboDK station object names for training "
                "or evaluation"
            )
        names = tuple(object_names or ())
        if prescan_enabled and not names:
            raise ValueError(
                "prescan_enabled requires configured object_names so the adapter "
                "can identify the visible target"
            )
        return names

    @staticmethod
    def _validate_prescan_configuration(
        enabled: bool,
        target_names: tuple[str, ...],
        center_xy_mm: tuple[float, float] | None,
        object_bottom_z_mm: float | None,
        max_object_dimensions_mm: tuple[float, float, float],
        center_tolerance_mm: float,
        use_real_robot: bool,
        depth_provider: DepthProvider | None,
        require_depth: bool,
    ) -> npt.NDArray[np.float64]:
        object_dimensions = np.asarray(max_object_dimensions_mm, dtype=float)
        if (
            object_dimensions.shape != (3,)
            or not np.isfinite(object_dimensions).all()
            or np.any(object_dimensions <= 0)
        ):
            raise ValueError(
                "prescan_max_object_dimensions_mm must contain three positive "
                "finite dimensions in robot-base X, Y, Z order"
            )
        if enabled and (
            center_xy_mm is None
            or len(center_xy_mm) != 2
            or not np.isfinite(center_xy_mm).all()
        ):
            raise ValueError(
                "prescan_center_xy_mm must provide finite robot-base X and Y "
                "coordinates when prescan is enabled"
            )
        if enabled and (
            not target_names
            or any(not name.strip() for name in target_names)
            or len(set(target_names)) != len(target_names)
        ):
            raise ValueError(
                "prescan_target_names must contain unique, non-empty RoboDK target "
                "names when prescan is enabled"
            )
        if enabled and (
            object_bottom_z_mm is None or not np.isfinite(object_bottom_z_mm)
        ):
            raise ValueError(
                "prescan_object_bottom_z_mm must provide the object bottom "
                "height in robot-base coordinates when prescan is enabled"
            )
        if not np.isfinite(center_tolerance_mm) or center_tolerance_mm < 0:
            raise ValueError(
                "prescan_center_tolerance_mm must be finite and non-negative"
            )
        if enabled and use_real_robot:
            raise ValueError(
                "RoboDK prescan is simulation-only until physical "
                "collision checking and camera calibration are implemented"
            )
        if enabled and depth_provider is None and not require_depth:
            raise ValueError(
                "prescan_enabled requires require_depth=True or a depth_provider"
            )
        return object_dimensions

    def _initialize_table_clearance(
        self,
        check_motion_safety: bool,
        table_reference_name: str,
        camera_dimensions: npt.NDArray[np.float64],
        table_clearance_mm: float,
        prescan_enabled: bool,
    ) -> None:
        self.table_reference: Item | None = None
        self.table_height_mm: float | None = None
        if not check_motion_safety and not prescan_enabled:
            logger.warning(
                "RoboDK motion safety checks are DISABLED for simulation. "
                "No distance, tabletop-clearance, or IK checks will reject moves."
            )
            return

        self.table_reference = self.rdk.Item(table_reference_name)
        if not self.table_reference.Valid():
            raise LookupError(
                f"RoboDK table reference {table_reference_name!r} was not found"
            )
        self.table_height_mm = self._table_height_in_robot_base_mm(
            self.robot, self.table_reference
        )
        if check_motion_safety:
            logger.warning(
                "Using camera-only tabletop clearance: box %s mm, reference %r at "
                "Z=%.1f mm, clearance %.1f mm. Camera dimensions are provisional; "
                "robot arm and mount collisions are not checked.",
                tuple(float(size) for size in camera_dimensions),
                table_reference_name,
                self.table_height_mm,
                table_clearance_mm,
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

    def _find_prescan_targets(self, target_names: tuple[str, ...]) -> list[Item]:
        targets = []
        for target_name in target_names:
            target = self.rdk.Item(target_name, ITEM_TYPE_TARGET)
            if not target.Valid():
                raise LookupError(
                    f"RoboDK prescan target {target_name!r} was not found. "
                    "Create and orient every configured prescan target in the "
                    "station."
                )
            targets.append(target)
        return targets

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
    def _camera_hfov_degrees_from_settings(settings: str) -> float:
        for setting in settings.split():
            key, separator, value = setting.partition("=")
            if separator and key.upper() == "FOV":
                hfov_degrees = float(value)
                if np.isfinite(hfov_degrees) and 0 < hfov_degrees < 180:
                    return hfov_degrees
                break
        raise ValueError(
            "RoboDK camera settings must provide a horizontal FOV between 0 and "
            "180 degrees for prescan"
        )

    def _depth_to_robot_base_points(
        self, depth_m: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float64]:
        if self._camera_hfov_degrees is None:
            raise RuntimeError("RoboDK camera FOV is unavailable for prescan")
        height, width = depth_m.shape
        hfov = np.deg2rad(self._camera_hfov_degrees)
        vfov = 2.0 * np.arctan(np.tan(hfov / 2.0) * height / width)
        x_pixels = (np.arange(width) + 0.5) * (2.0 / width) - 1.0
        y_pixels = 1.0 - (np.arange(height) + 0.5) * (2.0 / height)
        x_grid, y_grid = np.meshgrid(x_pixels, y_pixels)
        depth_mm = depth_m.astype(np.float64) * 1000.0
        local_points = np.stack(
            (
                x_grid * depth_mm * np.tan(hfov / 2.0),
                y_grid * depth_mm * np.tan(vfov / 2.0),
                -depth_mm,
            ),
            axis=-1,
        ).reshape(-1, 3)

        tcp_pose = np.asarray(self.robot.Pose().rows, dtype=float)
        agent_rotation = (
            tcp_pose[:3, :3]
            @ np.asarray(self._agent_frame_flip.rows, dtype=float)[:3, :3]
        )
        points = local_points @ agent_rotation.T + tcp_pose[:3, 3]
        valid = np.isfinite(depth_mm.reshape(-1)) & (depth_mm.reshape(-1) > 0)
        return points[valid]

    def _visible_prescan_object_key(
        self,
    ) -> tuple[str, tuple[float, ...]] | None:
        visible_items = [
            (name, item) for name, item in self._object_items.items() if item.Visible()
        ]
        if not visible_items:
            return None
        if len(visible_items) != 1:
            raise RuntimeError(
                "RoboDK prescan requires exactly one visible target object"
            )
        name, item = visible_items[0]
        pose_rows = np.asarray(item.PoseAbs().rows, dtype=float)
        return name, tuple(float(value) for value in pose_rows.ravel())

    def _capture_prescan_views(self) -> list[npt.NDArray[np.float64]]:
        if not self.prescan_targets or any(
            not target.Valid() for target in self.prescan_targets
        ):
            raise RuntimeError(
                "One or more configured RoboDK prescan targets are unavailable: "
                f"{self.prescan_target_names!r}"
            )

        self._clear_prescan_collision_aabb()
        point_clouds = []
        for view_index, target in enumerate(self.prescan_targets, start=1):
            try:
                self.robot.MoveJ(target)
            except TargetReachError as error:
                raise RuntimeError(
                    f"RoboDK could not reach prescan target "
                    f"{self.prescan_target_names[view_index - 1]!r} "
                    f"({view_index}/{len(self.prescan_targets)})."
                ) from error
            observations = self.get_observation()
            depth = observations[self.agent_id][self.sensor_id].get("depth")
            if depth is None:
                raise RuntimeError("Prescan capture did not contain a depth image")
            point_clouds.append(
                self._depth_to_robot_base_points(np.asarray(depth, dtype=np.float32))
            )
        return point_clouds

    def _run_prescan(self, object_key: tuple[str, tuple[float, ...]]) -> None:
        if self.depth_provider is None and self._depth_camera is None:
            raise RuntimeError(
                "Prescan requires RoboDK simulated depth or a configured depth provider"
            )
        if self.prescan_center_xy_mm is None:
            raise RuntimeError("Prescan center coordinates are not configured")
        if self.prescan_object_bottom_z_mm is None:
            raise RuntimeError("Prescan object bottom height is not configured")

        dimensions = self.prescan_max_object_dimensions_mm
        center_xy = self.prescan_center_xy_mm
        point_clouds = self._capture_prescan_views()
        all_points = np.concatenate(point_clouds, axis=0)
        radial_limit_mm = (
            float(np.hypot(dimensions[0] / 2.0, dimensions[1] / 2.0))
            + self.prescan_center_tolerance_mm
        )
        relative_xy = all_points[:, :2] - center_xy
        point_radii = np.linalg.norm(relative_xy, axis=1)
        min_z_mm = self.prescan_object_bottom_z_mm + 1.0
        max_z_mm = (
            self.prescan_object_bottom_z_mm
            + dimensions[2]
            + self.prescan_center_tolerance_mm
        )
        in_object_roi = (
            (point_radii <= radial_limit_mm)
            & (all_points[:, 2] >= min_z_mm)
            & (all_points[:, 2] <= max_z_mm)
        )
        object_points = all_points[in_object_roi]
        if object_points.shape[0] < 10:
            raise RuntimeError(
                "Prescan collected fewer than 10 object-surface points. Check "
                "camera FOV, depth source, object center, and scan visibility."
            )

        padding_mm = 5.0
        observed_min = object_points.min(axis=0)
        observed_max = object_points.max(axis=0)
        estimate_min = observed_min - padding_mm
        estimate_max = observed_max + padding_mm
        estimate_min[2] = self.prescan_object_bottom_z_mm
        estimate_min[0] = max(estimate_min[0], center_xy[0] - dimensions[0] / 2.0)
        estimate_min[1] = max(estimate_min[1], center_xy[1] - dimensions[1] / 2.0)
        estimate_max[0] = min(estimate_max[0], center_xy[0] + dimensions[0] / 2.0)
        estimate_max[1] = min(estimate_max[1], center_xy[1] + dimensions[1] / 2.0)
        estimate_max[2] = min(
            estimate_max[2], self.prescan_object_bottom_z_mm + dimensions[2]
        )
        estimate_min -= PRESCAN_AABB_SAFETY_MARGIN_MM
        estimate_max += PRESCAN_AABB_SAFETY_MARGIN_MM
        estimate_min[2] = -PRESCAN_AABB_BELOW_BASE_MM - PRESCAN_AABB_SAFETY_MARGIN_MM
        estimated_dimensions = estimate_max - estimate_min
        if not np.isfinite(estimated_dimensions).all() or np.any(
            estimated_dimensions <= 0
        ):
            raise RuntimeError(
                "Prescan points do not produce positive AABB dimensions within "
                "the configured object region. Check the object center, dimensions, "
                "bottom height, and scan visibility."
            )
        point_filter_bounds = np.asarray(
            (
                (
                    center_xy[0] - dimensions[0] / 2.0,
                    center_xy[1] - dimensions[1] / 2.0,
                    self.prescan_object_bottom_z_mm,
                ),
                (
                    center_xy[0] + dimensions[0] / 2.0,
                    center_xy[1] + dimensions[1] / 2.0,
                    self.prescan_object_bottom_z_mm + dimensions[2],
                ),
            )
        )
        self.prescan_geometry = {
            "object_name": object_key[0],
            "point_count": int(object_points.shape[0]),
            "scan_view_count": len(point_clouds),
            "bounds_min_mm": estimate_min.tolist(),
            "bounds_max_mm": estimate_max.tolist(),
            "dimensions_mm": estimated_dimensions.tolist(),
            "aabb_volume_mm3": float(np.prod(estimated_dimensions)),
            "maximum_assumed_dimensions_mm": dimensions.tolist(),
            "point_filter_min_mm": point_filter_bounds[0].tolist(),
            "point_filter_max_mm": point_filter_bounds[1].tolist(),
            "geometry_is_complete": False,
            "note": (
                "The point-cloud AABB is an observation estimate from the "
                "configured RoboDK target viewpoints and "
                "may omit occluded surfaces. The configured maximum dimensions "
                "and center tolerance are used to filter observed points; the "
                "estimated AABB is extended down to robot-base Z=0 and used for "
                "collision checking only after the prescan completes. Robot "
                "placement comes from the predefined RoboDK station targets."
            ),
        }
        if self.prescan_enabled:
            self._set_prescan_collision_aabb(estimate_min, estimate_max)
        if self.prescan_output_path is not None:
            self.prescan_output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.prescan_output_path,
                points_mm=object_points,
                metadata_json=np.asarray(json.dumps(self.prescan_geometry)),
            )
        self._prescanned_object_key = object_key
        logger.info(
            "RoboDK prescan captured %d object points from %d configured "
            "viewpoints; observed AABB dimensions are %s mm.",
            object_points.shape[0],
            len(point_clouds),
            tuple(float(value) for value in estimated_dimensions),
        )

    @staticmethod
    def _joint_values(joints: Mat | list[float]) -> list[float]:
        if isinstance(joints, Mat):
            return [float(value) for value in joints.list()]
        return [float(value) for value in joints]

    @staticmethod
    def _aabb_box_mesh(
        bounds_min: npt.NDArray[np.float64],
        bounds_max: npt.NDArray[np.float64],
    ) -> Mat:
        half_extents = (bounds_max - bounds_min) / 2.0
        hx, hy, hz = half_extents
        corners = np.asarray(
            (
                (-hx, -hy, -hz),
                (hx, -hy, -hz),
                (hx, hy, -hz),
                (-hx, hy, -hz),
                (-hx, -hy, hz),
                (hx, -hy, hz),
                (hx, hy, hz),
                (-hx, hy, hz),
            )
        )
        triangle_indices = [
            0,
            2,
            1,
            0,
            3,
            2,
            4,
            5,
            6,
            4,
            6,
            7,
            0,
            1,
            5,
            0,
            5,
            4,
            1,
            2,
            6,
            1,
            6,
            5,
            3,
            7,
            6,
            3,
            6,
            2,
            0,
            4,
            7,
            0,
            7,
            3,
        ]
        triangles = corners[triangle_indices]
        return Mat(triangles.T.tolist())

    def _clear_prescan_collision_aabb(self) -> None:
        if self._prescan_aabb_item is None:
            return
        for link_id in self._prescan_aabb_link_ids:
            self.rdk.setCollisionActivePair(
                COLLISION_OFF, self.robot, self._prescan_aabb_item, link_id, 0
            )
        if self._prescan_aabb_tool is not None:
            self.rdk.setCollisionActivePair(
                COLLISION_OFF, self._prescan_aabb_tool, self._prescan_aabb_item
            )
        self._prescan_aabb_item.Delete()
        self._prescan_aabb_item = None
        self._prescan_aabb_link_ids = ()
        self._prescan_aabb_tool = None

    def _set_prescan_collision_aabb(
        self,
        bounds_min: npt.NDArray[np.float64],
        bounds_max: npt.NDArray[np.float64],
    ) -> None:
        bounds_min = np.asarray(bounds_min, dtype=float)
        bounds_max = np.asarray(bounds_max, dtype=float)
        if (
            bounds_min.shape != (3,)
            or bounds_max.shape != (3,)
            or not np.isfinite(bounds_min).all()
            or not np.isfinite(bounds_max).all()
            or np.any(bounds_max <= bounds_min)
        ):
            raise ValueError("Prescan collision AABB bounds must be finite and ordered")

        joint_count = len(self._joint_values(self.robot.Joints()))
        link_ids = tuple(range(joint_count + 1))
        tool = self.robot.getLink(ITEM_TYPE_TOOL)
        if not tool.Valid():
            raise RuntimeError(
                "Prescan AABB collision checking requires an active RoboDK tool "
                "item with configured collision geometry"
            )

        self._clear_prescan_collision_aabb()
        aabb_item = self.rdk.AddShape(self._aabb_box_mesh(bounds_min, bounds_max))
        if not aabb_item.Valid():
            raise RuntimeError("RoboDK could not create the prescan collision AABB")
        self._prescan_aabb_item = aabb_item
        center = (bounds_min + bounds_max) / 2.0
        aabb_item.setPoseAbs(self.robot.PoseAbs() * transl(*center))
        aabb_item.setColor([0.2, 0.6, 1.0, 0.0])

        self._prescan_aabb_link_ids = ()
        self._prescan_aabb_tool = None
        configured = False
        try:
            self.rdk.setCollisionActive(COLLISION_ON)
            for link_id in link_ids:
                if (
                    self.rdk.setCollisionActivePair(
                        COLLISION_ON, self.robot, aabb_item, link_id, 0
                    )
                    != 1
                ):
                    raise RuntimeError(
                        f"RoboDK could not enable collision checking for robot "
                        f"link {link_id} against the prescan AABB"
                    )
                self._prescan_aabb_link_ids += (link_id,)
            if self.rdk.setCollisionActivePair(COLLISION_ON, tool, aabb_item) != 1:
                raise RuntimeError(
                    "RoboDK could not enable collision checking for the active "
                    "robot tool against the prescan AABB"
                )
            self._prescan_aabb_tool = tool
            configured = True
        finally:
            if not configured:
                self._clear_prescan_collision_aabb()

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
        if width <= 0 or height <= 0 or depth_words.size != expected_words:
            raise ValueError(
                "RoboDK depth snapshot has unexpected dimensions or payload length: "
                f"{width}x{height}, {len(depth_bytes)} bytes"
            )

        depth_mm = np.flipud(depth_words[2:].reshape(height, width)).astype(np.float32)
        return depth_mm * (far_length_mm / (np.iinfo(np.uint32).max * 1000.0))

    @staticmethod
    def _pose_quaternion(pose_mat: npt.ArrayLike) -> qt.quaternion:
        xyzw = Rotation.from_matrix(np.asarray(pose_mat)[:3, :3]).as_quat()
        return qt.quaternion(xyzw[3], xyzw[0], xyzw[1], xyzw[2])

    def _state(self) -> ProprioceptiveState:
        pose_mat = np.asarray(self.robot.Pose().rows)
        position = tuple(pose_mat[:3, 3] / 1000.0)
        agent_pose = pose_mat.copy()
        agent_pose[:3, :3] = (
            pose_mat[:3, :3] @ np.asarray(self._agent_frame_flip.rows)[:3, :3]
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
        if self._last_action_feedback is not None:
            sensor_observation["robodk_action_feedback"] = (
                self._last_action_feedback.copy()
            )

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
            depth_m = self._decode_robodk_depth(depth_bytes, self._depth_far_length_mm)
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
                        sensor_id: copy.deepcopy(sensor_observation)
                        for sensor_id in self.sensor_ids
                    }
                )
            }
        )

    def _save_observation(
        self,
        rgba: npt.NDArray[np.uint8],
        depth: npt.NDArray[np.float32] | npt.NDArray[np.float64] | None,
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
        if not self.check_motion_safety and self._prescan_aabb_item is None:
            return True
        if self.check_motion_safety:
            if self.table_height_mm is None:
                raise RuntimeError(
                    "Table height is unavailable while safety checks are enabled"
                )
            if self._minimum_camera_z_along_path(self.robot.Pose(), target_pose) < (
                self.table_height_mm + self.table_clearance_mm
            ):
                return False
        return self._test_linear_path(target_pose) == 0

    def _test_linear_path(self, target_pose: Mat) -> int:
        start_joints = self._joint_values(self.robot.Joints())
        try:
            return self.robot.MoveL_Test(start_joints, target_pose)
        finally:
            self.robot.setJoints(start_joints)

    @staticmethod
    def _table_height_in_robot_base_mm(robot: Item, table_reference: Item) -> float:
        """Get the table reference origin's Z coordinate in the robot base frame.

        Returns:
            The reference origin's base-frame Z coordinate in millimeters.

        Raises:
            ValueError: If the resolved height is not finite.
        """
        table_pose_in_base = robot.PoseAbs().inv() * table_reference.PoseAbs()
        table_height_mm = float(table_pose_in_base.Pos()[2])
        if not np.isfinite(table_height_mm):
            raise ValueError("Table reference height must be finite")
        return table_height_mm

    def _camera_bottom_z_mm(self, pose: Mat) -> float:
        pose_matrix = np.asarray(pose.rows, dtype=float)
        vertical_half_extent = np.abs(pose_matrix[2, :3]) @ (
            self.camera_dimensions_mm / 2.0
        )
        return float(pose_matrix[2, 3] - vertical_half_extent)

    def _minimum_camera_z_along_path(self, start_pose: Mat, target_pose: Mat) -> float:
        """Conservatively estimate the camera box's lowest Z along a linear move.

        Returns:
            The estimated minimum camera-box Z in the robot base frame.
        """
        start = np.asarray(start_pose.rows, dtype=float)
        target = np.asarray(target_pose.rows, dtype=float)
        translation_delta = target[:3, 3] - start[:3, 3]
        rotations = Rotation.from_matrix(np.stack((start[:3, :3], target[:3, :3])))
        angular_distance = float((rotations[0].inv() * rotations[1]).magnitude())
        translation_distance = float(np.linalg.norm(translation_delta))
        sample_count = max(
            1,
            int(np.ceil(translation_distance / 5.0)),
            int(np.ceil(np.rad2deg(angular_distance) / 2.0)),
        )
        slerp = Slerp((0.0, 1.0), rotations)
        half_extents = self.camera_dimensions_mm / 2.0
        radius = float(np.linalg.norm(half_extents))
        sample_bottom_z = []
        for fraction in np.linspace(0.0, 1.0, sample_count + 1):
            rotation = slerp([fraction]).as_matrix()[0]
            position = start[:3, 3] + translation_delta * fraction
            vertical_half_extent = np.abs(rotation[2, :]) @ half_extents
            sample_bottom_z.append(position[2] - vertical_half_extent)

        # Bound possible between-sample height change from translation and rotation.
        interpolation_uncertainty = (
            abs(translation_delta[2]) + radius * angular_distance
        ) / (2.0 * sample_count)
        return float(min(sample_bottom_z) - interpolation_uncertainty)

    def _move_to_pose(self, target_pose: Mat) -> None:
        if self.check_motion_safety:
            current_position = np.asarray(self.robot.Pose().Pos(), dtype=float)
            target_position = np.asarray(target_pose.Pos(), dtype=float)
            displacement_mm = np.linalg.norm(target_position - current_position)
            if displacement_mm > self.max_step_mm + 1e-6:
                raise UnsafeMoveError(
                    f"Requested move is {displacement_mm:.1f} mm; "
                    f"maximum is {self.max_step_mm:.1f} mm",
                    failure_type="motion_limit",
                )
            min_camera_z_mm = self._minimum_camera_z_along_path(
                self.robot.Pose(), target_pose
            )
            if self.table_height_mm is None:
                raise RuntimeError(
                    "Table height is unavailable while safety checks enabled"
                )
            min_allowed_camera_z_mm = self.table_height_mm + self.table_clearance_mm
            if min_camera_z_mm < min_allowed_camera_z_mm:
                raise UnsafeMoveError(
                    f"Camera clearance check estimated a minimum Z of "
                    f"{min_camera_z_mm:.1f} mm along the move; it must remain at or "
                    f"above table Z {self.table_height_mm:.1f} mm plus "
                    f"{self.table_clearance_mm:.1f} mm clearance",
                    failure_type="table_clearance",
                )
        if self.check_motion_safety or self._prescan_aabb_item is not None:
            motion_test_result = self._test_linear_path(target_pose)
            if motion_test_result > 0:
                raise UnsafeMoveError(
                    f"RoboDK detected {motion_test_result} collision pair(s) along "
                    "the proposed path; choose a different action",
                    failure_type="collision",
                )
            if motion_test_result < 0:
                raise UnsafeMoveError(
                    "RoboDK rejected the proposed path because it is unreachable "
                    "or infeasible; choose a different action",
                    failure_type="path",
                )
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
        self._last_action_feedback = None
        last_action: Action | None = None
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
            try:
                action.act(self)
            except UnsafeMoveError as error:
                self._last_action_feedback = {
                    "status": "rejected",
                    "reason": "unsafe",
                    "failure_type": error.failure_type,
                    "action_type": type(action).__name__,
                    "action_signature": repr(action),
                    "message": str(error),
                }
                logger.warning(
                    "Rejected unsafe %s action; robot did not move: %s",
                    type(action).__name__,
                    error,
                )
                break
            except TargetReachError as error:
                self._last_action_feedback = {
                    "status": "rejected",
                    "reason": "unreachable",
                    "action_type": type(action).__name__,
                    "action_signature": repr(action),
                    "message": str(error),
                }
                logger.warning(
                    "RoboDK could not reach the target for %s; robot did not move: %s",
                    type(action).__name__,
                    error,
                )
                break
            last_action = action

        if self._last_action_feedback is None and last_action is not None:
            self._last_action_feedback = {
                "status": "accepted",
                "action_type": type(last_action).__name__,
                "action_signature": repr(last_action),
            }

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

    def actuate_set_sensor_rotation(self, action: SetSensorRotation) -> None:
        rotation = np.asarray(
            self._rotation_matrix(action.rotation_quat).rows, dtype=float
        )[:3, :3]
        if not np.allclose(rotation, np.eye(3), atol=1e-8):
            raise NotImplementedError(
                "A fixed, tool-mounted camera cannot independently execute "
                "SetSensorRotation; calibrate a gimbal or remove this action "
                "from the motor policy"
            )

    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        self._last_action_feedback = None
        if not self.prescan_enabled:
            self.robot.MoveJ(self.home_joints)
        else:
            object_key = self._visible_prescan_object_key()
            if object_key is None:
                if self._prescanned_object_key is None:
                    self.robot.MoveJ(self.home_joints)
            elif object_key != self._prescanned_object_key:
                self._run_prescan(object_key)
        return self.get_observation(), self._state()

    def close(self) -> None:
        self._clear_prescan_collision_aabb()
        for name, item in self._object_items.items():
            item.setVisible(self._initial_object_visibility[name])
        if self._depth_camera is not None:
            self.rdk.Cam2D_Close(self._depth_camera)
        self.rdk.Disconnect()
