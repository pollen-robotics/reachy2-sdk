"""Reachy MobileBase module.

This package provides remote access (via socket) to the mobile base of a Reachy robot.
You can have access to basic information from the mobile base such as the battery voltage
or the odometry. You can also easily make the mobile base move by setting a goal position
in cartesian coordinates (x, y, theta) or directly send velocities (x_vel, y_vel, theta_vel).
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Dict, Optional

import grpc
from google.protobuf.empty_pb2 import Empty
from google.protobuf.wrappers_pb2 import FloatValue
from numpy import deg2rad, rad2deg, round
from reachy2_sdk_api.mobile_base_mobility_pb2 import (
    DirectionVector,
    GoToVector,
    TargetDirectionCommand,
)
from reachy2_sdk_api.mobile_base_mobility_pb2_grpc import MobileBaseMobilityServiceStub
from reachy2_sdk_api.mobile_base_utility_pb2 import (
    ControlModeCommand,
    ControlModePossiblities,
)
from reachy2_sdk_api.mobile_base_utility_pb2 import MobileBase as MobileBase_proto
from reachy2_sdk_api.mobile_base_utility_pb2 import (
    MobileBaseState,
    MobileBaseStatus,
    ZuuuModeCommand,
    ZuuuModePossiblities,
)
from reachy2_sdk_api.mobile_base_utility_pb2_grpc import MobileBaseUtilityServiceStub

from ..sensors.lidar import Lidar
from .part import Part


class MobileBase(Part):
    """The MobileBase class handles Reachy's mobile base.

    It holds:

    - the odometry of the base (you can also easily reset it),
    - the battery voltage to monitor the battery usage,
    - the control and drive mode of the base,
    - two methods to send target positions or target velocities.

    If you encounter a problem when using the base, you have access to an emergency shutdown method.
    """

    def __init__(
        self,
        mb_msg: MobileBase_proto,
        initial_state: MobileBaseState,
        grpc_channel: grpc.Channel,
    ) -> None:
        """Set up the connection with the mobile base."""
        self._logger = logging.getLogger(__name__)
        super().__init__(mb_msg, grpc_channel, MobileBaseUtilityServiceStub(grpc_channel))

        self._mobility_stub = MobileBaseMobilityServiceStub(grpc_channel)

        self._drive_mode: str = ZuuuModePossiblities.keys()[initial_state.zuuu_mode.mode].lower()
        self._control_mode: str = ControlModePossiblities.keys()[initial_state.control_mode.mode].lower()

        self._max_xy_vel = 1.0
        self._max_rot_vel = 180.0
        self._max_xy_goto = 1.0

        self.lidar = Lidar(initial_state.lidar_safety, grpc_channel)

        self._update_with(initial_state)

    def __repr__(self) -> str:
        """Clean representation of a mobile base."""
        repr_template = (
            "<MobileBase on={on} \n" " lidar_safety_enabled={lidar_safety_enabled} \n" " battery_voltage={battery_voltage}>"
        )
        return repr_template.format(
            on=self.is_on(),
            lidar_safety_enabled=self.lidar.safety_enabled,
            battery_voltage=self.battery_voltage,
        )

    @property
    def battery_voltage(self) -> float:
        """Return the battery voltage. Battery should be recharged if it reaches 24.5V or below."""
        battery_level = float(round(self._battery_level, 1))
        if battery_level < 24.5:
            self._logger.warning(f"Low battery level: {battery_level}V. Consider recharging.")
        return float(round(self._battery_level, 1))

    @property
    def odometry(self) -> Dict[str, float]:
        """Return the odometry of the base. x, y are in meters and theta in degree."""
        response = self._stub.GetOdometry(Empty())
        odom = {
            "x": round(response.x.value, 3),
            "y": round(response.y.value, 3),
            "theta": round(rad2deg(response.theta.value), 3),
        }
        return odom

    def _set_drive_mode(self, mode: str) -> None:
        """Set the base's drive mode."""
        all_drive_modes = [mode.lower() for mode in ZuuuModePossiblities.keys()][1:]
        possible_drive_modes = [mode for mode in all_drive_modes if mode not in ("speed", "goto")]
        if mode in possible_drive_modes:
            req = ZuuuModeCommand(mode=getattr(ZuuuModePossiblities, mode.upper()))
            self._stub.SetZuuuMode(req)
            self._drive_mode = mode
        else:
            raise ValueError(f"Drive mode requested should be in {possible_drive_modes}!")

    def _set_control_mode(self, mode: str) -> None:
        """Set the base's control mode."""
        possible_control_modes = [mode.lower() for mode in ControlModePossiblities.keys()][1:]
        if mode in possible_control_modes:
            req = ControlModeCommand(mode=getattr(ControlModePossiblities, mode.upper()))
            self._stub.SetControlMode(req)
            self._control_mode = mode
        else:
            raise ValueError(f"Control mode requested should be in {possible_control_modes}!")

    def reset_odometry(self) -> None:
        """Reset the odometry."""
        self._stub.ResetOdometry(Empty())
        time.sleep(0.03)

    def set_speed(self, x_vel: float, y_vel: float, rot_vel: float) -> None:
        """Send target speed. x_vel, y_vel are in m/s and rot_vel in deg/s for 200ms.

        The 200ms duration is predifined at the ROS level of the mobile base's code.
        This mode is prefered if the user wants to send speed instructions frequently.
        """
        for vel, value in {"x_vel": x_vel, "y_vel": y_vel}.items():
            if abs(value) > self._max_xy_vel:
                raise ValueError(f"The asbolute value of {vel} should not be more than {self._max_xy_vel}!")

        if abs(rot_vel) > self._max_rot_vel:
            raise ValueError(f"The asbolute value of rot_vel should not be more than {self._max_rot_vel}!")

        if self._drive_mode != "cmd_vel":
            self._set_drive_mode("cmd_vel")

        req = TargetDirectionCommand(
            direction=DirectionVector(
                x=FloatValue(value=x_vel),
                y=FloatValue(value=y_vel),
                theta=FloatValue(value=deg2rad(rot_vel)),
            )
        )
        self._mobility_stub.SendDirection(req)

    def goto(
        self,
        x: float,
        y: float,
        theta: float,
        timeout: Optional[float] = None,
        tolerance: Dict[str, float] = {"delta_x": 0.1, "delta_y": 0.1, "delta_theta": 15, "distance": 0.1},
    ) -> None:
        """Send target position. x, y are in meters and theta is in degree.

        (x, y) will define the position of the mobile base in cartesian space
        and theta its orientation. The zero position is set when the mobile base is
        started or if the  reset_odometry method is called.
        A timeout in seconds is defined so that the mobile base does get stuck in a go
        to call.
        The tolerance represents the margin along x, y and theta for which we consider
        that the mobile base has arrived its goal.
        """
        if self.is_off():
            self._logger.warning("Mobile base is off. Goto not sent.")

        exc_queue: Queue[Exception] = Queue()

        if not timeout:
            # We consider that the max velocity for the mobile base is 0.5 m/s
            # timeout is 2*_max_xy_goto / max velocity
            timeout = 2 * self._max_xy_goto / 0.5

        def _wrapped_goto() -> None:
            try:
                asyncio.run(
                    self._goto_async(
                        x=x,
                        y=y,
                        theta=theta,
                        timeout=timeout,
                        tolerance=tolerance,
                    ),
                )
            except Exception as e:
                exc_queue.put(e)

        with ThreadPoolExecutor() as exec:
            exec.submit(_wrapped_goto)
        if not exc_queue.empty():
            raise exc_queue.get()

    async def _goto_async(
        self,
        x: float,
        y: float,
        theta: float,
        timeout: float,
        tolerance: Dict[str, float] = {"delta_x": 0.1, "delta_y": 0.1, "delta_theta": 15, "distance": 0.1},
    ) -> None:
        """Async version of the goto method."""
        for pos, value in {"x": x, "y": y}.items():
            if abs(value) > self._max_xy_goto:
                raise ValueError(f"The asbolute value of {pos} should not be more than {self._max_xy_goto}!")

        req = GoToVector(
            x_goal=FloatValue(value=x),
            y_goal=FloatValue(value=y),
            theta_goal=FloatValue(value=deg2rad(theta)),
        )
        self._mobility_stub.SendGoTo(req)

        arrived = await self._is_arrived_in_given_time(time.time(), timeout, tolerance)

        if not arrived and self.lidar.obstacle_detection_status == "OBJECT_DETECTED_STOP":
            # Error type must be modified
            raise ValueError("Target not reached. Mobile base stopped because of obstacle.")

    async def _is_arrived_in_given_time(self, starting_time: float, timeout: float, tolerance: Dict[str, float]) -> bool:
        arrived: bool = False
        while time.time() - starting_time < timeout:
            arrived = True
            distance_to_goal = self._distance_to_goto_goal()
            for delta_key in tolerance.keys():
                if tolerance[delta_key] < abs(distance_to_goal[delta_key]):
                    arrived = False
                    break
            await asyncio.sleep(0.1)
            if arrived:
                break
        return arrived

    def _distance_to_goto_goal(self) -> Dict[str, float]:
        response = self._mobility_stub.DistanceToGoal(Empty())
        distance = {
            "delta_x": round(response.delta_x.value, 3),
            "delta_y": round(response.delta_y.value, 3),
            "delta_theta": round(rad2deg(response.delta_theta.value), 3),
            "distance": round(response.distance.value, 3),
        }
        return distance

    def turn_on(self) -> None:
        """Stop the mobile base immediately by changing its drive mode to 'brake'."""
        self._set_drive_mode("brake")

    def turn_off(self) -> None:
        """Set the mobile base in free wheel mode."""
        self._set_drive_mode("free_wheel")

    def is_on(self) -> bool:
        """Return True if the mobile base is not compliant."""
        return not self._drive_mode == "free_wheel"

    def is_off(self) -> bool:
        """Return True if the mobile base is compliant."""
        if self._drive_mode == "free_wheel":
            return True
        return False

    def _update_with(self, new_state: MobileBaseState) -> None:
        self._battery_level = new_state.battery_level.level.value
        self.lidar._update_with(new_state.lidar_safety)
        self._drive_mode = ZuuuModePossiblities.keys()[new_state.zuuu_mode.mode].lower()
        self._control_mode = ControlModePossiblities.keys()[new_state.control_mode.mode].lower()

    def _update_audit_status(self, new_status: MobileBaseStatus) -> None:
        pass
