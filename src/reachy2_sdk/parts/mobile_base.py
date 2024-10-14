"""Reachy MobileBase module.

Handles all specific methods to a MobileBase.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Dict, Optional

import grpc
import numpy as np
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
    """MobileBase class for controlling Reachy's mobile base.

    This class provides methods to interact with and control the mobile base of a Reachy robot. It allows
    users to access essential information such as battery voltage and odometry, as well as send commands
    to move the base to specified positions or velocities. The class supports different drive modes and
    control modes, and provides methods for resetting the base's odometry.

    Attributes:
        lidar: Lidar object for handling safety features.
    """

    def __init__(
        self,
        mb_msg: MobileBase_proto,
        initial_state: MobileBaseState,
        grpc_channel: grpc.Channel,
    ) -> None:
        """Initialize the MobileBase with its gRPC communication and configuration.

        This sets up the gRPC communication channel and service stubs for controlling the
        mobile base, initializes the drive and control modes.
        It also sets up the LIDAR safety monitoring.

        Args:
            mb_msg: A MobileBase_proto message containing the configuration details for the mobile base.
            initial_state: The initial state of the mobile base, as a MobileBaseState object.
            grpc_channel: The gRPC channel used to communicate with the mobile base service.
        """
        self._logger = logging.getLogger(__name__)
        super().__init__(mb_msg, grpc_channel, MobileBaseUtilityServiceStub(grpc_channel))

        self._mobility_stub = MobileBaseMobilityServiceStub(grpc_channel)

        self._drive_mode: str = ZuuuModePossiblities.keys()[initial_state.zuuu_mode.mode].lower()
        self._control_mode: str = ControlModePossiblities.keys()[initial_state.control_mode.mode].lower()

        self._max_xy_vel = 1.0
        self._max_rot_vel = 180.0
        self._max_xy_goto = 1.0

        self.lidar = Lidar(initial_state.lidar_safety, grpc_channel, self)

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
        """Return the battery voltage.

        The battery should be recharged if the voltage reaches 24.5V or below. If the battery level is low,
        a warning message is logged.

        Returns:
            The current battery voltage as a float, rounded to one decimal place.
        """
        battery_level = float(round(self._battery_level, 1))
        if battery_level < 24.5:
            self._logger.warning(f"Low battery level: {battery_level}V. Consider recharging.")
        return float(round(self._battery_level, 1))

    @property
    def odometry(self) -> Dict[str, float]:
        """Return the odometry of the base.

        The odometry includes the x and y positions in meters and theta in degrees, along with the
        velocities in the x, y directions in meters per degrees and the angular velocity in degrees per second.

        Returns:
            A dictionary containing the current odometry with keys 'x', 'y', 'theta', 'vx', 'vy', and 'vtheta',
            each rounded to three decimal places.
        """
        response = self._stub.GetOdometry(self._part_id)
        odom = {
            "x": response.x.value,
            "y": response.y.value,
            "theta": rad2deg(response.theta.value),
            "vx": response.vx.value,
            "vy": response.vy.value,
            "vtheta": rad2deg(response.vtheta.value),
        }
        return odom

    @property
    def last_cmd_vel(self) -> Dict[str, float]:
        """Return the last command velocity sent to the base.

        The velocity includes the x and y components in meters per second and the theta component in degrees per second.

        Returns:
            A dictionary containing the last command velocity with keys 'x', 'y', and 'theta',
            each rounded to three decimal places.
        """
        response = self._mobility_stub.GetLastDirection(self._part_id)
        cmd_vel = {
            "x": round(response.x.value, 3),
            "y": round(response.y.value, 3),
            "theta": round(rad2deg(response.theta.value), 3),
        }
        return cmd_vel

    def _set_drive_mode(self, mode: str) -> None:
        """Set the base's drive mode.

        The drive mode must be one of the allowed modes, excluding 'speed' and 'goto'. If the mode is
        valid, the base's drive mode is set accordingly.

        Args:
            mode: The desired drive mode as a string. Possible drive modes are:
                ['cmd_vel', 'brake', 'free_wheel', 'emergency_stop', 'cmd_goto'].

        Raises:
            ValueError: If the specified drive mode is not one of the allowed modes.
        """
        all_drive_modes = [mode.lower() for mode in ZuuuModePossiblities.keys()][1:]
        possible_drive_modes = [mode for mode in all_drive_modes if mode not in ("speed", "goto")]
        if mode in possible_drive_modes:
            req = ZuuuModeCommand(mode=getattr(ZuuuModePossiblities, mode.upper()))
            self._stub.SetZuuuMode(req)
            self._drive_mode = mode
        else:
            raise ValueError(f"Drive mode requested should be in {possible_drive_modes}!")

    def _set_control_mode(self, mode: str) -> None:
        """Set the base's control mode.

        The control mode must be one of the allowed modes. If the mode is valid, the base's control mode is set accordingly.

        Args:
            mode: The desired control mode as a string. Possible control modes are: ['open_loop', 'pid']

        Raises:
            ValueError: If the specified control mode is not one of the allowed modes.
        """
        possible_control_modes = [mode.lower() for mode in ControlModePossiblities.keys()][1:]
        if mode in possible_control_modes:
            req = ControlModeCommand(mode=getattr(ControlModePossiblities, mode.upper()))
            self._stub.SetControlMode(req)
            self._control_mode = mode
        else:
            raise ValueError(f"Control mode requested should be in {possible_control_modes}!")

    def is_on(self) -> bool:
        """Check if the mobile base is currently stiff (not in free-wheel mode).

        Returns:
            `True` if the mobile base is not compliant (stiff), `False` otherwise.
        """
        return not self._drive_mode == "free_wheel"

    def is_off(self) -> bool:
        """Check if the mobile base is currently compliant (in free-wheel mode).

        Returns:
            True if the mobile base is compliant (in free-wheel mode), `False` otherwise.
        """
        if self._drive_mode == "free_wheel":
            return True
        return False

    def get_current_odometry(self, degrees: bool = True) -> Dict[str, float]:
        """Get the current odometry of the mobile base in its reference frame.

        Args:
            degrees (bool, optional): Whether to return the orientation (`theta` and `vtheta`) in degrees. 
                                    Defaults to True.
        
        Returns:
            Dict[str, float]: A dictionary containing the current odometry of the mobile base with:
                - 'x': Position along the x-axis (in meters).
                - 'y': Position along the y-axis (in meters).
                - 'theta': Orientation (in degrees by default, radians if `degrees=False`).
                - 'vx': Linear velocity along the x-axis (in meters per second).
                - 'vy': Linear velocity along the y-axis (in meters per second).
                - 'vtheta': Angular velocity (in degrees per second by default, radians if `degrees=False`).
        """
        current_state = self.odometry.copy()
        if not degrees:
            current_state["theta"] = deg2rad(current_state["theta"])
            current_state["vtheta"] = deg2rad(current_state["vtheta"])

        return current_state

    def goto(
        self,
        x: float,
        y: float,
        theta: float,
        timeout: Optional[float] = None,
        tolerance: Dict[str, float] = {"delta_x": 0.05, "delta_y": 0.05, "delta_theta": 5, "distance": 0.05},
    ) -> None:
        """Send the mobile base to a specified target position.

        The (x, y) coordinates define the position in Cartesian space, and theta specifies the orientation in degrees.
        The zero position is set when the mobile base is started or when the `reset_odometry` method is called. A timeout
        can be provided to avoid the mobile base getting stuck. The tolerance values define the acceptable margins for
        reaching the target position.

        Args:
            x: The target x-coordinate in meters.
            y: The target y-coordinate in meters.
            theta: The target orientation in degrees.
            timeout: Optional; the maximum time allowed to reach the target, in seconds.
            tolerance: A dictionary specifying the tolerances for x, y, theta, and overall distance to
                consider the target reached. Defaults to {"delta_x": 0.05, "delta_y": 0.05, "delta_theta": 5, "distance": 0.05}.

        Raises:
            ValueError: If the target is not reached and the mobile base is stopped due to an obstacle.
        """
        if self.is_off():
            self._logger.warning("Mobile base is off. Goto not sent.")
            return

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
        tolerance: Dict[str, float] = {"delta_x": 0.05, "delta_y": 0.05, "delta_theta": 5, "distance": 0.05},
    ) -> None:
        """Async version of the `goto` method.

        This method sends the mobile base to the specified target asynchronously.

        Args:
            x: The target x-coordinate in meters.
            y: The target y-coordinate in meters.
            theta: The target orientation in degrees.
            timeout: The maximum time allowed to reach the target, in seconds.
            tolerance: A dictionary specifying the tolerances for x, y, theta, and overall distance to
                consider the target reached.
        """
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
        """Check if the mobile base arrived at the goal within the given time.

        This method periodically checks the distance to the goal and determines if the mobile base
        reaches the specified position and orientation within the allowed time and tolerance.

        Args:
            starting_time: The time when the checking started, in seconds.
            timeout: The maximum time allowed to reach the target, in seconds.
            tolerance: A dictionary specifying the tolerances for x, y, theta, and overall distance to
                consider the target reached.

        Returns:
            True if the mobile base reaches the target within the time limit, otherwise False.
        """
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
        """Get the distance to the current goto goal.

        The distances returned include delta_x, delta_y, delta_theta, and overall distance.

        Returns:
            A dictionary containing the distance values to the goal, with keys 'delta_x', 'delta_y',
            'delta_theta', and 'distance', all rounded to three decimal places.
        """
        response = self._mobility_stub.DistanceToGoal(self._part_id)
        distance = {
            "delta_x": round(response.delta_x.value, 3),
            "delta_y": round(response.delta_y.value, 3),
            "delta_theta": round(rad2deg(response.delta_theta.value), 3),
            "distance": round(response.distance.value, 3),
        }
        return distance

    def translate_by(self, x: float, y: float, timeout: Optional[float] = None) -> None:
        """Send a target position relative to the current position of the mobile base.

        The (x, y) coordinates specify the desired translation in the mobile base's Cartesian space.

        Args:
            x: The desired translation along the x-axis in meters.
            y: The desired translation along the y-axis in meters.
            timeout: An optional timeout for reaching the target position, in seconds.
        """
        odometry = self.odometry
        x_current = odometry["x"]
        y_current = odometry["y"]
        theta = odometry["theta"]
        theta_rad = deg2rad(theta)
        x_goal = x_current + (x * np.cos(theta_rad) - y * np.sin(theta_rad))
        y_goal = y_current + (x * np.sin(theta_rad) + y * np.cos(theta_rad))
        self.goto(x_goal, y_goal, theta, timeout=timeout)

    def rotate_by(self, theta: float, timeout: Optional[float] = None) -> None:
        """Send a target rotation relative to the current rotation of the mobile base.

        The theta parameter defines the desired rotation in degrees.

        Args:
            theta: The desired rotation in degrees, relative to the current orientation.
            timeout: An optional timeout for completing the rotation, in seconds.
        """
        odometry = self.odometry
        x = odometry["x"]
        y = odometry["y"]
        theta = odometry["theta"] + theta
        self.goto(x, y, theta, timeout=timeout)

    def reset_odometry(self) -> None:
        """Reset the odometry.

        This method resets the mobile base's odometry, so that the current position is now (x, y, theta) = (0, 0, 0).
        """
        self._stub.ResetOdometry(self._part_id)
        time.sleep(0.03)

    def set_goal_speed(self, x: float | int = 0, y: float | int = 0, theta: float | int = 0) -> None:
        """Set the goal speed for the mobile base.

        This method sets the target velocities for the mobile base's movement along the x and y axes, as well as 
        its rotational speed. The actual movement is executed after calling `send_speed_command`.

        Args:
            x (float | int, optional): Linear velocity along the x-axis in meters per second. Defaults to 0.
            y (float | int, optional): Linear velocity along the y-axis in meters per second. Defaults to 0.
            theta (float | int, optional): Rotational velocity (around the z-axis) in degrees per second. Defaults to 0.

        Raises:
            TypeError: If any of the velocity values (`x`, `y`, `theta`) are not of type `float` or `int`.

        Notes:
            - Use `send_speed_command` after this method to execute the movement.
            - The velocities will be used to command the mobile base for a short duration (0.2 seconds).
        """
        for vel in [x, y, theta]:
            if not isinstance(vel, float) | isinstance(vel, int):
                raise TypeError("goal_speed must be a float or int")

        self._x_vel_goal = x
        self._y_vel_goal = y
        self._rot_vel_goal = theta

    def send_speed_command(self) -> None:
        """Send the speed command to the mobile base, based on previously set goal speeds.

        This method sends the velocity commands for the mobile base that were set with `set_goal_speed`. 
        The command will be executed for a duration of 200ms, which is predefined at the ROS level of the mobile base code.

        Raises:
            ValueError: If the absolute value of `x_vel`, `y_vel`, or `rot_vel` exceeds the configured maximum values.
            Warning: If the mobile base is off, no command is sent, and a warning is logged.

        Notes:
            - This method is optimal for sending frequent speed instructions to the mobile base.
            - The goal velocities must be set with `set_goal_speed` before calling this function.
        """
        if self.is_off():
            self._logger.warning(f"{self._part_id.name} is off. speed_command not sent.")
            return
        for vel, value in {"x_vel": self._x_vel_goal, "y_vel": self._y_vel_goal}.items():
            if abs(value) > self._max_xy_vel:
                raise ValueError(f"The absolute value of {vel} should not be more than {self._max_xy_vel}!")

        if abs(self._rot_vel_goal) > self._max_rot_vel:
            raise ValueError(f"The absolute value of rot_vel should not be more than {self._max_rot_vel}!")

        if self._drive_mode != "cmd_vel":
            self._set_drive_mode("cmd_vel")

        req = TargetDirectionCommand(
            direction=DirectionVector(
                x=FloatValue(value=self._x_vel_goal),
                y=FloatValue(value=self._y_vel_goal),
                theta=FloatValue(value=deg2rad(self._rot_vel_goal)),
            )
        )
        self._mobility_stub.SendDirection(req)

    def _update_with(self, new_state: MobileBaseState) -> None:
        """Update the mobile base's state with newly received data from the gRPC server.

        This method updates the battery level, LIDAR safety information, drive mode, and control mode
        of the mobile base.

        Args:
            new_state: The new state of the mobile base, as a MobileBaseState object.
        """
        self._battery_level = new_state.battery_level.level.value
        self.lidar._update_with(new_state.lidar_safety)
        self._drive_mode = ZuuuModePossiblities.keys()[new_state.zuuu_mode.mode].lower()
        self._control_mode = ControlModePossiblities.keys()[new_state.control_mode.mode].lower()

    def _update_audit_status(self, new_status: MobileBaseStatus) -> None:
        """Update the audit status of the mobile base.

        This is a placeholder method and does not perform any actions.

        Args:
            new_status: The new status of the mobile base, as a MobileBaseStatus object.
        """
        pass

    def _set_speed_limits(self, value: int) -> None:
        """Set the speed limits for the mobile base.

        This method overrides the base class implementation to set speed limits.

        Args:
            value: The speed limit value to be set, as an integer.
        """
        return super()._set_speed_limits(value)
