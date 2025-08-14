"""Reachy MobileBase module.

Handles all specific methods to a MobileBase.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import grpc
import numpy as np
from google.protobuf.wrappers_pb2 import FloatValue
from numpy import deg2rad, rad2deg, round
from reachy2_sdk_api.goto_pb2 import GoToId, GoToRequest, OdometryGoal
from reachy2_sdk_api.goto_pb2_grpc import GoToServiceStub
from reachy2_sdk_api.mobile_base_mobility_pb2 import (
    DirectionVector,
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
from .goto_based_part import IGoToBasedPart
from .part import Part


class MobileBase(Part, IGoToBasedPart):
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
        goto_stub: GoToServiceStub,
    ) -> None:
        """Initialize the MobileBase with its gRPC communication and configuration.

        This sets up the gRPC communication channel and service stubs for controlling the
        mobile base, initializes the drive and control modes.
        It also sets up the LIDAR safety monitoring.

        Args:
            mb_msg: A MobileBase_proto message containing the configuration details for the mobile base.
            initial_state: The initial state of the mobile base, as a MobileBaseState object.
            grpc_channel: The gRPC channel used to communicate with the mobile base service.
            goto_stub: The gRPC service stub for the GoTo service.
        """
        self._logger = logging.getLogger(__name__)
        super().__init__(mb_msg, grpc_channel, MobileBaseUtilityServiceStub(grpc_channel))
        IGoToBasedPart.__init__(self, self._part_id, goto_stub)

        self._mobility_stub = MobileBaseMobilityServiceStub(grpc_channel)

        self._drive_mode: str = ZuuuModePossiblities.keys()[initial_state.zuuu_mode.mode].lower()
        self._control_mode: str = ControlModePossiblities.keys()[initial_state.control_mode.mode].lower()
        self._battery_level = 30.0

        self._max_xy_vel = 0.61
        self._max_rot_vel = 114.0
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
            "vx": round(response.x.value, 3),
            "vy": round(response.y.value, 3),
            "vtheta": round(rad2deg(response.theta.value), 3),
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
        wait: bool = False,
        degrees: bool = True,
        distance_tolerance: Optional[float] = 0.05,
        angle_tolerance: Optional[float] = None,
        timeout: float = 100,
    ) -> GoToId:
        """Send the mobile base to a specified target position.

        The (x, y) coordinates define the position in Cartesian space, and theta specifies the orientation in degrees.
        The zero position is set when the mobile base is started or when the `reset_odometry` method is called. A timeout
        can be provided to avoid the mobile base getting stuck. The tolerance values define the acceptable margins for
        reaching the target position.

        Args:
            x: The target x-coordinate in meters.
            y: The target y-coordinate in meters.
            theta: The target orientation in degrees.
            wait: If True, the function waits until the movement is completed before returning.
                    Defaults to False.
            degrees: If True, the theta value and angle_tolerance are treated as degrees.
                    Defaults to True.
            distance_tolerance: Optional; the tolerance to the target position to consider the goto finished, in meters.
            angle_tolerance: Optional; the angle tolerance to the target to consider the goto finished, in meters.
            timeout: Optional; the maximum time allowed to reach the target, in seconds.

        Returns:
            GoToId: The unique GoToId identifier for the movement command.

        Raises:
            TypeError: If the target is not reached and the mobile base is stopped due to an obstacle.
        """
        if self.is_off():
            self._logger.warning("Mobile base is off. Goto not sent.")
            return

        self._check_goto_parameters(target=[x, y, theta])
        self._check_optional_goto_parameters(distance_tolerance, angle_tolerance, timeout)

        if degrees:
            theta = deg2rad(theta)
            if angle_tolerance is not None:
                angle_tolerance = deg2rad(angle_tolerance)

        if angle_tolerance is None:
            angle_tolerance = deg2rad(5.0)

        vector_goal = TargetDirectionCommand(
            id=self._part_id,
            direction=DirectionVector(
                x=FloatValue(value=x),
                y=FloatValue(value=y),
                theta=FloatValue(value=theta),
            ),
        )

        odometry_goal = OdometryGoal(
            odometry_goal=vector_goal,
            distance_tolerance=FloatValue(value=distance_tolerance),
            angle_tolerance=FloatValue(value=angle_tolerance),
            timeout=FloatValue(value=timeout),
        )

        request = GoToRequest(
            odometry_goal=odometry_goal,
        )

        response = self._goto_stub.GoToOdometry(request)

        if response.id == -1:
            self._logger.error(f"Unable to go to requested position x={x}, y={y}, theta={theta}. No command sent.")
        elif wait:
            self._wait_goto(response, timeout)

        return response

    def translate_by(
        self,
        x: float,
        y: float,
        wait: bool = False,
        distance_tolerance: Optional[float] = 0.05,
        timeout: float = 100,
    ) -> GoToId:
        """Send a target position relative to the current position of the mobile base.

        The (x, y) coordinates specify the desired translation in the mobile base's Cartesian space.

        Args:
            x: The desired translation along the x-axis in meters.
            y: The desired translation along the y-axis in meters.
            wait:  If True, the function waits until the movement is completed before returning.
            distance_tolerance: Optional; The distance tolerance to the target to consider the goto finished, in meters.
            timeout: An optional timeout for reaching the target position, in seconds.

        Returns:
            The GoToId of the movement command, created using the `goto` method.
        """
        try:
            goto = self.get_goto_queue()[-1]
        except IndexError:
            goto = self.get_goto_playing()

        if goto.id != -1:
            odom_request = self._get_goto_request(goto)
        else:
            odom_request = None

        angle_tolerance = None

        if odom_request is not None:
            base_odom = odom_request.request.target
            angle_tolerance = deg2rad(odom_request.request.angle_tolerance)
        else:
            base_odom = self.odometry

        theta_goal = deg2rad(base_odom["theta"])
        x_goal = base_odom["x"] + (x * np.cos(theta_goal) - y * np.sin(theta_goal))
        y_goal = base_odom["y"] + (x * np.sin(theta_goal) + y * np.cos(theta_goal))

        return self.goto(
            x_goal,
            y_goal,
            theta_goal,
            wait=wait,
            distance_tolerance=distance_tolerance,
            angle_tolerance=angle_tolerance,
            degrees=False,
            timeout=timeout,
        )

    def rotate_by(
        self,
        theta: float,
        wait: bool = False,
        degrees: bool = True,
        angle_tolerance: Optional[float] = None,
        timeout: float = 100,
    ) -> GoToId:
        """Send a target rotation relative to the current rotation of the mobile base.

        The theta parameter defines the desired rotation in degrees.

        Args:
            theta: The desired rotation in degrees, relative to the current orientation.
            wait: If True, the function waits until the rotation is completed before returning.
            degrees: If True, the theta value and angle_tolerance are treated as degrees, otherwise as radians.
            angle_tolerance: Optional; The angle tolerance to the target to consider the goto finished.
            timeout: An optional timeout for completing the rotation, in seconds.
        """
        try:
            goto = self.get_goto_queue()[-1]
        except IndexError:
            goto = self.get_goto_playing()

        if goto.id != -1:
            odom_request = self._get_goto_request(goto)
        else:
            odom_request = None

        distance_tolerance = 0.05

        if odom_request is not None:
            base_odom = odom_request.request.target
            if angle_tolerance is None:
                angle_tolerance = odom_request.request.angle_tolerance
            distance_tolerance = odom_request.request.distance_tolerance
        else:
            base_odom = self.odometry

        if degrees:
            theta = base_odom["theta"] + theta
        else:
            theta = deg2rad(base_odom["theta"]) + theta
        x = base_odom["x"]
        y = base_odom["y"]

        return self.goto(
            x,
            y,
            theta,
            wait=wait,
            degrees=degrees,
            distance_tolerance=distance_tolerance,
            angle_tolerance=angle_tolerance,
            timeout=timeout,
        )

    def reset_odometry(self) -> None:
        """Reset the odometry.

        This method resets the mobile base's odometry, so that the current position is now (x, y, theta) = (0, 0, 0).
        If any goto is being played, stop the goto and the queued ones.
        """
        if self.get_goto_playing().id != -1 or len(self.get_goto_queue()) != 0:
            self._logger.warning(
                "Odometry reset requested while goto in progress: aborting the current goto and all queued gotos."
            )
        self._stub.ResetOdometry(self._part_id)
        time.sleep(0.05)

    def set_goal_speed(self, vx: float | int = 0, vy: float | int = 0, vtheta: float | int = 0) -> None:
        """Set the goal speed for the mobile base.

        This method sets the target velocities for the mobile base's movement along the x and y axes, as well as
        its rotational speed. The actual movement is executed after calling `send_speed_command`.

        Args:
            vx (float | int, optional): Linear velocity along the x-axis in meters per second. Defaults to 0.
            vy (float | int, optional): Linear velocity along the y-axis in meters per second. Defaults to 0.
            vtheta (float | int, optional): Rotational velocity (around the z-axis) in degrees per second. Defaults to 0.

        Raises:
            TypeError: If any of the velocity values (`vx`, `vy`, `vtheta`) are not of type `float` or `int`.

        Notes:
            - Use `send_speed_command` after this method to execute the movement.
            - The velocities will be used to command the mobile base for a short duration (0.2 seconds).
        """
        for vel in [vx, vy, vtheta]:
            if not isinstance(vel, float) | isinstance(vel, int):
                raise TypeError("goal_speed must be a float or int")

        self._x_vel_goal = vx
        self._y_vel_goal = vy
        self._rot_vel_goal = vtheta

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
                self._x_vel_goal = self._max_xy_vel * np.sign(value)
                self._logger.warning(
                    f"{vel} value {value} m/s exceeds the allowed limit. Setting maximum of {self._max_xy_vel} m/s."
                )

        if abs(self._rot_vel_goal) > self._max_rot_vel:
            self._rot_vel_goal = self._max_rot_vel * np.sign(self._rot_vel_goal)
            self._logger.warning(
                f"rot_vel value {value} m/s exceeds the allowed limit. Setting maximum of {self._max_rot_vel} m/s."
            )

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
        pass  # pragma: no cover

    def _set_speed_limits(self, value: int) -> None:
        """Set the speed limits for the mobile base.

        This method overrides the base class implementation to set speed limits.

        Args:
            value: The speed limit value to be set, as an integer.
        """
        return super()._set_speed_limits(value)

    def _check_goto_parameters(self, target: Any, duration: Optional[float] = None, q0: Optional[List[float]] = None) -> None:
        """Check the validity of the parameters for the `goto` method.

        Args:
            duration: Not required here.
            target: The target goal, as a list [x, y, theta] in the odometry coordinate system.
            q0: Not required here. Defaults to None.

        Raises:
            TypeError: If the x goal is not a float or int.
            TypeError: If the y goal is not a float or int.
            TypeError: If the theta goal is not a float or int.
        """
        self._check_type_float(target[0], "x")
        self._check_type_float(target[1], "y")
        self._check_type_float(target[2], "theta")

        try:
            goto = self.get_goto_queue()[-1]
        except IndexError:
            goto = self.get_goto_playing()

        if goto.id != -1:
            odom_request = self._get_goto_request(goto)
        else:
            odom_request = None

        if odom_request is not None:
            base_odom = odom_request.request.target
            x_offset = abs(target[0] - base_odom["x"])
            y_offset = abs(target[1] - base_odom["y"])
        else:
            x_offset = abs(target[0] - self.odometry["x"])
            y_offset = abs(target[1] - self.odometry["y"])
        for pos, value in {"x": x_offset, "y": y_offset}.items():
            if abs(value) > self._max_xy_goto:
                raise ValueError(f"The displacement in {pos} should not be more than {self._max_xy_goto}, got {abs(value)}")

    def _check_optional_goto_parameters(
        self, distance_tolerance: Optional[float], angle_tolerance: Optional[float], timeout: Optional[float]
    ) -> None:
        """Check the validity of the optional parameters for the `goto` method.

        Args:
            distance_tolerance: The distance tolerance value to be checked.
            angle_tolerance: The angle tolerance value to be checked.
            timeout: The timeout value to be checked.

        Raises:
            ValueError: If the distance_tolerance is negative.
            ValueError: If the angle_tolerance is negative.
            ValueError: If the timeout is negative or null.
        """
        if distance_tolerance is not None:
            self._check_type_float(distance_tolerance, "distance_tolerance")
            if distance_tolerance < 0:
                raise ValueError(f"distance_tolerance must be a positive value, got {distance_tolerance}")
        if angle_tolerance is not None:
            self._check_type_float(angle_tolerance, "angle_tolerance")
            if angle_tolerance < 0:
                raise ValueError(f"angle_tolerance must be a positive value, got {angle_tolerance}")
        if timeout is not None:
            self._check_type_float(timeout, "timeout")
            if timeout <= 0:
                raise ValueError(f"timeout must be a positive value greater than 0, got {timeout}")

    def _check_type_float(self, value: Any, arg_name: str) -> None:
        """Check the type of the value parameter.

        Args:
            value: The value to be checked.

        Raises:
            TypeError: If the value is not a float or int.
        """
        if not (isinstance(value, float) | isinstance(value, int)):
            raise TypeError(f"{arg_name} must be a float or int, got {type(value)} instead")

    def set_max_xy_goto(self, value: float) -> None:
        """Set the maximum displacement in the x and y directions for the mobile base.

        Args:
            value: The maximum displacement value to be set, in meters.
        """
        self._max_xy_goto = value

    def goto_posture(
        self,
        common_posture: str = "default",
        duration: float = 2,
        wait: bool = False,
        wait_for_goto_end: bool = True,
        interpolation_mode: str = "minimum_jerk",
    ) -> GoToId:
        """Mobile base is not affected by goto_posture. No command is sent."""
        return super().goto_posture(common_posture, duration, wait, wait_for_goto_end, interpolation_mode)
