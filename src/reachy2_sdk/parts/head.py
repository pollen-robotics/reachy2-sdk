"""Reachy Head module.

Handles all specific method to an Head:
- the inverse kinematics
- look_at function
"""

import logging
from typing import List

import grpc
import numpy as np
from google.protobuf.wrappers_pb2 import FloatValue
from pyquaternion import Quaternion as pyQuat
from reachy2_sdk_api.goto_pb2 import CartesianGoal, GoToId, GoToRequest, JointsGoal
from reachy2_sdk_api.goto_pb2_grpc import GoToServiceStub
from reachy2_sdk_api.head_pb2 import Head as Head_proto
from reachy2_sdk_api.head_pb2 import (
    HeadState,
    NeckCartesianGoal,
    NeckJointGoal,
    NeckOrientation,
    SpeedLimitRequest,
    TorqueLimitRequest,
)
from reachy2_sdk_api.head_pb2_grpc import HeadServiceStub
from reachy2_sdk_api.kinematics_pb2 import ExtEulerAngles, Point, Quaternion, Rotation3d

from ..orbita.orbita3d import Orbita3d
from ..utils.utils import get_grpc_interpolation_mode
from .goto_based_part import IGoToBasedPart
from .joints_based_part import JointsBasedPart


class Head(JointsBasedPart, IGoToBasedPart):
    """Head class.

    It exposes the neck orbita actuator at the base of the head.
    It provides look_at utility function to directly orient the head so it looks at a cartesian point
    expressed in Reachy's coordinate system.
    """

    def __init__(
        self,
        head_msg: Head_proto,
        initial_state: HeadState,
        grpc_channel: grpc.Channel,
        goto_stub: GoToServiceStub,
    ) -> None:
        """Initialize the head with its actuators."""
        self._logger = logging.getLogger(__name__)
        JointsBasedPart.__init__(self, head_msg, grpc_channel, HeadServiceStub(grpc_channel))
        IGoToBasedPart.__init__(self, self, goto_stub)

        self._setup_head(head_msg, initial_state)
        self._actuators = {
            "neck": self.neck,
        }

    def _setup_head(self, head: Head_proto, initial_state: HeadState) -> None:
        """Set up the head with its actuators.

        It will create the actuators neck and antennas and set their initial state.
        """
        description = head.description
        self._neck = Orbita3d(
            uid=description.neck.id.id,
            name=description.neck.id.name,
            initial_state=initial_state.neck_state,
            grpc_channel=self._grpc_channel,
        )

    def __repr__(self) -> str:
        """Clean representation of an Head."""
        s = "\n\t".join([act_name + ": " + str(actuator) for act_name, actuator in self._actuators.items()])
        return f"""<Head on={self.is_on()} actuators=\n\t{
            s
        }\n>"""

    @property
    def neck(self) -> Orbita3d:
        return self._neck

    def get_orientation(self) -> pyQuat:
        """Get the current orientation of the head.

        It will return the quaternion (x, y, z, w).
        """
        quat = self._stub.GetOrientation(self._part_id).q
        return pyQuat(w=quat.w, x=quat.x, y=quat.y, z=quat.z)

    def get_joints_positions(self) -> List[float]:
        """Return the current joints positions of the neck.

        It will return the List[roll, pitch, yaw].
        """
        roll = self.neck._joints["roll"].present_position
        pitch = self.neck._joints["pitch"].present_position
        yaw = self.neck._joints["yaw"].present_position
        return [roll, pitch, yaw]

    def look_at(self, x: float, y: float, z: float, duration: float = 2.0, interpolation_mode: str = "minimum_jerk") -> GoToId:
        """Compute and send neck rpy position to look at the (x, y, z) point in Reachy cartesian space (torso frame).

        X is forward, Y is left and Z is upward. They all expressed in meters.
        """
        if duration == 0:
            raise ValueError("duration cannot be set to 0.")
        if not self.neck.is_on():
            self._logger.warning("head.neck is off. No command sent.")
            return GoToId(id=-1)

        request = GoToRequest(
            cartesian_goal=CartesianGoal(
                neck_cartesian_goal=NeckCartesianGoal(
                    id=self._part_id,
                    point=Point(x=x, y=y, z=z),
                    duration=FloatValue(value=duration),
                )
            ),
            interpolation_mode=get_grpc_interpolation_mode(interpolation_mode),
        )
        response = self._goto_stub.GoToCartesian(request)
        return response

    def rotate_to(
        self,
        roll: float,
        pitch: float,
        yaw: float,
        duration: float = 2.0,
        interpolation_mode: str = "minimum_jerk",
        degrees: bool = True,
    ) -> GoToId:
        """Send neck to rpy position.

        Rotation is done in order roll, pitch, yaw.
        """
        if duration == 0:
            raise ValueError("duration cannot be set to 0.")
        if not self.neck.is_on():
            self._logger.warning("head.neck is off. No command sent.")
            return GoToId(id=-1)

        if degrees:
            roll = np.deg2rad(roll)
            pitch = np.deg2rad(pitch)
            yaw = np.deg2rad(yaw)
        request = GoToRequest(
            joints_goal=JointsGoal(
                neck_joint_goal=NeckJointGoal(
                    id=self._part_id,
                    joints_goal=NeckOrientation(
                        rotation=Rotation3d(
                            rpy=ExtEulerAngles(
                                roll=FloatValue(value=roll), pitch=FloatValue(value=pitch), yaw=FloatValue(value=yaw)
                            )
                        )
                    ),
                    duration=FloatValue(value=duration),
                )
            ),
            interpolation_mode=get_grpc_interpolation_mode(interpolation_mode),
        )
        response = self._goto_stub.GoToJoints(request)
        return response

    def orient(self, q: pyQuat, duration: float = 2.0, interpolation_mode: str = "minimum_jerk") -> GoToId:
        """Send neck to the orientation given as a quaternion."""
        if duration == 0:
            raise ValueError("duration cannot be set to 0.")
        if not self.neck.is_on():
            self._logger.warning("head.neck is off. No command sent.")
            return GoToId(id=-1)

        request = GoToRequest(
            joints_goal=JointsGoal(
                neck_joint_goal=NeckJointGoal(
                    id=self._part_id,
                    joints_goal=NeckOrientation(rotation=Rotation3d(q=Quaternion(w=q.w, x=q.x, y=q.y, z=q.z))),
                    duration=FloatValue(value=duration),
                )
            ),
            interpolation_mode=get_grpc_interpolation_mode(interpolation_mode),
        )
        response = self._goto_stub.GoToJoints(request)
        return response

    def set_torque_limits(self, value: int) -> None:
        """Choose percentage of torque max value applied as limit of all head's motors."""
        if not isinstance(value, float | int):
            raise ValueError(f"Expected one of: float, int for torque_limit, got {type(value).__name__}")
        if not (0 <= value <= 100):
            raise ValueError(f"torque_limit must be in [0, 100], got {value}.")
        req = TorqueLimitRequest(
            id=self._part_id,
            limit=value,
        )
        self._stub.SetTorqueLimit(req)

    def set_speed_limits(self, value: int) -> None:
        """Choose percentage of speed max value applied as limit of all head's motors."""
        if not isinstance(value, float | int):
            raise ValueError(f"Expected one of: float, int for speed_limit, got {type(value).__name__}")
        if not (0 <= value <= 100):
            raise ValueError(f"speed_limit must be in [0, 100], got {value}.")
        req = SpeedLimitRequest(
            id=self._part_id,
            limit=value,
        )
        self._stub.SetSpeedLimit(req)

    def send_goal_positions(self) -> None:
        for actuator in self._actuators.values():
            actuator.send_goal_positions()

    def set_pose(
        self,
        wait_for_moves_end: bool = True,
        duration: float = 2,
        interpolation_mode: str = "minimum_jerk",
    ) -> GoToId:
        """Send all joints to standard positions in specified duration.

        Setting wait_for_goto_end to False will cancel all gotos on all parts and immediately send the commands.
        Otherwise, the commands will be sent to a part when all gotos of its queue has been played.
        """
        if not wait_for_moves_end:
            self.cancel_all_moves()
        if self.neck.is_on():
            return self.rotate_to(0, -10, 0, duration, interpolation_mode)
        else:
            self._logger.warning("head.neck is off. No command sent.")
        return GoToId(id=-1)

    def _update_with(self, new_state: HeadState) -> None:
        """Update the head with a newly received (partial) state received from the gRPC server."""
        self.neck._update_with(new_state.neck_state)
