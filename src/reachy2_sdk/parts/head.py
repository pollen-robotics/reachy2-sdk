"""Reachy Head module.

Handles all specific method to an Head:
- the inverse kinematics
- look_at function
"""
from typing import List

import grpc
import numpy as np
from google.protobuf.wrappers_pb2 import FloatValue
from pyquaternion import Quaternion as pyQuat
from reachy2_sdk_api.goto_pb2 import (
    CartesianGoal,
    GoToAck,
    GoToId,
    GoToRequest,
    JointsGoal,
)
from reachy2_sdk_api.goto_pb2_grpc import GoToServiceStub
from reachy2_sdk_api.head_pb2 import Head as Head_proto
from reachy2_sdk_api.head_pb2 import (
    HeadState,
    NeckCartesianGoal,
    NeckJointGoal,
    NeckOrientation,
)
from reachy2_sdk_api.head_pb2_grpc import HeadServiceStub
from reachy2_sdk_api.kinematics_pb2 import ExtEulerAngles, Point, Quaternion, Rotation3d
from reachy2_sdk_api.part_pb2 import PartId

from ..orbita.orbita3d import Orbita3d
from ..orbita.orbita_joint import OrbitaJoint
from ..utils.custom_dict import CustomDict
from ..utils.utils import get_grpc_interpolation_mode

# from .dynamixel_motor import DynamixelMotor


class Head:
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
        self._grpc_channel = grpc_channel
        self._goto_stub = goto_stub
        self._head_stub = HeadServiceStub(grpc_channel)
        self._part_id = PartId(id=head_msg.part_id.id, name=head_msg.part_id.name)

        self._setup_head(head_msg, initial_state)
        self._actuators = {
            "neck": self.neck,
            # "l_antenna" : self.l_antenna,
            # r_antenna" : self.r_antenna,
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
        # self.l_antenna = DynamixelMotor(
        #     uid=description.l_antenna.id.id,
        #     name=description.l_antenna.id.name,
        #     initial_state=initial_state.l_antenna_state,
        #     grpc_channel=self._grpc_channel,
        # )
        # self.r_antenna = DynamixelMotor(
        #     uid=description.r_antenna.id.id,
        #     name=description.r_antenna.id.name,
        #     initial_state=initial_state.r_antenna_state,
        #     grpc_channel=self._grpc_channel,
        # )

    def __repr__(self) -> str:
        """Clean representation of an Head."""
        s = "\n\t".join([act_name + ": " + str(actuator) for act_name, actuator in self._actuators.items()])
        return f"""<Head on={self.is_on()} actuators=\n\t{
            s
        }\n>"""

    @property
    def neck(self) -> Orbita3d:
        return self._neck

    @property
    def joints(self) -> CustomDict[str, OrbitaJoint]:
        """Get all the arm's joints."""
        _joints: CustomDict[str, OrbitaJoint] = CustomDict({})
        for actuator_name, actuator in self._actuators.items():
            for joint in actuator._joints.values():
                _joints[actuator_name + "." + joint._axis_type] = joint
        return _joints

    def get_orientation(self) -> pyQuat:
        """Get the current orientation of the head.

        It will return the quaternion (x, y, z, w).
        """
        quat = self._head_stub.GetOrientation(self._part_id).q
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

    def turn_on(self) -> None:
        """Turn all motors of the part on.

        All head's motors will then be stiff.
        """
        self._head_stub.TurnOn(self._part_id)

    def turn_off(self) -> None:
        """Turn all motors of the part off.

        All head's motors will then be compliant.
        """
        self._head_stub.TurnOff(self._part_id)

    def is_on(self) -> bool:
        """Return True if all actuators of the arm are stiff"""
        for actuator in self._actuators.values():
            if not actuator.is_on():
                return False
        return True

    def is_off(self) -> bool:
        """Return True if all actuators of the arm are stiff"""
        for actuator in self._actuators.values():
            if actuator.is_on():
                return False
        return True

    def get_move_playing(self) -> GoToId:
        """Return the id of the goto currently playing on the head"""
        response = self._goto_stub.GetPartGoToPlaying(self._part_id)
        return response

    def get_moves_queue(self) -> List[GoToId]:
        """Return the list of all goto ids waiting to be played on the head"""
        response = self._goto_stub.GetPartGoToQueue(self._part_id)
        return [goal_id for goal_id in response.goto_ids]

    def cancel_all_moves(self) -> GoToAck:
        """Ask the cancellation of all waiting goto on the head"""
        response = self._goto_stub.CancelPartAllGoTo(self._part_id)
        return response

    def _update_with(self, new_state: HeadState) -> None:
        """Update the head with a newly received (partial) state received from the gRPC server."""
        self.neck._update_with(new_state.neck_state)
        # self.l_antenna._update_with(new_state.l_antenna_state)
        # self.r_antenna._update_with(new_state.r_antenna_state)
