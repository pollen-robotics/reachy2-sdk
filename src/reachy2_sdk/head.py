"""Reachy Head module.

Handles all specific method to an Head:
- the inverse kinematics
- look_at function
"""
from typing import Dict, Optional, Tuple

import grpc
from google.protobuf.wrappers_pb2 import FloatValue
from pyquaternion import Quaternion as pyQuat
from reachy2_sdk_api.head_pb2 import Head as Head_proto
from reachy2_sdk_api.head_pb2 import (
    HeadLookAtGoal,
    HeadPosition,
    HeadState,
    NeckFKRequest,
    NeckIKRequest,
    NeckOrientation,
)
from reachy2_sdk_api.head_pb2_grpc import HeadServiceStub
from reachy2_sdk_api.kinematics_pb2 import ExtEulerAngles, Point, Quaternion, Rotation3d
from reachy2_sdk_api.part_pb2 import PartId

from .orbita3d import Orbita3d

# from .dynamixel_motor import DynamixelMotor


class Head:
    """Head class.

    It exposes the neck orbita actuator at the base of the head.
    It provides look_at utility function to directly orient the head so it looks at a cartesian point
    expressed in Reachy's coordinate system.
    """

    def __init__(self, head_msg: Head_proto, initial_state: HeadState, grpc_channel: grpc.Channel) -> None:
        """Initialize the head with its actuators."""
        self._grpc_channel = grpc_channel
        self._head_stub = HeadServiceStub(grpc_channel)
        self.part_id = PartId(id=head_msg.part_id.id, name=head_msg.part_id.name)

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
        self.neck = Orbita3d(
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
        return f"""<Head actuators=\n\t{
            s
        }\n>"""

    def get_orientation(self) -> pyQuat:
        """Get the current orientation of the head.

        It will return the quaternion (x, y, z, w).
        """
        quat = self._head_stub.GetOrientation(self.part_id).q
        return pyQuat(w=quat.w, x=quat.x, y=quat.y, z=quat.z)

    def forward_kinematics(self, rpy_position: Optional[Tuple[float, float, float]] = None) -> pyQuat:
        """Compute the forward kinematics of the head.

        It will return the quaternion (x, y, z, w).
        You can either specify a given joints position, otherwise it will use the current robot position.
        """
        if rpy_position is None:
            return self.get_orientation()
        else:
            req = NeckFKRequest(
                id=self.part_id,
                position=HeadPosition(
                    neck_position=Rotation3d(
                        rpy=ExtEulerAngles(
                            roll=rpy_position[0],
                            pitch=rpy_position[1],
                            yaw=rpy_position[2],
                        )
                    )
                ),
            )
            res = self._head_stub.ComputeNeckFK(req)
            quat = res.orientation.q
            return pyQuat(w=quat.w, x=quat.x, y=quat.y, z=quat.z)

    def inverse_kinematics(
        self,
        orientation: Optional[pyQuat] = None,
        rpy_q0: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[float, float, float]:
        """Compute the inverse kinematics of the arm.

        Given a goal quaternion (x, y, z, w)
        it will try to compute a joint solution to reach this target (or get close).

        It will raise a ValueError if no solution is found.

        You can also specify a basic joint configuration as a prior for the solution.
        """
        req_params = {
            "id": self.part_id,
        }
        if orientation is not None:
            req_params["target"] = NeckOrientation(
                q=Quaternion(
                    w=orientation.w,
                    x=orientation.x,
                    y=orientation.y,
                    z=orientation.z,
                )
            )
        if rpy_q0 is not None:
            req_params["q0"] = Rotation3d(rpy=ExtEulerAngles(roll=rpy_q0[0], pitch=rpy_q0[1], yaw=rpy_q0[2]))
        req = NeckIKRequest(**req_params)
        rpy_pos = self._head_stub.ComputeNeckIK(req)
        return (
            rpy_pos.position.rpy.roll,
            rpy_pos.position.rpy.pitch,
            rpy_pos.position.rpy.yaw,
        )

    def look_at(self, x: float, y: float, z: float, duration: float) -> None:
        """Compute and send neck rpy position to look at the (x, y, z) point in Reachy cartesian space (torso frame).

        X is forward, Y is left and Z is upward. They all expressed in meters.
        """
        req = HeadLookAtGoal(id=self.part_id, point=Point(x=x, y=y, z=z), duration=FloatValue(value=duration))
        self._head_stub.LookAt(req)

    def turn_on(self) -> None:
        """Turn all motors of the part on.

        All head's motors will then be stiff.
        """
        self._head_stub.TurnOn(self.part_id)

    def turn_off(self) -> None:
        """Turn all motors of the part off.

        All head's motors will then be compliant.
        """
        self._head_stub.TurnOff(self.part_id)

    def _update_with(self, new_state: HeadState) -> None:
        """Update the head with a newly received (partial) state received from the gRPC server."""
        self.neck._update_with(new_state.neck_state)
        # self.l_antenna._update_with(new_state.l_antenna_state)
        # self.r_antenna._update_with(new_state.r_antenna_state)

    @property
    def compliant(self) -> Dict[str, bool]:
        """Get compliancy of all the part's actuators"""
        return {"neck": self.neck.compliant}  # , "l_antenna": self.l_antenna.compliant, "r_antenna": self.r_antenna.compliant}

    @compliant.setter
    def compliant(self, value: bool) -> None:
        """Set compliancy of all the part's actuators"""
        if not isinstance(value, bool):
            raise ValueError("Expecting bool as compliant value")
        if value:
            self._head_stub.TurnOff(self.part_id)
        else:
            self._head_stub.TurnOn(self.part_id)