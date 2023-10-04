from grpc import Channel
from reachy_sdk_api_v2.orbita3d_pb2 import (
    Orbita3DField,
    Orbita3DStateRequest,
)

from reachy_sdk_api_v2.component_pb2 import ComponentId
from reachy_sdk_api_v2.orbita3d_pb2_grpc import Orbita3DServiceStub
from .orbita_utils import OrbitaJoint


class Orbita3d:
    def __init__(self, name: str, grpc_channel: Channel):
        self.name = name
        self._stub = Orbita3DServiceStub(grpc_channel)

        init_state = {
            "present_position": 20.0,
            "present_speed": 0.0,
            "present_load": 0.0,
            "temperature": 0.0,
            "goal_position": 100.0,
            "speed_limit": 0.0,
            "torque_limit": 0.0,
        }

        self.roll = OrbitaJoint(initial_state=init_state.copy(), axis_type="roll")
        self.pitch = OrbitaJoint(initial_state=init_state.copy(), axis_type="pitch")
        self.yaw = OrbitaJoint(initial_state=init_state.copy(), axis_type="yaw")

        self.compliant = False

    # TODO: perform the update in a thread
    # TODO: find a smarter way to do this
    def update_3dstate(self) -> None:
        resp = self._stub.GetState(
            Orbita3DStateRequest(
                id=ComponentId(id=self.name),
                fields=[
                    Orbita3DField.PRESENT_POSITION,
                    Orbita3DField.PRESENT_SPEED,
                    Orbita3DField.PRESENT_LOAD,
                    Orbita3DField.TEMPERATURE,
                    Orbita3DField.GOAL_POSITION,
                    Orbita3DField.SPEED_LIMIT,
                    Orbita3DField.TORQUE_LIMIT,
                ],
            )
        )

        self.roll.present_position = resp.present_position.roll
        self.pitch.present_position = resp.present_position.pitch
        self.yaw.present_position = resp.present_position.yaw

        self.roll.present_speed = resp.present_speed.roll
        self.pitch.present_speed = resp.present_speed.pitch
        self.yaw.present_speed = resp.present_speed.yaw

        self.roll.present_load = resp.present_load.roll
        self.pitch.present_load = resp.present_load.pitch
        self.yaw.present_load = resp.present_load.yaw

        self.roll.goal_position = resp.goal_position.roll
        self.pitch.goal_position = resp.goal_position.pitch
        self.yaw.goal_position = resp.goal_position.yaw

        self.roll.speed_limit = resp.speed_limit.roll
        self.pitch.speed_limit = resp.speed_limit.pitch
        self.yaw.speed_limit = resp.speed_limit.yaw

        self.roll.torque_limit = resp.torque_limit.roll
        self.pitch.torque_limit = resp.torque_limit.pitch
        self.yaw.torque_limit = resp.present_position.yaw
