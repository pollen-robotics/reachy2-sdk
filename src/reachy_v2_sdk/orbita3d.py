from grpc import Channel

# from reachy_sdk_api_v2.orbita3d_pb2 import (
#     Orbita3DField,
#     Orbita3DStateRequest,
# )

# from reachy_sdk_api_v2.component_pb2 import ComponentId
from reachy_sdk_api_v2.orbita3d_pb2_grpc import Orbita3DServiceStub
from reachy_sdk_api_v2.orbita3d_pb2 import Orbita3DState
from .orbita_utils import OrbitaAxis


class Orbita3d:
    def __init__(self, name: str, grpc_channel: Channel):
        self.name = name
        self._stub = Orbita3DServiceStub(grpc_channel)

        self.roll = OrbitaAxis("roll")
        self.pitch = OrbitaAxis("pitch")
        self.yaw = OrbitaAxis("yaw")

        self.compliant = False

    # TODO: perform the update in a thread
    # TODO: find a smarter way to do this
    # def update_3dstate(self) -> None:
    #     resp = self._stub.GetState(
    #         Orbita3DStateRequest(
    #             id=ComponentId(id=self.name),
    #             fields=[
    #                 Orbita3DField.PRESENT_POSITION,
    #                 Orbita3DField.PRESENT_SPEED,
    #                 Orbita3DField.PRESENT_LOAD,
    #                 Orbita3DField.TEMPERATURE,
    #                 Orbita3DField.GOAL_POSITION,
    #                 Orbita3DField.SPEED_LIMIT,
    #                 Orbita3DField.TORQUE_LIMIT,
    #             ],
    #         )
    #     )

    #     self.roll._present_position = resp.present_position.roll
    #     self.pitch._present_position = resp.present_position.pitch
    #     self.yaw._present_position = resp.present_position.yaw

    #     self.roll._present_speed = resp.present_speed.roll
    #     self.pitch._present_speed = resp.present_speed.pitch
    #     self.yaw._present_speed = resp.present_speed.yaw

    #     self.roll._present_load = resp.present_load.roll
    #     self.pitch._present_load = resp.present_load.pitch
    #     self.yaw._present_load = resp.present_load.yaw

    #     self.roll._goal_position = resp.goal_position.roll
    #     self.pitch._goal_position = resp.goal_position.pitch
    #     self.yaw._goal_position = resp.goal_position.yaw

    #     self.roll._speed_limit = resp.speed_limit.roll
    #     self.pitch._speed_limit = resp.speed_limit.pitch
    #     self.yaw._speed_limit = resp.speed_limit.yaw

    #     self.roll._torque_limit = resp.torque_limit.roll
    #     self.pitch._torque_limit = resp.torque_limit.pitch
    #     self.yaw._torque_limit = resp.present_position.yaw

    def _update_with(self, new_state: Orbita3DState) -> None:
        """Update the orbita with a newly received (partial) state received from the gRPC server."""
        self.roll._temperature = new_state.temperature.roll
        self.pitch._temperature = new_state.temperature.pitch
        self.yaw._temperature = new_state.temperature.yaw

        self.roll._present_position = new_state.present_position.roll
        self.pitch._present_position = new_state.present_position.pitch
        self.yaw._present_position = new_state.present_position.yaw

        self.roll._present_speed = new_state.present_speed.roll
        self.pitch._present_speed = new_state.present_speed.pitch
        self.yaw._present_speed = new_state.present_speed.yaw

        self.roll._present_load = new_state.present_load.roll
        self.pitch._present_load = new_state.present_load.pitch
        self.yaw._present_load = new_state.present_load.yaw

        self.roll._goal_position = new_state.goal_position.roll
        self.pitch._goal_position = new_state.goal_position.pitch
        self.yaw._goal_position = new_state.goal_position.yaw

        self.roll._speed_limit = new_state.speed_limit.roll
        self.pitch._speed_limit = new_state.speed_limit.pitch
        self.yaw._speed_limit = new_state.speed_limit.yaw

        self.roll._torque_limit = new_state.torque_limit.roll
        self.pitch._torque_limit = new_state.torque_limit.pitch
        self.yaw._torque_limit = new_state.present_position.yaw

        self.compliant = new_state.compliant.value
