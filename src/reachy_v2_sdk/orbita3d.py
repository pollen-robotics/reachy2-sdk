from grpc import Channel

from .register import Register

from typing import Dict

from reachy_sdk_api_v2.orbita3d_pb2_grpc import Orbita3DServiceStub

from reachy_sdk_api_v2.orbita3d_pb2 import Orbita3DState, Float3D
from .orbita_utils import OrbitaJoint


class Orbita3d:
    compliant = Register(readonly=False, label="compliant")

    def __init__(self, name: str, initial_state: Orbita3DState, grpc_channel: Channel):
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

        self._state: Dict[str, bool] = {}

        self.roll = OrbitaJoint(initial_state=init_state.copy(), axis_type="roll")
        self.pitch = OrbitaJoint(initial_state=init_state.copy(), axis_type="pitch")
        self.yaw = OrbitaJoint(initial_state=init_state.copy(), axis_type="yaw")

        self.compliant = False

    def _update_with(self, new_state: Orbita3DState) -> None:
        """Update the orbita with a newly received (partial) state received from the gRPC server."""
        for field, value in new_state.ListFields():
            if field.name == "compliant":
                self._state[field.name] = value.value
            else:
                if isinstance(value, Float3D):
                    for axis, val in value.ListFields():
                        joint = getattr(self, axis.name)
                        joint._state[field.name] = val
