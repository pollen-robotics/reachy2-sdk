from grpc import Channel

from .register import Register

from typing import Dict

from google.protobuf.wrappers_pb2 import BoolValue

from reachy_sdk_api_v2.orbita3d_pb2_grpc import Orbita3DServiceStub

from reachy_sdk_api_v2.orbita3d_pb2 import Orbita3DState, Float3D, Vector3D
from reachy_sdk_api_v2.kinematics_pb2 import Rotation3D

from .orbita_utils import OrbitaJoint, OrbitaMotor, OrbitaAxis


class Orbita3d:
    compliant = Register(readonly=False, type=BoolValue, label="compliant")

    def __init__(self, uid: int, name: str, initial_state: Orbita3DState, grpc_channel: Channel):  # noqa: C901
        self.name = name
        self.id = uid
        self._stub = Orbita3DServiceStub(grpc_channel)

        self._state: Dict[str, bool] = {}
        init_state: Dict[str, Dict[str, float]] = {}

        for field, value in initial_state.ListFields():
            if field.name == "compliant":
                self._state[field.name] = value.value
            else:
                if isinstance(value, Rotation3D):
                    for _, rpy in value.ListFields():
                        for axis, val in rpy.ListFields():
                            print(f"{axis.name}: {val}")
                            if axis.name not in init_state:
                                init_state[axis.name] = {}
                            init_state[axis.name][field.name] = val
                if isinstance(value, Float3D):
                    for motor, val in value.ListFields():
                        if motor.name not in init_state:
                            init_state[motor.name] = {}
                        init_state[motor.name][field.name] = val
                if isinstance(value, Vector3D):
                    for axis, val in value.ListFields():
                        if axis.name not in init_state:
                            init_state[axis.name] = {}
                        init_state[axis.name][field.name] = val

        self.roll = OrbitaJoint(initial_state=init_state["roll"], axis_type="roll")
        self.pitch = OrbitaJoint(initial_state=init_state["pitch"], axis_type="pitch")
        self.yaw = OrbitaJoint(initial_state=init_state["yaw"], axis_type="yaw")

        self._motor1 = OrbitaMotor(initial_state=init_state["motor_1"])
        self._motor2 = OrbitaMotor(initial_state=init_state["motor_2"])
        self._motor3 = OrbitaMotor(initial_state=init_state["motor_3"])

        self._axis_x = OrbitaAxis(initial_state=init_state["x"])
        self._axis_y = OrbitaAxis(initial_state=init_state["y"])
        self._axis_z = OrbitaAxis(initial_state=init_state["z"])

    @property
    def temperatures(self) -> Dict[str, float]:
        return {"motor_1": self._motor1.temperature, "motor_2": self._motor2.temperature, "motor_3": self._motor3.temperature}

    def _update_with(self, new_state: Orbita3DState) -> None:
        """Update the orbita with a newly received (partial) state received from the gRPC server."""
        for field, value in new_state.ListFields():
            if field.name == "compliant":
                self._state[field.name] = value
            else:
                if isinstance(value, Float3D):
                    for axis, val in value.ListFields():
                        joint = getattr(self, axis.name)
                        joint._state[field.name] = val
