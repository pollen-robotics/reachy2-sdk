"""Reachy Arm module.

Handles all specific method to an Arm (left and/or right) especially:
- the forward kinematics
- the inverse kinematics
"""

from typing import List, Any

from google.protobuf.empty_pb2 import Empty

import grpc

from reachy_sdk_api import orbita3d_pb2, orbita3d_pb2_grpc

from reachy_sdk_api_v2.component_pb2 import ComponentId


class Orbita3D:
    def __init__(self, orbita: orbita3d_pb2.Orbita3D, stub: orbita3d_pb2_grpc.Orbita3DServiceStub) -> None:
        """Set up the arm with its kinematics."""
        self.id = ComponentId(id=orbita.id)


class Orbita3DSDK:
    """Arm abstract class used for both left/right arms.

    It exposes the kinematics of the arm:
    - you can access the joints actually used in the kinematic chain,
    - you can compute the forward and inverse kinematics
    """

    def __init__(self, host: str, orbita3d_port: int = 50071) -> None:
        """Set up the connection with the mobile base."""
        self._host = host
        self._orbita3d_port = orbita3d_port
        self._grpc_channel = grpc.insecure_channel(f"{self._host}:{self._orbita3d_port}")

        self._stub = orbita3d_pb2_grpc.Orbita3DServiceStub(self._grpc_channel)

        self._orbita3d_list: List[Orbita3D] = []
        self._get_all_orbita3d()

    def _get_all_orbita3d(self) -> None:
        orbitas = self._stub.GetAllOrbita3D(Empty())
        for orbita in orbitas.info:
            orbita3d = Orbita3D(orbita, self._stub)
            self._orbita3d_list.append(orbita3d)
            self._orbita3d_id_to_component = dict(zip([orbita3d.id for orbita3d in self._orbita3d_list], self._orbita3d_list))

    def get_list(self) -> List[Orbita3D]:
        return self._orbita3d_list

    def __getitem__(self, id: str) -> Any:
        return self._orbita3d_id_to_component[id]
