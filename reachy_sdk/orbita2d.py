"""Reachy Arm module.

Handles all specific method to an Arm (left and/or right) especially:
- the forward kinematics
- the inverse kinematics
"""
import grpc

from google.protobuf.empty_pb2 import Empty

from typing import List, Any

from reachy_sdk_api_v2 import orbita2d_pb2, orbita2d_pb2_grpc

from reachy_sdk_api_v2.component_pb2 import ComponentId


class Orbita2D:
    def __init__(self, orbita: orbita2d_pb2.Orbita2DInfo, stub: orbita2d_pb2_grpc.Orbita2DServiceStub) -> None:
        """Set up the arm with its kinematics."""
        self.id = ComponentId(id=orbita.id.id)


class Orbita2DSDK:
    """Arm abstract class used for both left/right arms.

    It exposes the kinematics of the arm:
    - you can access the joints actually used in the kinematic chain,
    - you can compute the forward and inverse kinematics
    """

    def __init__(self, grpc_channel: grpc.Channel) -> None:
        """Set up the connection with the mobile base."""
        self._stub = orbita2d_pb2_grpc.Orbita2DServiceStub(grpc_channel)

        self._orbita2d_list: List[Orbita2D] = []
        self._get_all_orbita2d()

    def _get_all_orbita2d(self) -> None:
        orbitas = self._stub.GetAllOrbita2D(Empty())
        for orbita in orbitas.info:
            orbita2d = Orbita2D(orbita, self._stub)
            self._orbita2d_list.append(orbita2d)
            self._orbita2d_id_to_component = dict(
                zip([orbita2d.id.id for orbita2d in self._orbita2d_list], self._orbita2d_list)
            )

    def get_list(self) -> List[Orbita2D]:
        return self._orbita2d_list

    def __getitem__(self, id: str) -> Any:
        return self._orbita2d_id_to_component[id]
