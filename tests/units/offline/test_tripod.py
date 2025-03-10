import grpc
import pytest
from google.protobuf.wrappers_pb2 import FloatValue
from reachy2_sdk_api.part_pb2 import PartId
from reachy2_sdk_api.tripod_pb2 import Tripod as Tripod_proto
from reachy2_sdk_api.tripod_pb2 import (
    TripodAxis,
    TripodDescription,
    TripodJoint,
    TripodJointState,
    TripodState,
)

from reachy2_sdk.parts.tripod import Tripod


@pytest.mark.offline
def test_class() -> None:
    grpc_channel = grpc.insecure_channel("dummy:5050")

    tripod_id = PartId(name="tripod", id=100)

    tripod_proto = Tripod_proto(
        part_id=tripod_id,
        description=TripodDescription(height_joint=TripodJoint(axis=TripodAxis.HEIGHT)),
    )

    tripod_state = TripodState(
        part_id=tripod_id,
        height=TripodJointState(
            joint=TripodJoint(axis=TripodAxis.HEIGHT),
            present_position=FloatValue(value=1.0),
            goal_position=FloatValue(value=2.0),
        ),
    )

    tripod = Tripod(proto_msg=tripod_proto, initial_state=tripod_state, grpc_channel=grpc_channel)

    assert str(tripod) == "<Tripod height=1.0 >"
    assert tripod.height == 1.0

    new_tripod_state = TripodState(
        part_id=tripod_id,
        height=TripodJointState(
            joint=TripodJoint(axis=TripodAxis.HEIGHT),
            present_position=FloatValue(value=3.0),
            goal_position=FloatValue(value=1.0),
        ),
    )

    tripod._update_with(new_tripod_state)

    assert tripod.height == 3.0
    assert str(tripod) == "<Tripod height=3.0 >"

    with pytest.raises(TypeError):
        tripod.set_height("l")
