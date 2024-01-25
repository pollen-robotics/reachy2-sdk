import grpc
import numpy as np
import pytest
from google.protobuf.wrappers_pb2 import BoolValue, FloatValue
from reachy2_sdk_api.hand_pb2 import Hand as Hand_proto
from reachy2_sdk_api.hand_pb2 import (
    HandPosition,
    HandState,
    HandTemperatures,
    JointLimits,
    JointsLimits,
    ParallelGripperLimits,
    ParallelGripperPosition,
    Temperatures,
)

from reachy2_sdk.hand import Hand


@pytest.mark.offline
def test_class() -> None:
    grpc_channel = grpc.insecure_channel("dummy:5050")

    goal_position_rad = 3
    present_position_rad = 4

    hand_proto = Hand_proto()
    hand_state = HandState(
        opening=FloatValue(value=0.2),
        force=FloatValue(value=2),
        holding_object=BoolValue(value=True),
        goal_position=HandPosition(parallel_gripper=ParallelGripperPosition(position=goal_position_rad)),
        present_position=HandPosition(parallel_gripper=ParallelGripperPosition(position=present_position_rad)),
        joints_limits=JointsLimits(parallel_gripper=ParallelGripperLimits(limits=JointLimits(max=5, min=6))),
        temperatures=HandTemperatures(parallel_gripper=Temperatures(driver=7, motor=8)),
    )

    hand = Hand(hand_msg=hand_proto, initial_state=hand_state, grpc_channel=grpc_channel)

    assert hand.opening == 20

    with pytest.raises(ValueError):
        hand.open(-1)

    with pytest.raises(ValueError):
        hand.open(101)

    with pytest.raises(ValueError):
        hand.close(-1)

    with pytest.raises(ValueError):
        hand.close(101)

    assert hand._goal_position == round(np.rad2deg(goal_position_rad), 1)
    assert hand._present_position == round(np.rad2deg(present_position_rad), 1)

    goal_position_rad = 5
    present_position_rad = 6

    hand_state = HandState(
        opening=FloatValue(value=0.7),
        force=FloatValue(value=3),
        holding_object=BoolValue(value=False),
        goal_position=HandPosition(parallel_gripper=ParallelGripperPosition(position=goal_position_rad)),
        present_position=HandPosition(parallel_gripper=ParallelGripperPosition(position=present_position_rad)),
        joints_limits=JointsLimits(parallel_gripper=ParallelGripperLimits(limits=JointLimits(max=5, min=6))),
        temperatures=HandTemperatures(parallel_gripper=Temperatures(driver=7, motor=8)),
    )
    hand._update_with(hand_state)

    assert hand.opening == 70

    # assert hand._goal_position == round(np.rad2deg(goal_position_rad), 1)
    # assert hand._present_position == round(np.rad2deg(present_position_rad), 1)
