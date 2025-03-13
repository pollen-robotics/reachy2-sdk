import time

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

from reachy2_sdk.grippers.gripper_joint import GripperJoint


@pytest.mark.offline
def test_class() -> None:
    goal_position_rad = 3
    present_position_rad = 4
    compliant = True

    hand_proto = Hand_proto()
    hand_state = HandState(
        opening=FloatValue(value=0.2),
        force=FloatValue(value=2),
        holding_object=BoolValue(value=True),
        goal_position=HandPosition(parallel_gripper=ParallelGripperPosition(position=FloatValue(value=goal_position_rad))),
        present_position=HandPosition(
            parallel_gripper=ParallelGripperPosition(position=FloatValue(value=present_position_rad))
        ),
        joints_limits=JointsLimits(parallel_gripper=ParallelGripperLimits(limits=JointLimits(max=5, min=6))),
        temperatures=HandTemperatures(parallel_gripper=Temperatures(driver=7, motor=8)),
        compliant=BoolValue(value=compliant),
    )

    joint = GripperJoint(initial_state=hand_state)

    assert joint.__repr__() != ""

    assert joint.opening == 20

    assert joint._goal_position == goal_position_rad
    assert joint._present_position == present_position_rad
    assert joint.goal_position == np.rad2deg(goal_position_rad)
    assert joint.present_position == np.rad2deg(present_position_rad)
    assert joint.is_on() is False
    assert joint.is_off() is True

    goal_position_rad = 5
    present_position_rad = 6

    hand_state = HandState(
        opening=FloatValue(value=0.7),
        force=FloatValue(value=3),
        holding_object=BoolValue(value=False),
        goal_position=HandPosition(parallel_gripper=ParallelGripperPosition(position=FloatValue(value=goal_position_rad))),
        present_position=HandPosition(
            parallel_gripper=ParallelGripperPosition(position=FloatValue(value=present_position_rad))
        ),
        joints_limits=JointsLimits(parallel_gripper=ParallelGripperLimits(limits=JointLimits(max=5, min=6))),
        temperatures=HandTemperatures(parallel_gripper=Temperatures(driver=7, motor=8)),
    )
    joint._update_with(hand_state)

    assert joint.opening == 70

    joint._compliant = True

    assert joint._goal_position == goal_position_rad
    assert joint._present_position == present_position_rad
    assert joint.goal_position == np.rad2deg(goal_position_rad)
    assert joint.present_position == np.rad2deg(present_position_rad)

    joint._is_moving = True

    for _ in range(10):
        joint._check_joint_movement()

    assert joint._is_moving == False
