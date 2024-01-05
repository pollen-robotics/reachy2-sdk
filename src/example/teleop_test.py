import time

import numpy as np
from basic_tests import build_pose_matrix
from google.protobuf.wrappers_pb2 import FloatValue
from reachy2_sdk_api.arm_pb2 import ArmCartesianGoal
from reachy2_sdk_api.kinematics_pb2 import ExtEulerAngles, Matrix4x4, Point, Rotation3d

from reachy2_sdk import ReachySDK

reachy = ReachySDK(host="localhost")


goal = build_pose_matrix(1, 1, 0)

reachy.r_arm._arm_stub.TurnOn(reachy.r_arm.part_id)


def goto(x, y, z):
    goal = np.array(
        [
            [0, 0, 1, x],
            [0, 1, 0, y],
            [1, 0, 0, z],
            [0, 0, 0, 1],
        ]
    )
    goal_dict = {
        "id": {"id": 1, "name": "r_arm"},
        "goal_pose": goal,
        "duration": FloatValue(value=1.0),
    }
    # target = ArmCartesianGoal(
    #     id=reachy.r_arm.part_id,
    #     goal_pose=goal,
    #     duration=FloatValue(value=1.0),
    # )

    target = ArmCartesianGoal(
        id={"id": 1, "name": "r_arm"},
        goal_pose=Matrix4x4(data=goal.flatten().tolist()),
        duration=FloatValue(value=1.0),
    )
    reachy.r_arm._arm_stub.SendArmCartesianGoal(target)


goto(1, 1, 0)

print("coucou")
radius = 0.5  # Circle radius
fixed_x = 1  # Fixed x-coordinate
center_y, center_z = 0, 0  # Center of the circle in y-z plane
num_steps = 200  # Number of steps to complete the circle
frequency = 2  # Update frequency in Hz
step = 0  # Current step
# for step in range(num_steps):
while True:
    # Calculate angle for this step
    angle = 2 * np.pi * (step / num_steps)
    step += 1
    if step >= num_steps:
        step = 0
    # Calculate y and z coordinates
    y = center_y + radius * np.cos(angle)
    z = center_z + radius * np.sin(angle)

    # Call the goto function with the constant x and calculated y, z coordinates
    goto(fixed_x, y, z)
    time.sleep(0.01)
