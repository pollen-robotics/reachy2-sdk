import time

import numpy as np
import numpy.typing as npt
from reachy2_sdk import ReachySDK
from reachy2_sdk_api.goto_pb2 import GoalStatus, GoToId


def connect_to_reachy():
    print("Trying to connect on localhost Reachy...")
    time.sleep(1.0)
    reachy = ReachySDK(host="localhost")
    time.sleep(1.0)
    if reachy._grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
        return
    return reachy


def turn_on():
    reachy = connect_to_reachy()

    print("Turning on...")

    reachy.turn_on()

    reachy.disconnect()
    ReachySDK.clear()


def turn_on_debug():
    reachy = connect_to_reachy()

    print("Turning on...")

    reachy.turn_on()

    print("The arm should move now")

    reachy.r_arm.elbow.pitch.goal_position = -90
    time.sleep(1.0)
    reachy.r_arm.elbow.pitch.goal_position = 0

    print("The arm should NOT move now !")

    reachy.turn_off()
    time.sleep(0.1)
    reachy.r_arm.elbow.pitch.goal_position = -90
    time.sleep(0.1)
    reachy.turn_on()

    time.sleep(1.0)
    reachy.turn_off()
    time.sleep(0.1)
    reachy.turn_on()
    time.sleep(0.1)

    reachy.disconnect()
    ReachySDK.clear()


if __name__ == "__main__":
    # turn_on()
    turn_on_debug()
