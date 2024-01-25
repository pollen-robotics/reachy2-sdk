import time
from math import e

import numpy as np
import numpy.typing as npt
from reachy2_sdk_api.goto_pb2 import GoalStatus

from reachy2_sdk import ReachySDK


def main_test() -> None:
    print("Trying to connect on localhost Reachy...")
    time.sleep(1.0)
    reachy = ReachySDK(host="localhost")
    try:
        time.sleep(1.0)
        if reachy.grpc_status == "disconnected":
            print("Failed to connect to Reachy, exiting...")
            return

        print("connected")
        reachy.turn_on()
        while True:
            print("plop")
            print(reachy)
            reachy.r_arm.gripper.close()
            reachy.l_arm.gripper.close()
            print(reachy.l_arm.gripper.opening)
            time.sleep(1.0)
            reachy.r_arm.gripper.open()
            reachy.l_arm.gripper.open()
            time.sleep(1.0)

    except Exception as e:
        print(f"Exception: {e}")
    finally:
        reachy.turn_off()
        reachy.disconnect()


if __name__ == "__main__":
    # head_test()
    main_test()
    # deco_test()
    # multi_test()
