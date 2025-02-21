"""Example of setting Reachy to zero pose using Reachy SDK."""

import logging
import time

from reachy2_sdk import ReachySDK

if __name__ == "__main__":
    print("Reachy SDK example: set to default posture")

    # display messages from SDK
    logging.basicConfig(level=logging.INFO)

    # connect to Reachy
    reachy = ReachySDK(host="localhost")

    # check if connection is successful
    if not reachy.is_connected:
        exit("Reachy is not connected.")

    print("Reachy basic information:")
    print(reachy.info)
    print("Reachy joint status:")
    print(reachy.r_arm.joints)

    print("Turning on Reachy...")
    reachy.turn_on()

    time.sleep(0.2)

    print("Set to default posture...")
    reachy.goto_posture("default")

    time.sleep(1)

    exit("Exiting example")
