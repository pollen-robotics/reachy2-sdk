import logging

from reachy2_sdk import ReachySDK

if __name__ == "__main__":
    print("Reachy SDK example: set to zero pose")

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

    print("Set to zero pose...")
    reachy.set_pose("zero")

    exit("Exiting example")
