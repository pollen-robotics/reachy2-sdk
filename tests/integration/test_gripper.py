import time

from reachy2_sdk import ReachySDK


def main_test() -> None:
    print("Trying to connect on localhost Reachy...")
    time.sleep(1.0)
    reachy = ReachySDK(host="localhost")
    try:
        time.sleep(1.0)
        if not reachy.is_connected():
            print("Failed to connect to Reachy, exiting...")
            return

        print("connected")
        reachy.turn_on()
        time.sleep(0.5)
        while True:
            reachy.r_arm.gripper.close()
            reachy.l_arm.gripper.close()
            print(reachy.l_arm.gripper.opening)
            time.sleep(1.0)
            reachy.r_arm.gripper.open()
            reachy.l_arm.gripper.open()
            print(reachy.l_arm.gripper.opening)
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
