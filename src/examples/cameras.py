import logging

import cv2

from reachy2_sdk import ReachySDK
from reachy2_sdk.media.camera import CameraView


def display_teleop_cam() -> None:
    if reachy.cameras.teleop is None:
        exit("There is no teleop camera.")

    print(f"Left camara parameters {reachy.cameras.teleop.get_parameters(CameraView.LEFT)}")
    # print(reachy.cameras.teleop.get_parameters(CameraView.RIGHT))

    try:
        while True:
            frame, ts = reachy.cameras.teleop.get_frame(CameraView.LEFT)
            frame_r, ts_r = reachy.cameras.teleop.get_frame(CameraView.RIGHT)
            print(f"timestamps secs: left {ts} - right {ts_r}")
            cv2.imshow("left", frame)
            cv2.imshow("right", frame_r)
            cv2.waitKey(1)

    except KeyboardInterrupt:
        logging.info("User Interrupt")


"""
def display_SR_cam() -> None:
    if reachy.cameras.SR is None:
        exit("There is no SR camera.")

    try:
        while reachy.cameras.SR.capture():
            cv2.imshow("sr_depthNode_left", reachy.cameras.SR.get_depth_frame(CameraView.LEFT))
            cv2.imshow("sr_depthNode_right", reachy.cameras.SR.get_depth_frame(CameraView.RIGHT))
            cv2.imshow("depth", reachy.cameras.SR.get_depthmap())
            cv2.imshow("disparity", reachy.cameras.SR.get_disparity())
            cv2.waitKey(1)

    except KeyboardInterrupt:
        logging.info("User Interrupt")
"""

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    reachy = ReachySDK(host="localhost")

    if not reachy.is_connected:
        exit("Reachy is not connected.")

    if reachy.cameras is None:
        exit("There is no connected camera.")

    display_teleop_cam()
    # display_SR_cam()
