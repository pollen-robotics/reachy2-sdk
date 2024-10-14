"""Example script to display live frames from the teleoperation and depth cameras."""

import argparse
import logging

import cv2

from reachy2_sdk import ReachySDK
from reachy2_sdk.media.camera import CameraView


def display_teleop_cam() -> None:
    """Display live frames from the teleoperation camera.

    This function retrieves and displays frames from the left and right
    views of the teleoperation camera. The function terminates
    upon a keyboard interrupt.

    Raises:
        SystemExit: If the teleop camera is not available.
    """
    if reachy.cameras.teleop is None:
        exit("There is no teleop camera.")

    print(f"Left camera parameters {reachy.cameras.teleop.get_parameters(CameraView.LEFT)}")
    print(f"Left camera extrinsic parameters {reachy.cameras.teleop.get_extrinsics(CameraView.LEFT)}")
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


def display_depth_cam() -> None:
    """Display live frames from the depth camera.

    This function retrieves and displays RGB and depth frames from the depth camera.
    It normalizes the depth map for visualization and shows the RGB frame and normalized depth
    frame side by side. The function exits upon a keyboard interrupt.

    Raises:
        SystemExit: If the depth camera is not available.
    """
    if reachy.cameras.depth is None:
        exit("There is no depth camera.")

    print(f"Depth camera parameters {reachy.cameras.depth.get_parameters()}")
    print(f"Depth camera extrinsic parameters {reachy.cameras.depth.get_extrinsics()}")

    try:
        while True:
            rgb, ts = reachy.cameras.depth.get_frame()
            depth, ts_r = reachy.cameras.depth.get_depth_frame()
            depth_map_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # type: ignore [attr-defined]
            cv2.imshow("frame", rgb)
            cv2.imshow("depthn", depth_map_normalized)
            cv2.waitKey(1)

    except KeyboardInterrupt:
        logging.info("User Interrupt")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    argParser = argparse.ArgumentParser(description="SDK camera example")
    argParser.add_argument(
        "mode",
        type=str,
        choices=["teleop", "depth"],
    )
    args = argParser.parse_args()

    reachy = ReachySDK(host="localhost")

    if not reachy.is_connected:
        exit("Reachy is not connected.")

    if reachy.cameras is None:
        exit("There is no connected camera.")

    if args.mode == "teleop":
        display_teleop_cam()
    elif args.mode == "depth":
        display_depth_cam()
