import logging
from typing import List

import cv2
from reachy2_sdk_api.video_pb2 import CameraInfo, View

from reachy2_sdk import ReachySDK
from reachy2_sdk.video import Camera

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    reachy = ReachySDK(host="localhost")

    if reachy.grpc_status == "disconnected":
        exit("Reachy is not connected.")

    list_cam: List[CameraInfo] = reachy.video.get_all_cameras()

    # select camera here
    cam_type = Camera.TELEOP
    # cam_type = Camera.SR

    cam = reachy.video.get_camera_info(list_cam, cam_type)

    if cam is None:
        exit("Camera was not found")

    reachy.video.init_camera(cam)

    try:
        while reachy.video.capture(cam):
            if cam_type == Camera.TELEOP:
                frame = reachy.video.get_frame(cam, View.LEFT)
                frame_r = reachy.video.get_frame(cam, View.RIGHT)
                cv2.imshow("left", frame)
                cv2.imshow("right", frame_r)
            elif cam_type == Camera.SR:
                cv2.imshow("sr_depthNode_left", reachy.video.get_depth_frame(cam, View.LEFT))
                cv2.imshow("sr_depthNode_right", reachy.video.get_depth_frame(cam, View.RIGHT))
                cv2.imshow("depth", reachy.video.get_depthmap(cam))
                cv2.imshow("disparity", reachy.video.get_disparity(cam))
            cv2.waitKey(1)

    except KeyboardInterrupt:
        logging.info("User Interrupt")
    finally:
        reachy.video.close_camera(cam)
        reachy.disconnect()
