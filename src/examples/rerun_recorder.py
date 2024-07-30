import logging
import time
from typing import List

import numpy as np
import rerun as rr
from rerun_loader_urdf import URDFLogger
from urdf_parser_py import urdf

from reachy2_sdk import ReachySDK
from reachy2_sdk.media.camera import CameraView


def display_teleop_cam() -> None:
    if reachy.cameras.teleop is None:
        exit("There is no teleop camera.")

    print(f"Left camara parameters {reachy.cameras.teleop.get_parameters(CameraView.LEFT)}")
    # print(reachy.cameras.teleop.get_parameters(CameraView.RIGHT))

    try:
        while True:
            frame, ts = reachy.cameras.teleop.get_compressed_frame(CameraView.LEFT)
            frame_r, ts_r = reachy.cameras.teleop.get_compressed_frame(CameraView.RIGHT)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB)
            # print(f"timestamps secs: left {ts} - right {ts_r}")

            rr.log("/teleop_camera/left_image/compressed", rr.ImageEncoded(contents=frame, format=rr.ImageFormat.JPEG))
            rr.log("/teleop_camera/right_image/compressed", rr.ImageEncoded(contents=frame_r, format=rr.ImageFormat.JPEG))

            # cv2.imshow("left", frame)
            # cv2.imshow("right", frame_r)
            # cv2.waitKey(1)
            time.sleep(0.2)

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


def _get_joints(joint_name: str, urdf: urdf) -> urdf.Joint:
    for j in urdf.joints:
        if j.name == joint_name:
            return j
    raise RuntimeError("Invalid joint name")


def _log_arm_joints_poses(arm_pos: List[float], urdf_logger: URDFLogger, left: bool) -> None:
    side = "r"
    if left:
        side = "l"
    joint = _get_joints(f"{side}_shoulder_pitch", urdf_logger.urdf)
    joint.origin.rotation = [0.0, arm_pos[0], 0.0]
    urdf_logger.log_joint(urdf_logger.joint_entity_path(joint), joint=joint)

    joint = _get_joints(f"{side}_shoulder_roll", urdf_logger.urdf)
    joint.origin.rotation = [arm_pos[1], 0.0, 0.0]
    urdf_logger.log_joint(urdf_logger.joint_entity_path(joint), joint=joint)

    joint = _get_joints(f"{side}_elbow_pitch", urdf_logger.urdf)
    joint.origin.rotation = [0.0, arm_pos[3], 0.0]
    urdf_logger.log_joint(urdf_logger.joint_entity_path(joint), joint=joint)

    joint = _get_joints(f"{side}_elbow_yaw", urdf_logger.urdf)
    joint.origin.rotation = [0.0, 0.0, arm_pos[2]]
    urdf_logger.log_joint(urdf_logger.joint_entity_path(joint), joint=joint)

    joint = _get_joints(f"{side}_wrist_roll", urdf_logger.urdf)
    joint.origin.rotation = [arm_pos[4], 0.0, 0.0]
    urdf_logger.log_joint(urdf_logger.joint_entity_path(joint), joint=joint)

    joint = _get_joints(f"{side}_wrist_pitch", urdf_logger.urdf)
    joint.origin.rotation = [0.0, arm_pos[5], 0.0]
    urdf_logger.log_joint(urdf_logger.joint_entity_path(joint), joint=joint)

    joint = _get_joints(f"{side}_wrist_yaw", urdf_logger.urdf)
    joint.origin.rotation = [0.0, 0.0, arm_pos[6]]
    urdf_logger.log_joint(urdf_logger.joint_entity_path(joint), joint=joint)


def _log_head_poses(rpy_head: List[float], urdf_logger: URDFLogger) -> None:
    joint = _get_joints("neck_roll", urdf_logger.urdf)
    joint.origin.rotation = [rpy_head[0], 0.0, 0.0]
    urdf_logger.log_joint(urdf_logger.joint_entity_path(joint), joint=joint)

    joint = _get_joints("neck_pitch", urdf_logger.urdf)
    joint.origin.rotation = [0.0, rpy_head[1], 0.0]
    urdf_logger.log_joint(urdf_logger.joint_entity_path(joint), joint=joint)

    joint = _get_joints("neck_yaw", urdf_logger.urdf)
    joint.origin.rotation = [0.0, 0.0, rpy_head[2]]
    urdf_logger.log_joint(urdf_logger.joint_entity_path(joint), joint=joint)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    reachy = ReachySDK(host="localhost")

    rr.init("recorder_example", spawn=True)
    rr.connect()

    if not reachy.is_connected:
        exit("Reachy is not connected.")

    # if reachy.cameras is None:
    #    exit("There is no connected camera.")
    torso_entity = "world/world_joint/base_link/back_bar_joint/back_bar/torso_base/torso"
    urdf_path = "/home/fabien/Dev/Python/reachy2_rerun_test/reachy_v3_fix2.urdf"

    urdf_logger = URDFLogger(urdf_path, torso_entity)

    urdf_logger.log()

    try:
        while True:
            rpy_head = np.deg2rad(reachy.head.get_joints_positions())
            _log_head_poses(rpy_head, urdf_logger)

            # todo get radian directly
            l_arm_pos = np.deg2rad(reachy.l_arm.get_joints_positions())
            _log_arm_joints_poses(l_arm_pos, urdf_logger, True)

            r_arm_pos = np.deg2rad(reachy.r_arm.get_joints_positions())
            _log_arm_joints_poses(r_arm_pos, urdf_logger, False)

            time.sleep(0.2)

    except KeyboardInterrupt:
        logging.info("User Interrupt")
