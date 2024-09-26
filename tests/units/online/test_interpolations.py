import time
from threading import Thread

import numpy as np
import pytest

from reachy2_sdk.reachy_sdk import ReachySDK

from .test_basic_movements import build_pose_matrix


class LoopThread:
    def __init__(self, reachy: ReachySDK, arm: str):
        self.reachy = reachy
        self.arm = arm
        self.running = False
        self.forward_list = []
        self.thread = Thread(
            target=self.loop_forward_function,
            args=(
                self.reachy,
                self.arm,
            ),
        )

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def loop_forward_function(self, reachy: ReachySDK, arm: str):
        while self.running:
            if arm == "right":
                self.forward_list.append(reachy.r_arm.forward_kinematics())
            else:
                self.forward_list.append(reachy.l_arm.forward_kinematics())
            time.sleep(0.5)


@pytest.mark.online
def test_send_cartesian_interpolation(reachy_sdk_zeroed: ReachySDK) -> None:
    A = build_pose_matrix(0.3, -0.2, -0.3)
    reachy_sdk_zeroed.r_arm.goto_from_matrix(A, wait=True)
    B = build_pose_matrix(0.3, -0.4, -0.3)
    t = LoopThread(reachy_sdk_zeroed, "r_arm")
    t.start()
    tic = time.time()
    reachy_sdk_zeroed.r_arm.send_cartesian_interpolation(B, duration=3.0)
    elapsed_time = time.time() - tic
    t.stop()
    assert np.isclose(elapsed_time, 3.0, 1)

    B_forward = reachy_sdk_zeroed.r_arm.forward_kinematics()
    print(t.forward_list)
    assert np.allclose(B_forward, B, atol=1e-03)

    for pose in t.forward_list:
        assert np.isclose(pose[0, 3], 0.3, 1e-03)
        assert np.isclose(pose[2, 3], -0.3, 1e-03)
        assert pose[1, 3] <= -0.2 and pose[1, 3] >= -0.4
