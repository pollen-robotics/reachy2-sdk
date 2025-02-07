import threading
import time

import numpy as np
import pytest

from reachy2_sdk.reachy_sdk import ReachySDK


@pytest.mark.mobile_base
def test_modes(reachy_sdk_zeroed: ReachySDK) -> None:
    if reachy_sdk_zeroed.mobile_base is not None:
        reachy_sdk_zeroed.turn_on()
        time.sleep(0.2)
        assert reachy_sdk_zeroed.mobile_base.is_on()

        reachy_sdk_zeroed.mobile_base.turn_off()
        time.sleep(0.2)
        assert reachy_sdk_zeroed.mobile_base._drive_mode == "free_wheel"
        assert reachy_sdk_zeroed.mobile_base.is_off()

        reachy_sdk_zeroed.mobile_base._set_drive_mode("brake")
        time.sleep(0.2)
        assert reachy_sdk_zeroed.mobile_base._drive_mode == "brake"
        assert reachy_sdk_zeroed.mobile_base.is_on()

        reachy_sdk_zeroed.mobile_base._set_drive_mode("cmd_goto")
        time.sleep(0.2)
        assert reachy_sdk_zeroed.mobile_base._drive_mode == "cmd_goto"
        assert reachy_sdk_zeroed.mobile_base.is_on()

        reachy_sdk_zeroed.mobile_base._set_control_mode("pid")
        time.sleep(0.2)
        assert reachy_sdk_zeroed.mobile_base._control_mode == "pid"


@pytest.mark.mobile_base
def test_lidar_safety_distances(reachy_sdk_zeroed: ReachySDK) -> None:
    if reachy_sdk_zeroed.mobile_base is not None:
        reachy_sdk_zeroed.mobile_base.lidar.safety_slowdown_distance = 5.0
        time.sleep(0.2)
        assert np.isclose(reachy_sdk_zeroed.mobile_base.lidar.safety_slowdown_distance, 5.0, atol=1e-03)

        reachy_sdk_zeroed.mobile_base.lidar.safety_critical_distance = 1.0
        time.sleep(0.2)
        assert np.isclose(reachy_sdk_zeroed.mobile_base.lidar.safety_slowdown_distance, 5.0, atol=1e-03)
        assert np.isclose(reachy_sdk_zeroed.mobile_base.lidar.safety_critical_distance, 1.0, atol=1e-03)

        reachy_sdk_zeroed.mobile_base.lidar.reset_safety_default_distances()

        time.sleep(0.5)
        assert np.isclose(reachy_sdk_zeroed.mobile_base.lidar.safety_slowdown_distance, 0.7, atol=1e-03)
        assert np.isclose(reachy_sdk_zeroed.mobile_base.lidar.safety_critical_distance, 0.55, atol=1e-03)


@pytest.mark.mobile_base
def test_reset_odometry(reachy_sdk_zeroed: ReachySDK) -> None:
    if reachy_sdk_zeroed.mobile_base is not None:
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["x"], 0.0, atol=1e-03)
        assert np.isclose(odom["y"], 0.0, atol=1e-03)
        assert np.isclose(odom["theta"], 0.0, atol=1e-01)

        reachy_sdk_zeroed.mobile_base.set_goal_speed(vx=0.3, vy=0.2, vtheta=20)
        tic = time.time()
        while time.time() - tic < 2:
            reachy_sdk_zeroed.mobile_base.send_speed_command()
            time.sleep(0.01)
        time.sleep(0.3)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert not np.isclose(odom["x"], 0.0, atol=0.3)
        assert not np.isclose(odom["y"], 0.0, atol=0.2)
        assert not np.isclose(odom["theta"], 0.0, atol=20)

        reachy_sdk_zeroed.mobile_base.reset_odometry()
        time.sleep(0.2)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["x"], 0.0, atol=1e-03)
        assert np.isclose(odom["y"], 0.0, atol=1e-03)
        assert np.isclose(odom["theta"], 0.0, atol=1e-01)

        reachy_sdk_zeroed.mobile_base.goto(x=0.5, y=0.5, theta=50, wait=True)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert not np.isclose(odom["x"], 0.0, atol=0.4)
        assert not np.isclose(odom["y"], 0.0, atol=0.4)
        assert not np.isclose(odom["theta"], 0.0, atol=45)

        reachy_sdk_zeroed.mobile_base.reset_odometry()
        time.sleep(0.2)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["x"], 0.0, atol=1e-03)
        assert np.isclose(odom["y"], 0.0, atol=1e-03)
        assert np.isclose(odom["theta"], 0.0, atol=1e-01)


@pytest.mark.mobile_base
def test_odometry_pos(reachy_sdk_zeroed: ReachySDK) -> None:
    if reachy_sdk_zeroed.mobile_base is not None:
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["x"], 0.0, atol=1e-03)
        assert np.isclose(odom["y"], 0.0, atol=1e-03)
        assert np.isclose(odom["theta"], 0.0, atol=1e-01)

        reachy_sdk_zeroed.mobile_base.set_goal_speed(vx=0.5, vy=0.0, vtheta=0)
        tic = time.time()
        while time.time() - tic < 2:
            reachy_sdk_zeroed.mobile_base.send_speed_command()
            time.sleep(0.01)
        time.sleep(0.3)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert odom["x"] > 0.4
        assert np.isclose(odom["y"], 0.0, atol=1e-02)
        assert np.isclose(odom["theta"], 0.0, atol=1e-01)

        reachy_sdk_zeroed.mobile_base.goto(x=0.0, y=0.0, theta=0, wait=True)
        time.sleep(0.3)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["x"], 0.0, atol=1e-02)
        assert np.isclose(odom["y"], 0.0, atol=1e-02)
        assert np.isclose(odom["theta"], 0.0, atol=1e-01)

        reachy_sdk_zeroed.mobile_base.set_goal_speed(vx=0.0, vy=-0.5, vtheta=0)
        tic = time.time()
        while time.time() - tic < 2:
            reachy_sdk_zeroed.mobile_base.send_speed_command()
            time.sleep(0.01)
        time.sleep(0.3)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["x"], 0.0, atol=1e-02)
        assert odom["y"] < -0.4
        assert np.isclose(odom["theta"], 0.0, atol=1e-01)

        reachy_sdk_zeroed.mobile_base.reset_odometry()
        time.sleep(0.1)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["x"], 0.0, atol=1e-03)
        assert np.isclose(odom["y"], 0.0, atol=1e-03)
        assert np.isclose(odom["theta"], 0.0, atol=1e-01)

        reachy_sdk_zeroed.mobile_base.set_goal_speed(vx=0.0, vy=0.0, vtheta=50)
        tic = time.time()
        while time.time() - tic < 2:
            reachy_sdk_zeroed.mobile_base.send_speed_command()
            time.sleep(0.01)
        time.sleep(0.3)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["x"], 0.0, atol=1e-02)
        assert np.isclose(odom["y"], 0.0, atol=1e-02)
        assert odom["theta"] > 50


@pytest.mark.mobile_base
def test_odometry_vel(reachy_sdk_zeroed: ReachySDK) -> None:
    if reachy_sdk_zeroed.mobile_base is not None:
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert odom["vx"] == 0.0
        assert odom["vy"] == 0.0
        assert odom["vtheta"] == 0.0

        reachy_sdk_zeroed.mobile_base.set_goal_speed(vx=0.5, vy=0.0, vtheta=0)
        tic = time.time()
        while time.time() - tic < 2:
            reachy_sdk_zeroed.mobile_base.send_speed_command()
            time.sleep(0.01)
        time.sleep(0.3)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert odom["vx"] == 0.0
        assert odom["vy"] == 0.0
        assert odom["vtheta"] == 0.0

        reachy_sdk_zeroed.mobile_base._set_max_xy_goto(2.0)
        reachy_sdk_zeroed.mobile_base.goto(x=0.0, y=0.0, theta=0)
        time.sleep(0.3)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert odom["vx"] < 0.0
        assert np.isclose(odom["vy"], 0.0, atol=1e-02)
        assert np.isclose(odom["vtheta"], 0.0, atol=1e-01)

        reachy_sdk_zeroed.mobile_base.goto(x=0.0, y=0.0, theta=0)
        time.sleep(0.3)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert odom["vx"] < 0.0
        assert np.isclose(odom["vy"], 0.0, atol=1e-02)
        assert np.isclose(odom["vtheta"], 0.0, atol=1e-01)

        def send_speed_forward():
            reachy_sdk_zeroed.mobile_base.set_goal_speed(vx=0.5, vy=0.0, vtheta=0)
            tic = time.time()
            while time.time() - tic < 5:
                reachy_sdk_zeroed.mobile_base.send_speed_command()
                time.sleep(0.01)

        forward_thread = threading.Thread(target=send_speed_forward)
        forward_thread.start()
        time.sleep(1)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["vx"], 0.5, atol=1e-03)
        assert odom["vy"] == 0.0
        assert odom["vtheta"] == 0.0
        time.sleep(2)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["vx"], 0.5, atol=1e-03)
        assert odom["vy"] == 0.0
        assert odom["vtheta"] == 0.0
        time.sleep(3)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert odom["vx"] == 0.0
        assert odom["vy"] == 0.0
        assert odom["vtheta"] == 0.0

        def send_y_theta_speeds():
            reachy_sdk_zeroed.mobile_base.set_goal_speed(vx=0.0, vy=0.4, vtheta=20)
            tic = time.time()
            while time.time() - tic < 5:
                reachy_sdk_zeroed.mobile_base.send_speed_command()
                time.sleep(0.01)

        forward_thread = threading.Thread(target=send_y_theta_speeds)
        forward_thread.start()
        time.sleep(1)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert odom["vx"] == 0.0
        assert np.isclose(odom["vy"], 0.4, atol=1e-03)
        assert np.isclose(odom["vtheta"], 20, atol=1e-03)
        time.sleep(2)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert odom["vx"] == 0.0
        assert np.isclose(odom["vy"], 0.4, atol=1e-03)
        assert np.isclose(odom["vtheta"], 20, atol=1e-03)
        time.sleep(3)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert odom["vx"] == 0.0
        assert odom["vy"] == 0.0
        assert odom["vtheta"] == 0.0


@pytest.mark.mobile_base
def test_mobile_base_goto(reachy_sdk_zeroed: ReachySDK) -> None:
    if reachy_sdk_zeroed.mobile_base is not None:
        assert reachy_sdk_zeroed.mobile_base is not None
        reachy_sdk_zeroed.mobile_base.goto(x=0.5, y=0.5, theta=0.5)
