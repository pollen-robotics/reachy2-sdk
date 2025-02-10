import threading
import time

import numpy as np
import pytest

from reachy2_sdk.reachy_sdk import ReachySDK

from .test_basic_movements import is_goto_finished


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
        time.sleep(0.5)
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

        tocancel1 = reachy_sdk_zeroed.mobile_base.goto(x=0.7, y=-0.5, theta=-30)
        tocancel2 = reachy_sdk_zeroed.mobile_base.goto(x=0.4, y=-0.2, theta=-10)
        assert not is_goto_finished(reachy_sdk_zeroed, tocancel1)
        assert not is_goto_finished(reachy_sdk_zeroed, tocancel2)
        assert not len(reachy_sdk_zeroed.mobile_base.get_goto_queue()) == 0
        reachy_sdk_zeroed.mobile_base.reset_odometry()
        time.sleep(0.1)
        assert is_goto_finished(reachy_sdk_zeroed, tocancel1)
        assert is_goto_finished(reachy_sdk_zeroed, tocancel2)
        assert len(reachy_sdk_zeroed.mobile_base.get_goto_queue()) == 0


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
        time.sleep(0.5)
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
        time.sleep(0.4)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert odom["vx"] == 0.0
        assert odom["vy"] == 0.0
        assert odom["vtheta"] == 0.0

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
        with pytest.raises(ValueError):
            reachy_sdk_zeroed.mobile_base.goto(x=1.5, y=0.2, theta=0)
        with pytest.raises(ValueError):
            reachy_sdk_zeroed.mobile_base.goto(x=0.0, y=1.2, theta=0)

        with pytest.raises(TypeError):
            reachy_sdk_zeroed.mobile_base.goto(x="mistake", y=0.2, theta=0)
        with pytest.raises(TypeError):
            reachy_sdk_zeroed.mobile_base.goto(x=0.2, y="mistake", theta=0)
        with pytest.raises(TypeError):
            reachy_sdk_zeroed.mobile_base.goto(x=0.2, y=0.2, theta="mistake")

        goto1 = reachy_sdk_zeroed.mobile_base.goto(x=0.5, y=0.5, theta=50)
        while not is_goto_finished(reachy_sdk_zeroed, goto1):
            time.sleep(0.1)
        time.sleep(0.3)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["x"], 0.5, atol=0.05)
        assert np.isclose(odom["y"], 0.5, atol=0.05)
        assert np.isclose(odom["theta"], 50, atol=5)

        goto2 = reachy_sdk_zeroed.mobile_base.goto(x=0.8, y=0.2, theta=-20, wait=True)
        assert is_goto_finished(reachy_sdk_zeroed, goto2)
        time.sleep(0.3)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["x"], 0.8, atol=0.05)
        assert np.isclose(odom["y"], 0.2, atol=0.05)
        assert np.isclose(odom["theta"], -20, atol=5)

        goto3 = reachy_sdk_zeroed.mobile_base.goto(x=0.0, y=-0.4, theta=30)
        time.sleep(0.1)
        assert not is_goto_finished(reachy_sdk_zeroed, goto3)
        time.sleep(0.2)
        reachy_sdk_zeroed.cancel_goto_by_id(goto3)
        time.sleep(0.1)
        assert is_goto_finished(reachy_sdk_zeroed, goto3)
        request3 = reachy_sdk_zeroed.get_goto_request(goto3)
        assert request3.part == "mobile_base"
        assert np.isclose(request3.request.goal_positions["x"], 0.0, atol=1e-03)
        assert np.isclose(request3.request.goal_positions["y"], -0.4, atol=1e-03)
        assert np.isclose(request3.request.goal_positions["theta"], 30, atol=1e-03)
        assert np.isclose(request3.request.distance_tolerance, 0.05, atol=1e-03)
        assert np.isclose(request3.request.angle_tolerance, 5, atol=1e-03)

        goto4 = reachy_sdk_zeroed.mobile_base.goto(
            x=-0.2, y=0.3, theta=70, distance_tolerance=0.02, angle_tolerance=2, wait=True
        )
        time.sleep(0.1)
        assert is_goto_finished(reachy_sdk_zeroed, goto4)
        request4 = reachy_sdk_zeroed.get_goto_request(goto4)
        assert request4.part == "mobile_base"
        assert np.isclose(request4.request.goal_positions["x"], -0.2, atol=1e-03)
        assert np.isclose(request4.request.goal_positions["y"], 0.3, atol=1e-03)
        assert np.isclose(request4.request.goal_positions["theta"], 70, atol=1e-03)
        assert np.isclose(request4.request.distance_tolerance, 0.02, atol=1e-03)
        assert np.isclose(request4.request.angle_tolerance, 2, atol=1e-03)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["x"], -0.2, atol=0.02)
        assert np.isclose(odom["y"], 0.3, atol=0.02)
        assert np.isclose(odom["theta"], 70, atol=2)

        goto5 = reachy_sdk_zeroed.mobile_base.goto(x=0, y=0.3, theta=np.deg2rad(60), distance_tolerance=0.02, degrees=False)
        request5 = reachy_sdk_zeroed.get_goto_request(goto5)
        assert request5.part == "mobile_base"
        assert np.isclose(request5.request.goal_positions["x"], 0, atol=1e-03)
        assert np.isclose(request5.request.goal_positions["y"], 0.3, atol=1e-03)
        assert np.isclose(request5.request.goal_positions["theta"], 60, atol=1e-03)
        assert np.isclose(request5.request.distance_tolerance, 0.02, atol=1e-03)
        assert np.isclose(request5.request.angle_tolerance, 5, atol=1e-03)

        reachy_sdk_zeroed.cancel_goto_by_id(goto5)
        time.sleep(0.1)
        assert is_goto_finished(reachy_sdk_zeroed, goto5)

        goto6 = reachy_sdk_zeroed.mobile_base.goto(
            x=0, y=0.3, theta=np.deg2rad(40), distance_tolerance=0.02, angle_tolerance=np.deg2rad(2), degrees=False
        )
        request6 = reachy_sdk_zeroed.get_goto_request(goto6)
        assert request6.part == "mobile_base"
        assert np.isclose(request6.request.goal_positions["x"], 0, atol=1e-03)
        assert np.isclose(request6.request.goal_positions["y"], 0.3, atol=1e-03)
        assert np.isclose(request6.request.goal_positions["theta"], 40, atol=1e-03)
        assert np.isclose(request6.request.distance_tolerance, 0.02, atol=1e-03)
        assert np.isclose(request6.request.angle_tolerance, 2, atol=1e-03)

        reachy_sdk_zeroed.cancel_goto_by_id(goto6)
        time.sleep(0.1)
        assert is_goto_finished(reachy_sdk_zeroed, goto6)


@pytest.mark.mobile_base
def test_mobile_base_goto_timeout(reachy_sdk_zeroed: ReachySDK) -> None:
    if reachy_sdk_zeroed.mobile_base is not None:
        goto1 = reachy_sdk_zeroed.mobile_base.goto(x=0.8, y=0.5, theta=80, timeout=1, wait=True)
        time.sleep(1.1)
        assert is_goto_finished(reachy_sdk_zeroed, goto1)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert not np.isclose(odom["x"], 0.8, atol=0.05)
        assert not np.isclose(odom["y"], 0.5, atol=0.05)
        assert not np.isclose(odom["theta"], 80, atol=5)

        request1 = reachy_sdk_zeroed.get_goto_request(goto1)
        assert request1.part == "mobile_base"
        assert np.isclose(request1.request.goal_positions["x"], 0.8, atol=1e-03)
        assert np.isclose(request1.request.goal_positions["y"], 0.5, atol=1e-03)
        assert np.isclose(request1.request.goal_positions["theta"], 80, atol=1e-03)
        assert np.isclose(request1.request.timeout, 1, atol=1e-03)

        tic = time.time()
        goto2 = reachy_sdk_zeroed.mobile_base.goto(x=0.0, y=-0.2, theta=0, timeout=0.8)
        while not is_goto_finished(reachy_sdk_zeroed, goto2):
            time.sleep(0.01)
        assert np.isclose(time.time() - tic, 0.8, atol=0.2)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert not np.isclose(odom["x"], 0.0, atol=0.05)
        assert not np.isclose(odom["y"], -0.2, atol=0.05)
        assert not np.isclose(odom["theta"], 0, atol=5)

        request2 = reachy_sdk_zeroed.get_goto_request(goto2)
        assert request1.part == "mobile_base"
        assert np.isclose(request2.request.goal_positions["x"], 0.0, atol=1e-03)
        assert np.isclose(request2.request.goal_positions["y"], -0.2, atol=1e-03)
        assert np.isclose(request2.request.goal_positions["theta"], 0, atol=1e-03)
        assert np.isclose(request2.request.timeout, 0.8, atol=1e-03)


@pytest.mark.mobile_base
def test_mobile_base_translate_by(reachy_sdk_zeroed: ReachySDK) -> None:
    if reachy_sdk_zeroed.mobile_base is not None:
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        trans0 = reachy_sdk_zeroed.mobile_base.translate_by(x=0.1, y=0.1, wait=True)
        time.sleep(0.1)
        assert is_goto_finished(reachy_sdk_zeroed, trans0)
        request0 = reachy_sdk_zeroed.get_goto_request(trans0)
        assert request0.part == "mobile_base"
        assert np.isclose(request0.request.goal_positions["x"], odom["x"] + 0.1, atol=1e-02)
        assert np.isclose(request0.request.goal_positions["y"], odom["x"] + 0.1, atol=1e-02)
        assert np.isclose(request0.request.goal_positions["theta"], odom["theta"], atol=1e-02)
        assert np.isclose(request0.request.distance_tolerance, 0.05, atol=1e-03)
        assert np.isclose(request0.request.angle_tolerance, 5, atol=1e-03)

        reachy_sdk_zeroed.mobile_base.goto(x=0.1, y=0, theta=0, angle_tolerance=2)
        trans1 = reachy_sdk_zeroed.mobile_base.translate_by(x=0.8, y=-0.1)
        trans2 = reachy_sdk_zeroed.mobile_base.translate_by(x=0.5, y=0.3, distance_tolerance=0.1)
        time.sleep(0.1)

        request1 = reachy_sdk_zeroed.get_goto_request(trans1)
        request2 = reachy_sdk_zeroed.get_goto_request(trans2)

        assert request1.part == "mobile_base"
        assert np.isclose(request1.request.goal_positions["x"], 0.9, atol=1e-03)
        assert np.isclose(request1.request.goal_positions["y"], -0.1, atol=1e-03)
        assert np.isclose(request1.request.goal_positions["theta"], 0, atol=1e-03)
        assert np.isclose(request1.request.distance_tolerance, 0.05, atol=1e-03)
        assert np.isclose(request1.request.angle_tolerance, 2, atol=1e-03)

        assert request2.part == "mobile_base"
        assert np.isclose(request2.request.goal_positions["x"], 1.4, atol=1e-03)
        assert np.isclose(request2.request.goal_positions["y"], 0.2, atol=1e-03)
        assert np.isclose(request2.request.goal_positions["theta"], 0, atol=1e-03)
        assert np.isclose(request2.request.distance_tolerance, 0.1, atol=1e-03)
        assert np.isclose(request2.request.angle_tolerance, 2, atol=1e-03)

        reachy_sdk_zeroed.mobile_base.goto(x=1.3, y=0.2, theta=50)
        trans3 = reachy_sdk_zeroed.mobile_base.translate_by(x=0, y=0.5, timeout=50)
        time.sleep(0.1)

        request3 = reachy_sdk_zeroed.get_goto_request(trans3)

        assert request3.part == "mobile_base"
        assert np.isclose(request3.request.goal_positions["x"], 0.917, atol=1e-03)
        assert np.isclose(request3.request.goal_positions["y"], 0.521, atol=1e-03)
        assert np.isclose(request3.request.goal_positions["theta"], 50, atol=1e-03)
        assert np.isclose(request3.request.distance_tolerance, 0.05, atol=1e-03)
        assert np.isclose(request3.request.angle_tolerance, 5, atol=1e-03)
        assert np.isclose(request3.request.timeout, 50, atol=1e-03)

        assert not len(reachy_sdk_zeroed.mobile_base.get_goto_queue()) == 0
        reachy_sdk_zeroed.mobile_base.cancel_all_goto()
        time.sleep(0.1)
        assert len(reachy_sdk_zeroed.mobile_base.get_goto_queue()) == 0


@pytest.mark.mobile_base
def test_mobile_base_rotate_by(reachy_sdk_zeroed: ReachySDK) -> None:
    if reachy_sdk_zeroed.mobile_base is not None:
        rot0 = reachy_sdk_zeroed.mobile_base.rotate_by(theta=35, wait=True)
        time.sleep(0.1)
        assert is_goto_finished(reachy_sdk_zeroed, rot0)
        request0 = reachy_sdk_zeroed.get_goto_request(rot0)
        assert request0.part == "mobile_base"
        assert np.isclose(request0.request.goal_positions["x"], 0, atol=1e-03)
        assert np.isclose(request0.request.goal_positions["y"], 0, atol=1e-03)
        assert np.isclose(request0.request.goal_positions["theta"], 35, atol=1e-03)
        assert np.isclose(request0.request.distance_tolerance, 0.05, atol=1e-03)
        assert np.isclose(request0.request.angle_tolerance, 5, atol=1e-03)

        reachy_sdk_zeroed.mobile_base.goto(x=0.1, y=0, theta=10, distance_tolerance=0.02)
        rot1 = reachy_sdk_zeroed.mobile_base.rotate_by(theta=50, angle_tolerance=2)
        rot2 = reachy_sdk_zeroed.mobile_base.rotate_by(theta=np.deg2rad(-20), angle_tolerance=np.deg2rad(3), degrees=False)
        time.sleep(0.1)

        request1 = reachy_sdk_zeroed.get_goto_request(rot1)
        request2 = reachy_sdk_zeroed.get_goto_request(rot2)

        assert request1.part == "mobile_base"
        assert np.isclose(request1.request.goal_positions["x"], 0.1, atol=1e-03)
        assert np.isclose(request1.request.goal_positions["y"], 0, atol=1e-03)
        assert np.isclose(request1.request.goal_positions["theta"], 60, atol=1e-03)
        assert np.isclose(request1.request.distance_tolerance, 0.02, atol=1e-03)
        assert np.isclose(request1.request.angle_tolerance, 2, atol=1e-03)

        assert request2.part == "mobile_base"
        assert np.isclose(request2.request.goal_positions["x"], 0.1, atol=1e-03)
        assert np.isclose(request2.request.goal_positions["y"], 0, atol=1e-03)
        assert np.isclose(request2.request.goal_positions["theta"], 40, atol=1e-03)
        assert np.isclose(request2.request.distance_tolerance, 0.02, atol=1e-03)
        assert np.isclose(request2.request.angle_tolerance, 3, atol=1e-03)

        assert not len(reachy_sdk_zeroed.mobile_base.get_goto_queue()) == 0
        reachy_sdk_zeroed.mobile_base.cancel_all_goto()
        time.sleep(0.1)
        assert len(reachy_sdk_zeroed.mobile_base.get_goto_queue()) == 0

        reachy_sdk_zeroed.mobile_base.goto(x=0.2, y=0.2, theta=50, wait=True)
        time.sleep(0.1)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        rot3 = reachy_sdk_zeroed.mobile_base.rotate_by(theta=np.deg2rad(-40), degrees=False, timeout=50, wait=True)
        time.sleep(0.1)

        request3 = reachy_sdk_zeroed.get_goto_request(rot3)

        assert request3.part == "mobile_base"
        assert np.isclose(request3.request.goal_positions["x"], odom["x"], atol=1e-03)
        assert np.isclose(request3.request.goal_positions["y"], odom["y"], atol=1e-03)
        assert np.isclose(request3.request.goal_positions["theta"], odom["theta"] - 40, atol=1e-03)
        assert np.isclose(request3.request.distance_tolerance, 0.05, atol=1e-03)
        assert np.isclose(request3.request.angle_tolerance, 5, atol=1e-03)
        assert np.isclose(request3.request.timeout, 50, atol=1e-03)

        assert is_goto_finished(reachy_sdk_zeroed, rot3)
