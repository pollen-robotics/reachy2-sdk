import threading
import time

import numpy as np
import pytest

from reachy2_sdk.reachy_sdk import ReachySDK

from .test_basic_movements import is_goto_finished

# Note: depending on the test, the expected tolerance can be very strict or depend on the requested tolerance of a goto command.

DEFAULT_DIST_TOL = 0.01  # in meters (sometimes used as m/s)
DEFAULT_ANGLE_TOL = 1.0  # in degrees (sometimes used as rad/s)

GOTO_DIST_TOL = 0.05  # in meters (default value when no tolerance is provided in goto command)
GOTO_ANGLE_TOL = 5  # in degrees (default value when no tolerance is provided in goto command)

HIGH_PRECISION_TOL = 0.0001


@pytest.mark.online
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


@pytest.mark.online
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


@pytest.mark.online
def test_reset_odometry(reachy_sdk_zeroed: ReachySDK) -> None:
    if reachy_sdk_zeroed.mobile_base is not None:
        reachy_sdk_zeroed.mobile_base.reset_odometry()
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["x"], 0.0, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["y"], 0.0, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["theta"], 0.0, atol=DEFAULT_ANGLE_TOL)

        reachy_sdk_zeroed.mobile_base.set_goal_speed(vx=0.3, vy=0.2, vtheta=20)
        tic = time.time()
        while time.time() - tic < 2:
            reachy_sdk_zeroed.mobile_base.send_speed_command()
            time.sleep(0.1)
        time.sleep(0.4)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert not np.isclose(odom["x"], 0.0, atol=0.3)
        assert not np.isclose(odom["y"], 0.0, atol=0.2)
        assert not np.isclose(odom["theta"], 0.0, atol=20)
        assert np.isclose(odom["x"], 0.43, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["y"], 0.61, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["theta"], 42.4, atol=DEFAULT_ANGLE_TOL)

        reachy_sdk_zeroed.mobile_base.reset_odometry()
        time.sleep(0.2)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["x"], 0.0, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["y"], 0.0, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["theta"], 0.0, atol=DEFAULT_ANGLE_TOL)

        dist_tol = 0.01
        angle_tol = 0.1
        reachy_sdk_zeroed.mobile_base.goto(
            x=0.5, y=0.5, theta=50, wait=True, distance_tolerance=dist_tol, angle_tolerance=angle_tol
        )
        time.sleep(0.2)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["x"], 0.5, atol=dist_tol)
        assert np.isclose(odom["y"], 0.5, atol=angle_tol)
        assert np.isclose(odom["theta"], 50.0, atol=angle_tol)

        reachy_sdk_zeroed.mobile_base.reset_odometry()
        time.sleep(0.2)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["x"], 0.0, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["y"], 0.0, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["theta"], 0.0, atol=DEFAULT_ANGLE_TOL)

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


@pytest.mark.online
def test_odometry_pos(reachy_sdk_zeroed: ReachySDK) -> None:
    if reachy_sdk_zeroed.mobile_base is not None:
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["x"], 0.0, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["y"], 0.0, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["theta"], 0.0, atol=DEFAULT_ANGLE_TOL)

        reachy_sdk_zeroed.mobile_base.set_goal_speed(vx=0.3, vy=0.0, vtheta=0)
        tic = time.time()
        while time.time() - tic < 2:
            reachy_sdk_zeroed.mobile_base.send_speed_command()
            time.sleep(0.1)
        time.sleep(0.4)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert odom["x"] > 0.6
        assert np.isclose(odom["y"], 0.0, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["theta"], 0.0, atol=DEFAULT_ANGLE_TOL)

        reachy_sdk_zeroed.mobile_base.goto(x=0.0, y=0.0, theta=0, wait=True)
        time.sleep(0.5)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["x"], 0.0, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["y"], 0.0, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["theta"], 0.0, atol=DEFAULT_ANGLE_TOL)

        reachy_sdk_zeroed.mobile_base.set_goal_speed(vx=0.0, vy=-0.5, vtheta=0)
        tic = time.time()
        while time.time() - tic < 2:
            reachy_sdk_zeroed.mobile_base.send_speed_command()
            time.sleep(0.1)
        time.sleep(0.4)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["x"], 0.0, atol=DEFAULT_DIST_TOL)
        assert odom["y"] < -1.0
        assert np.isclose(odom["theta"], 0.0, atol=DEFAULT_ANGLE_TOL)

        reachy_sdk_zeroed.mobile_base.reset_odometry()
        time.sleep(0.1)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["x"], 0.0, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["y"], 0.0, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["theta"], 0.0, atol=DEFAULT_ANGLE_TOL)

        reachy_sdk_zeroed.mobile_base.set_goal_speed(vx=0.0, vy=0.0, vtheta=50)
        tic = time.time()
        while time.time() - tic < 2:
            reachy_sdk_zeroed.mobile_base.send_speed_command()
            time.sleep(0.1)
        time.sleep(0.4)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["x"], 0.0, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["y"], 0.0, atol=DEFAULT_DIST_TOL)
        assert odom["theta"] > 50


@pytest.mark.online
def test_odometry_vel(reachy_sdk_zeroed: ReachySDK) -> None:
    if reachy_sdk_zeroed.mobile_base is not None:
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert odom["vx"] == 0.0
        assert odom["vy"] == 0.0
        assert odom["vtheta"] == 0.0

        reachy_sdk_zeroed.mobile_base.set_goal_speed(vx=0.4, vy=0.0, vtheta=0)
        tic = time.time()
        while time.time() - tic < 2:
            reachy_sdk_zeroed.mobile_base.send_speed_command()
            time.sleep(0.1)
        time.sleep(0.4)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert odom["vx"] == 0.0
        assert odom["vy"] == 0.0
        assert odom["vtheta"] == 0.0

        reachy_sdk_zeroed.mobile_base.goto(x=0.0, y=0.0, theta=0)
        time.sleep(0.3)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert odom["vx"] < 0.0
        assert np.isclose(odom["vy"], 0.0, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["vtheta"], 0.0, atol=DEFAULT_ANGLE_TOL)

        reachy_sdk_zeroed.mobile_base.goto(x=0.0, y=0.0, theta=0)
        time.sleep(0.3)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert odom["vx"] < 0.0
        assert np.isclose(odom["vy"], 0.0, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["vtheta"], 0.0, atol=DEFAULT_ANGLE_TOL)

        def send_speed_forward():
            reachy_sdk_zeroed.mobile_base.set_goal_speed(vx=0.5, vy=0.0, vtheta=0)
            tic = time.time()
            while time.time() - tic < 5:
                reachy_sdk_zeroed.mobile_base.send_speed_command()
                time.sleep(0.1)

        forward_thread = threading.Thread(target=send_speed_forward)
        forward_thread.start()
        time.sleep(1)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["vx"], 0.5, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["vy"], 0.0, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["vtheta"], 0.0, atol=DEFAULT_ANGLE_TOL)
        time.sleep(2)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["vx"], 0.5, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["vy"], 0.0, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["vtheta"], 0.0, atol=DEFAULT_ANGLE_TOL)
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
                time.sleep(0.1)

        forward_thread = threading.Thread(target=send_y_theta_speeds)
        forward_thread.start()
        time.sleep(1)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["vx"], 0.0, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["vy"], 0.4, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["vtheta"], 20, atol=DEFAULT_ANGLE_TOL)
        time.sleep(2)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["vx"], 0.0, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["vy"], 0.4, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["vtheta"], 20, atol=DEFAULT_ANGLE_TOL)
        time.sleep(3)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert odom["vx"] == 0.0
        assert odom["vy"] == 0.0
        assert odom["vtheta"] == 0.0


@pytest.mark.online
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
        # At first, the precision is expected to be only slightly better than the default tolerance of the goto command.
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["x"], 0.5, atol=GOTO_DIST_TOL)
        assert np.isclose(odom["y"], 0.5, atol=GOTO_DIST_TOL)
        assert np.isclose(odom["theta"], 50, atol=GOTO_ANGLE_TOL)
        time.sleep(0.5)
        # But after a while, the precision should be much better.
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["x"], 0.5, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["y"], 0.5, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["theta"], 50, atol=DEFAULT_ANGLE_TOL)

        goto2 = reachy_sdk_zeroed.mobile_base.goto(x=0.8, y=0.2, theta=-20, wait=True)
        assert is_goto_finished(reachy_sdk_zeroed, goto2)
        time.sleep(0.5)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["x"], 0.8, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["y"], 0.2, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["theta"], -20, atol=DEFAULT_ANGLE_TOL)

        goto3 = reachy_sdk_zeroed.mobile_base.goto(x=0.0, y=-0.4, theta=30)
        time.sleep(0.1)
        assert not is_goto_finished(reachy_sdk_zeroed, goto3)
        time.sleep(0.2)
        reachy_sdk_zeroed.cancel_goto_by_id(goto3)
        time.sleep(0.1)
        assert is_goto_finished(reachy_sdk_zeroed, goto3)
        request3 = reachy_sdk_zeroed.get_goto_request(goto3)
        assert request3.part == "mobile_base"
        assert np.isclose(request3.request.target["x"], 0.0, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request3.request.target["y"], -0.4, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request3.request.target["theta"], 30, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request3.request.distance_tolerance, 0.05, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request3.request.angle_tolerance, 5, atol=HIGH_PRECISION_TOL)

        dist_tol = 0.02
        angle_tol = 2.0
        goto4 = reachy_sdk_zeroed.mobile_base.goto(
            x=-0.2, y=0.3, theta=70, distance_tolerance=dist_tol, angle_tolerance=angle_tol, wait=True
        )
        time.sleep(0.05)
        assert is_goto_finished(reachy_sdk_zeroed, goto4)
        request4 = reachy_sdk_zeroed.get_goto_request(goto4)
        assert request4.part == "mobile_base"
        assert np.isclose(request4.request.target["x"], -0.2, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request4.request.target["y"], 0.3, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request4.request.target["theta"], 70, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request4.request.distance_tolerance, 0.02, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request4.request.angle_tolerance, 2, atol=HIGH_PRECISION_TOL)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["x"], -0.2, atol=dist_tol)
        assert np.isclose(odom["y"], 0.3, atol=dist_tol)
        assert np.isclose(odom["theta"], 70, atol=angle_tol)

        goto5 = reachy_sdk_zeroed.mobile_base.goto(x=0, y=0.3, theta=np.deg2rad(60), distance_tolerance=0.02, degrees=False)
        request5 = reachy_sdk_zeroed.get_goto_request(goto5)
        assert request5.part == "mobile_base"
        assert np.isclose(request5.request.target["x"], 0, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request5.request.target["y"], 0.3, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request5.request.target["theta"], 60, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request5.request.distance_tolerance, 0.02, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request5.request.angle_tolerance, 5, atol=HIGH_PRECISION_TOL)

        reachy_sdk_zeroed.cancel_goto_by_id(goto5)
        time.sleep(0.1)
        assert is_goto_finished(reachy_sdk_zeroed, goto5)

        goto6 = reachy_sdk_zeroed.mobile_base.goto(
            x=0, y=0.3, theta=np.deg2rad(40), distance_tolerance=dist_tol, angle_tolerance=np.deg2rad(angle_tol), degrees=False
        )
        request6 = reachy_sdk_zeroed.get_goto_request(goto6)
        assert request6.part == "mobile_base"
        assert np.isclose(request6.request.target["x"], 0, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request6.request.target["y"], 0.3, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request6.request.target["theta"], 40, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request6.request.distance_tolerance, 0.02, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request6.request.angle_tolerance, 2, atol=HIGH_PRECISION_TOL)

        reachy_sdk_zeroed.cancel_goto_by_id(goto6)
        time.sleep(0.1)
        assert is_goto_finished(reachy_sdk_zeroed, goto6)


@pytest.mark.online
def test_mobile_base_goto_timeout(reachy_sdk_zeroed: ReachySDK) -> None:
    if reachy_sdk_zeroed.mobile_base is not None:
        with pytest.raises(ValueError):
            reachy_sdk_zeroed.mobile_base.goto(x=0.0, y=0.6, theta=0, timeout=0.0)

        with pytest.raises(TypeError):
            reachy_sdk_zeroed.mobile_base.goto(x=0.0, y=0.6, theta=0, timeout="mistake")

        goto1 = reachy_sdk_zeroed.mobile_base.goto(x=0.8, y=0.5, theta=80, timeout=1, wait=True)
        time.sleep(0.05)
        assert is_goto_finished(reachy_sdk_zeroed, goto1)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert not np.isclose(odom["x"], 0.8, atol=GOTO_DIST_TOL)
        assert not np.isclose(odom["y"], 0.5, atol=GOTO_DIST_TOL)
        assert not np.isclose(odom["theta"], 80, atol=GOTO_ANGLE_TOL)
        time.sleep(0.5)  # Giving it some time to get closer
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert not np.isclose(odom["x"], 0.8, atol=DEFAULT_DIST_TOL)
        assert not np.isclose(odom["y"], 0.5, atol=DEFAULT_DIST_TOL)
        assert not np.isclose(odom["theta"], 80, atol=DEFAULT_ANGLE_TOL)

        request1 = reachy_sdk_zeroed.get_goto_request(goto1)
        assert request1.part == "mobile_base"
        assert np.isclose(request1.request.target["x"], 0.8, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request1.request.target["y"], 0.5, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request1.request.target["theta"], 80, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request1.request.timeout, 1, atol=HIGH_PRECISION_TOL)

        tic = time.time()
        goto2 = reachy_sdk_zeroed.mobile_base.goto(x=0.0, y=-0.2, theta=0, timeout=0.8)
        while not is_goto_finished(reachy_sdk_zeroed, goto2):
            time.sleep(0.01)
        assert np.isclose(time.time() - tic, 0.8, atol=0.2)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert not np.isclose(odom["x"], 0.0, atol=DEFAULT_DIST_TOL)
        assert not np.isclose(odom["y"], -0.2, atol=DEFAULT_DIST_TOL)
        assert not np.isclose(odom["theta"], 0, atol=DEFAULT_ANGLE_TOL)

        request2 = reachy_sdk_zeroed.get_goto_request(goto2)
        assert request2.part == "mobile_base"
        assert np.isclose(request2.request.target["x"], 0.0, atol=DEFAULT_DIST_TOL)
        assert np.isclose(request2.request.target["y"], -0.2, atol=DEFAULT_DIST_TOL)
        assert np.isclose(request2.request.target["theta"], 0, atol=DEFAULT_ANGLE_TOL)
        assert np.isclose(request2.request.timeout, 0.8, atol=DEFAULT_DIST_TOL)


@pytest.mark.online
def test_mobile_base_goto_tolerances(reachy_sdk_zeroed: ReachySDK) -> None:
    if reachy_sdk_zeroed.mobile_base is not None:
        with pytest.raises(ValueError):
            reachy_sdk_zeroed.mobile_base.goto(x=0.0, y=0.6, theta=0, distance_tolerance=-0.2)
        with pytest.raises(ValueError):
            reachy_sdk_zeroed.mobile_base.goto(x=0.0, y=0.6, theta=0, angle_tolerance=-10)

        with pytest.raises(TypeError):
            reachy_sdk_zeroed.mobile_base.goto(x=0.0, y=0.6, theta=0, distance_tolerance="mistake")
        with pytest.raises(TypeError):
            reachy_sdk_zeroed.mobile_base.goto(x=0.0, y=0.6, theta=0, angle_tolerance="mistake")

        dist_tol = 0.2
        goto1 = reachy_sdk_zeroed.mobile_base.goto(x=0.9, y=0.9, theta=80, distance_tolerance=dist_tol, wait=True)
        time.sleep(0.05)
        assert is_goto_finished(reachy_sdk_zeroed, goto1)
        # At first, the precision is expected to be only slightly better than the provided tolerance.
        odom1 = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert not np.isclose(odom1["x"], 0.9, atol=DEFAULT_DIST_TOL)
        assert not np.isclose(odom1["y"], 0.9, atol=DEFAULT_DIST_TOL)
        time.sleep(2.0)
        odom2 = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom2["x"], 0.9, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom2["y"], 0.9, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom2["theta"], 80, atol=DEFAULT_ANGLE_TOL)

        request1 = reachy_sdk_zeroed.get_goto_request(goto1)
        assert request1.part == "mobile_base"
        assert np.isclose(request1.request.target["x"], 0.9, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request1.request.target["y"], 0.9, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request1.request.target["theta"], 80, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request1.request.distance_tolerance, 0.2, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request1.request.angle_tolerance, 5, atol=HIGH_PRECISION_TOL)

        goto2 = reachy_sdk_zeroed.mobile_base.goto(x=0.0, y=0.0, theta=0, distance_tolerance=0.02)
        goto3 = reachy_sdk_zeroed.mobile_base.goto(x=0.9, y=0.0, theta=20, distance_tolerance=0.3)
        goto4 = reachy_sdk_zeroed.mobile_base.goto(x=0.1, y=0.0, theta=0, distance_tolerance=0.2)

        while not is_goto_finished(reachy_sdk_zeroed, goto2):
            time.sleep(0.05)
        odom2 = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom2["x"], 0.0, atol=0.02)
        assert np.isclose(odom2["y"], 0.0, atol=0.02)
        assert np.isclose(odom2["theta"], 0, atol=GOTO_ANGLE_TOL)
        while not is_goto_finished(reachy_sdk_zeroed, goto3):
            time.sleep(0.05)
        odom3 = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom3["x"], 0.6, atol=0.3)
        assert np.isclose(odom3["y"], 0.0, atol=0.3)
        assert np.isclose(odom3["theta"], 20, atol=GOTO_ANGLE_TOL)
        while not is_goto_finished(reachy_sdk_zeroed, goto4):
            time.sleep(0.05)
        odom4 = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom4["x"], 0.3, atol=0.2)
        assert np.isclose(odom4["y"], 0.0, atol=0.2)
        assert np.isclose(odom4["theta"], 0, atol=GOTO_ANGLE_TOL)
        time.sleep(1.0)
        odom4_bis = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom4_bis["x"], 0.1, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom4_bis["y"], 0.0, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom4_bis["theta"], 0, atol=DEFAULT_ANGLE_TOL)

        goto5 = reachy_sdk_zeroed.mobile_base.goto(x=0.1, y=0.0, theta=60, angle_tolerance=20)
        goto6 = reachy_sdk_zeroed.mobile_base.goto(x=0.1, y=0.0, theta=-30, angle_tolerance=20)

        while not is_goto_finished(reachy_sdk_zeroed, goto5):
            time.sleep(0.05)
        odom5 = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom5["x"], 0.1, atol=GOTO_DIST_TOL)
        assert np.isclose(odom5["y"], 0.0, atol=GOTO_DIST_TOL)
        assert np.isclose(odom5["theta"], 40, atol=5)

        while not is_goto_finished(reachy_sdk_zeroed, goto6):
            time.sleep(0.05)
        odom6 = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom6["x"], 0.1, atol=GOTO_DIST_TOL)
        assert np.isclose(odom6["y"], 0.0, atol=GOTO_DIST_TOL)
        assert np.isclose(odom6["theta"], -10, atol=10)
        time.sleep(2)
        odom6_bis = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom6_bis["x"], 0.1, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom6_bis["y"], 0.0, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom6_bis["theta"], -30, atol=DEFAULT_ANGLE_TOL)


@pytest.mark.online
def test_mobile_base_translate_by(reachy_sdk_zeroed: ReachySDK) -> None:
    if reachy_sdk_zeroed.mobile_base is not None:
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        trans0 = reachy_sdk_zeroed.mobile_base.translate_by(x=0.1, y=0.1, wait=True)
        time.sleep(0.1)
        assert is_goto_finished(reachy_sdk_zeroed, trans0)
        request0 = reachy_sdk_zeroed.get_goto_request(trans0)
        assert request0.part == "mobile_base"
        assert np.isclose(request0.request.target["x"], odom["x"] + 0.1, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request0.request.target["y"], odom["x"] + 0.1, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request0.request.target["theta"], odom["theta"], atol=HIGH_PRECISION_TOL)
        assert np.isclose(request0.request.distance_tolerance, 0.05, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request0.request.angle_tolerance, 5, atol=HIGH_PRECISION_TOL)

        reachy_sdk_zeroed.mobile_base.goto(x=0.1, y=0, theta=0, angle_tolerance=2)
        trans1 = reachy_sdk_zeroed.mobile_base.translate_by(x=0.8, y=-0.1)
        trans2 = reachy_sdk_zeroed.mobile_base.translate_by(x=0.5, y=0.3, distance_tolerance=0.1)
        time.sleep(0.1)

        request1 = reachy_sdk_zeroed.get_goto_request(trans1)
        request2 = reachy_sdk_zeroed.get_goto_request(trans2)

        assert request1.part == "mobile_base"
        assert np.isclose(request1.request.target["x"], 0.9, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request1.request.target["y"], -0.1, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request1.request.target["theta"], 0, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request1.request.distance_tolerance, 0.05, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request1.request.angle_tolerance, 2, atol=HIGH_PRECISION_TOL)

        assert request2.part == "mobile_base"
        assert np.isclose(request2.request.target["x"], 1.4, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request2.request.target["y"], 0.2, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request2.request.target["theta"], 0, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request2.request.distance_tolerance, 0.1, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request2.request.angle_tolerance, 2, atol=HIGH_PRECISION_TOL)

        reachy_sdk_zeroed.mobile_base.goto(x=1.3, y=0.2, theta=50)
        trans3 = reachy_sdk_zeroed.mobile_base.translate_by(x=0, y=0.5, timeout=50)
        time.sleep(0.1)

        request3 = reachy_sdk_zeroed.get_goto_request(trans3)

        assert request3.part == "mobile_base"
        assert np.isclose(request3.request.target["x"], 0.917, atol=DEFAULT_DIST_TOL)
        assert np.isclose(request3.request.target["y"], 0.521, atol=DEFAULT_DIST_TOL)
        assert np.isclose(request3.request.target["theta"], 50, atol=DEFAULT_ANGLE_TOL)
        assert np.isclose(request3.request.distance_tolerance, 0.05, atol=DEFAULT_DIST_TOL)
        assert np.isclose(request3.request.angle_tolerance, 5, atol=DEFAULT_ANGLE_TOL)
        assert np.isclose(request3.request.timeout, 50, atol=HIGH_PRECISION_TOL)

        assert not len(reachy_sdk_zeroed.mobile_base.get_goto_queue()) == 0
        reachy_sdk_zeroed.mobile_base.cancel_all_goto()
        time.sleep(0.1)
        assert len(reachy_sdk_zeroed.mobile_base.get_goto_queue()) == 0


@pytest.mark.online
def test_mobile_base_rotate_by(reachy_sdk_zeroed: ReachySDK) -> None:
    if reachy_sdk_zeroed.mobile_base is not None:
        rot0 = reachy_sdk_zeroed.mobile_base.rotate_by(theta=35, wait=True)
        time.sleep(0.1)
        assert is_goto_finished(reachy_sdk_zeroed, rot0)
        request0 = reachy_sdk_zeroed.get_goto_request(rot0)
        assert request0.part == "mobile_base"
        assert np.isclose(request0.request.target["x"], 0, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request0.request.target["y"], 0, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request0.request.target["theta"], 35, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request0.request.distance_tolerance, 0.05, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request0.request.angle_tolerance, 5, atol=HIGH_PRECISION_TOL)

        reachy_sdk_zeroed.mobile_base.goto(x=0.1, y=0, theta=10, distance_tolerance=0.02)
        rot1 = reachy_sdk_zeroed.mobile_base.rotate_by(theta=50, angle_tolerance=2)
        rot2 = reachy_sdk_zeroed.mobile_base.rotate_by(theta=np.deg2rad(-20), angle_tolerance=np.deg2rad(3), degrees=False)
        time.sleep(0.1)

        request1 = reachy_sdk_zeroed.get_goto_request(rot1)
        request2 = reachy_sdk_zeroed.get_goto_request(rot2)

        assert request1.part == "mobile_base"
        assert np.isclose(request1.request.target["x"], 0.1, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request1.request.target["y"], 0, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request1.request.target["theta"], 60, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request1.request.distance_tolerance, 0.02, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request1.request.angle_tolerance, 2, atol=HIGH_PRECISION_TOL)

        assert request2.part == "mobile_base"
        assert np.isclose(request2.request.target["x"], 0.1, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request2.request.target["y"], 0, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request2.request.target["theta"], 40, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request2.request.distance_tolerance, 0.02, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request2.request.angle_tolerance, 3, atol=HIGH_PRECISION_TOL)

        assert not len(reachy_sdk_zeroed.mobile_base.get_goto_queue()) == 0
        reachy_sdk_zeroed.mobile_base.cancel_all_goto()
        time.sleep(0.1)
        assert len(reachy_sdk_zeroed.mobile_base.get_goto_queue()) == 0

        reachy_sdk_zeroed.mobile_base.goto(x=0.2, y=0.2, theta=50, wait=True)
        time.sleep(0.3)
        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        rot3 = reachy_sdk_zeroed.mobile_base.rotate_by(theta=np.deg2rad(-40), degrees=False, timeout=50, wait=True)
        time.sleep(0.1)

        request3 = reachy_sdk_zeroed.get_goto_request(rot3)

        assert request3.part == "mobile_base"
        assert np.isclose(request3.request.target["x"], odom["x"], atol=1e-2)
        assert np.isclose(request3.request.target["y"], odom["y"], atol=1e-2)
        assert np.isclose(request3.request.target["theta"], odom["theta"] - 40, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request3.request.distance_tolerance, 0.05, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request3.request.angle_tolerance, 5, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request3.request.timeout, 50, atol=HIGH_PRECISION_TOL)

        assert is_goto_finished(reachy_sdk_zeroed, rot3)


@pytest.mark.online
def test_set_max_xy_goto(reachy_sdk_zeroed: ReachySDK) -> None:
    if reachy_sdk_zeroed.mobile_base is not None:
        assert reachy_sdk_zeroed.mobile_base._max_xy_goto == 1.0
        with pytest.raises(ValueError):
            reachy_sdk_zeroed.mobile_base.goto(x=1.5, y=0.2, theta=10)
        reachy_sdk_zeroed.mobile_base.set_max_xy_goto(2.0)
        assert reachy_sdk_zeroed.mobile_base._max_xy_goto == 2.0

        goto1 = reachy_sdk_zeroed.mobile_base.goto(x=1.5, y=0.2, theta=10, wait=True)
        request1 = reachy_sdk_zeroed.get_goto_request(goto1)
        time.sleep(0.5)

        assert request1.part == "mobile_base"
        assert np.isclose(request1.request.target["x"], 1.5, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request1.request.target["y"], 0.2, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request1.request.target["theta"], 10, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request1.request.distance_tolerance, 0.05, atol=HIGH_PRECISION_TOL)
        assert np.isclose(request1.request.angle_tolerance, 5, atol=HIGH_PRECISION_TOL)

        odom = reachy_sdk_zeroed.mobile_base.get_current_odometry()
        assert np.isclose(odom["x"], 1.5, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["y"], 0.2, atol=DEFAULT_DIST_TOL)
        assert np.isclose(odom["theta"], 10, atol=DEFAULT_DIST_TOL)

        reachy_sdk_zeroed.mobile_base.set_max_xy_goto(1.0)


@pytest.mark.mobile_base
def test_get_map(reachy_sdk_zeroed: ReachySDK) -> None:
    if reachy_sdk_zeroed.mobile_base is not None:
        map = reachy_sdk_zeroed.mobile_base.lidar.get_map()
        assert map is not None


@pytest.mark.mobile_base
def test_obstacle_detection(reachy_sdk_zeroed: ReachySDK) -> None:
    if reachy_sdk_zeroed.mobile_base is not None:
        status = reachy_sdk_zeroed.mobile_base.lidar.obstacle_detection_status
        assert status == "NO_OBJECT_DETECTED"
