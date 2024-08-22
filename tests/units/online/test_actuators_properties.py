import time

import pytest

from reachy2_sdk.reachy_sdk import ReachySDK


@pytest.mark.online
def test_on_off(reachy_sdk_zeroed: ReachySDK) -> None:
    reachy_sdk_zeroed.turn_off_smoothly()
    time.sleep(2.5)
    assert reachy_sdk_zeroed.is_off()
    assert not reachy_sdk_zeroed.is_on()
    assert reachy_sdk_zeroed.r_arm.is_off()
    assert reachy_sdk_zeroed.l_arm.is_off()
    assert reachy_sdk_zeroed.head.is_off()
    if reachy_sdk_zeroed.mobile_base is not None:
        assert reachy_sdk_zeroed.mobile_base.is_off()

    reachy_sdk_zeroed.turn_on()
    time.sleep(0.5)
    assert not reachy_sdk_zeroed.is_off()
    assert reachy_sdk_zeroed.is_on()
    assert reachy_sdk_zeroed.r_arm.is_on()
    assert reachy_sdk_zeroed.l_arm.is_on()
    assert reachy_sdk_zeroed.head.is_on()
    if reachy_sdk_zeroed.mobile_base is not None:
        assert reachy_sdk_zeroed.mobile_base.is_on()

    reachy_sdk_zeroed.r_arm.shoulder.turn_off()
    time.sleep(0.5)
    assert not reachy_sdk_zeroed.r_arm.shoulder.is_on()
    assert reachy_sdk_zeroed.r_arm.elbow.is_on()
    assert reachy_sdk_zeroed.r_arm.wrist.is_on()
    assert not reachy_sdk_zeroed.r_arm.is_on()
    assert not reachy_sdk_zeroed.r_arm.is_off()
    assert reachy_sdk_zeroed.l_arm.is_on()
    assert reachy_sdk_zeroed.head.is_on()
    if reachy_sdk_zeroed.mobile_base is not None:
        assert reachy_sdk_zeroed.mobile_base.is_on()
    assert not reachy_sdk_zeroed.is_off()
    assert not reachy_sdk_zeroed.is_on()

    reachy_sdk_zeroed.r_arm.shoulder.turn_on()
    reachy_sdk_zeroed.head.neck.turn_off()
    time.sleep(0.5)
    assert reachy_sdk_zeroed.r_arm.is_on()
    assert not reachy_sdk_zeroed.r_arm.is_off()
    assert reachy_sdk_zeroed.l_arm.is_on()
    assert not reachy_sdk_zeroed.head.is_on()
    if reachy_sdk_zeroed.mobile_base is not None:
        assert reachy_sdk_zeroed.mobile_base.is_on()
    assert not reachy_sdk_zeroed.is_off()
    assert not reachy_sdk_zeroed.is_on()

    reachy_sdk_zeroed.turn_on()


@pytest.mark.online
def test_torque_limits(reachy_sdk_zeroed: ReachySDK) -> None:
    reachy_sdk_zeroed.r_arm.set_torque_limits(90)
    time.sleep(0.2)
    s_torques = reachy_sdk_zeroed.r_arm.shoulder.get_torque_limits()
    e_torques = reachy_sdk_zeroed.r_arm.elbow.get_torque_limits()
    w_torques = reachy_sdk_zeroed.r_arm.wrist.get_torque_limits()
    assert s_torques["motor_1"] == 90
    assert s_torques["motor_2"] == 90
    assert e_torques["motor_1"] == 90
    assert e_torques["motor_2"] == 90
    assert w_torques["motor_1"] == 90
    assert w_torques["motor_2"] == 90
    assert w_torques["motor_3"] == 90

    reachy_sdk_zeroed.r_arm.elbow.set_torque_limits(50)
    time.sleep(0.2)
    s_torques = reachy_sdk_zeroed.r_arm.shoulder.get_torque_limits()
    e_torques = reachy_sdk_zeroed.r_arm.elbow.get_torque_limits()
    w_torques = reachy_sdk_zeroed.r_arm.wrist.get_torque_limits()
    assert s_torques["motor_1"] == 90
    assert s_torques["motor_2"] == 90
    assert e_torques["motor_1"] == 50
    assert e_torques["motor_2"] == 50
    assert w_torques["motor_1"] == 90
    assert w_torques["motor_2"] == 90
    assert w_torques["motor_3"] == 90

    reachy_sdk_zeroed.head.neck.set_torque_limits(30)
    time.sleep(0.2)
    n_torques = reachy_sdk_zeroed.head.neck.get_torque_limits()
    assert n_torques["motor_1"] == 30
    assert n_torques["motor_2"] == 30
    assert n_torques["motor_3"] == 30


@pytest.mark.online
def test_speed_limits(reachy_sdk_zeroed: ReachySDK) -> None:
    reachy_sdk_zeroed.l_arm.set_speed_limits(70)
    time.sleep(0.2)
    s_speeds = reachy_sdk_zeroed.l_arm.shoulder.get_speed_limits()
    e_speeds = reachy_sdk_zeroed.l_arm.elbow.get_speed_limits()
    w_speeds = reachy_sdk_zeroed.l_arm.wrist.get_speed_limits()
    assert s_speeds["motor_1"] == 70
    assert s_speeds["motor_2"] == 70
    assert e_speeds["motor_1"] == 70
    assert e_speeds["motor_2"] == 70
    assert w_speeds["motor_1"] == 70
    assert w_speeds["motor_2"] == 70
    assert w_speeds["motor_3"] == 70

    reachy_sdk_zeroed.l_arm.elbow.set_speed_limits(40)
    time.sleep(0.2)
    s_speeds = reachy_sdk_zeroed.l_arm.shoulder.get_speed_limits()
    e_speeds = reachy_sdk_zeroed.l_arm.elbow.get_speed_limits()
    w_speeds = reachy_sdk_zeroed.l_arm.wrist.get_speed_limits()
    assert s_speeds["motor_1"] == 70
    assert s_speeds["motor_2"] == 70
    assert e_speeds["motor_1"] == 40
    assert e_speeds["motor_2"] == 40
    assert w_speeds["motor_1"] == 70
    assert w_speeds["motor_2"] == 70
    assert w_speeds["motor_3"] == 70

    reachy_sdk_zeroed.head.neck.set_speed_limits(30)
    time.sleep(0.2)
    n_speeds = reachy_sdk_zeroed.head.neck.get_speed_limits()
    assert n_speeds["motor_1"] == 30
    assert n_speeds["motor_2"] == 30
    assert n_speeds["motor_3"] == 30


@pytest.mark.online
def test_reset_default_limits(reachy_sdk_zeroed: ReachySDK) -> None:
    reachy_sdk_zeroed.l_arm.set_speed_limits(70)
    reachy_sdk_zeroed.l_arm.set_torque_limits(70)
    reachy_sdk_zeroed.r_arm.set_speed_limits(50)
    reachy_sdk_zeroed.r_arm.set_torque_limits(50)
    reachy_sdk_zeroed.head.neck.set_torque_limits(40)
    reachy_sdk_zeroed.head.neck.set_speed_limits(40)
    time.sleep(0.2)

    assert reachy_sdk_zeroed.l_arm.shoulder.get_speed_limits()['motor_1'] == 70
    assert reachy_sdk_zeroed.l_arm.elbow.get_torque_limits()['motor_2'] == 70
    assert reachy_sdk_zeroed.r_arm.elbow.get_speed_limits()['motor_1'] == 50
    assert reachy_sdk_zeroed.r_arm.wrist.get_torque_limits()['motor_3'] == 50
    print(reachy_sdk_zeroed.head.neck.get_torque_limits())
    print(reachy_sdk_zeroed.r_arm.wrist.get_torque_limits())

    assert reachy_sdk_zeroed.head.neck.get_torque_limits()['motor_2'] == 40
    assert reachy_sdk_zeroed.head.neck.get_speed_limits()['motor_3'] == 40

    reachy_sdk_zeroed.reset_default_limits()

    s_l_speeds = reachy_sdk_zeroed.l_arm.shoulder.get_speed_limits()
    e_l_speeds = reachy_sdk_zeroed.l_arm.elbow.get_speed_limits()
    w_l_speeds = reachy_sdk_zeroed.l_arm.wrist.get_speed_limits()
    assert s_l_speeds["motor_1"] == 100
    assert s_l_speeds["motor_2"] == 100
    assert e_l_speeds["motor_1"] == 100
    assert e_l_speeds["motor_2"] == 100
    assert w_l_speeds["motor_1"] == 100
    assert w_l_speeds["motor_2"] == 100
    assert w_l_speeds["motor_3"] == 100

    s_r_speeds = reachy_sdk_zeroed.r_arm.shoulder.get_speed_limits()
    e_r_speeds = reachy_sdk_zeroed.r_arm.elbow.get_speed_limits()
    w_r_speeds = reachy_sdk_zeroed.r_arm.wrist.get_speed_limits()
    assert s_r_speeds["motor_1"] == 100
    assert s_r_speeds["motor_2"] == 100
    assert e_r_speeds["motor_1"] == 100
    assert e_r_speeds["motor_2"] == 100
    assert w_r_speeds["motor_1"] == 100
    assert w_r_speeds["motor_2"] == 100
    assert w_r_speeds["motor_3"] == 100

    h_speeds = reachy_sdk_zeroed.head.neck.get_speed_limits()
    assert h_speeds["motor_1"] == 100
    assert h_speeds["motor_2"] == 100
    assert h_speeds["motor_3"] == 100

    s_l_torques = reachy_sdk_zeroed.l_arm.shoulder.get_torque_limits()
    e_l_torques = reachy_sdk_zeroed.l_arm.elbow.get_torque_limits()
    w_l_torques = reachy_sdk_zeroed.l_arm.wrist.get_torque_limits()
    assert s_l_torques["motor_1"] == 100
    assert s_l_torques["motor_2"] == 100
    assert e_l_torques["motor_1"] == 100
    assert e_l_torques["motor_2"] == 100
    assert w_l_torques["motor_1"] == 100
    assert w_l_torques["motor_2"] == 100
    assert w_l_torques["motor_3"] == 100

    s_r_torques = reachy_sdk_zeroed.r_arm.shoulder.get_torque_limits()
    e_r_torques = reachy_sdk_zeroed.r_arm.elbow.get_torque_limits()
    w_r_torques = reachy_sdk_zeroed.r_arm.wrist.get_torque_limits()
    assert s_r_torques["motor_1"] == 100
    assert s_r_torques["motor_2"] == 100
    assert e_r_torques["motor_1"] == 100
    assert e_r_torques["motor_2"] == 100
    assert w_r_torques["motor_1"] == 100
    assert w_r_torques["motor_2"] == 100
    assert w_r_torques["motor_3"] == 100

    h_torques = reachy_sdk_zeroed.head.neck.get_torque_limits()
    assert h_torques["motor_1"] == 100
    assert h_torques["motor_2"] == 100
    assert h_torques["motor_3"] == 100