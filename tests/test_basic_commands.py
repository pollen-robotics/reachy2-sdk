import time

import numpy as np
import numpy.typing as npt
import pytest
from pyquaternion import Quaternion
from reachy2_sdk_api.goto_pb2 import GoalStatus

from src.reachy2_sdk.reachy_sdk import ReachySDK


@pytest.fixture(scope="module")
def reachy_sdk() -> ReachySDK:
    reachy = ReachySDK(host="localhost")
    assert reachy.grpc_status == "connected"

    assert reachy.turn_on()

    yield reachy

    assert reachy.turn_off()

    reachy.disconnect()
    ReachySDK.clear()


@pytest.mark.online
def test_compliancy(reachy_sdk: ReachySDK) -> None:
    reachy_sdk.turn_on()
    time.sleep(0.5)
    assert not reachy_sdk.r_arm.shoulder.compliant
    assert not reachy_sdk.r_arm.elbow.compliant
    assert not reachy_sdk.r_arm.wrist.compliant
    assert not reachy_sdk.l_arm.shoulder.compliant
    assert not reachy_sdk.l_arm.elbow.compliant
    assert not reachy_sdk.l_arm.wrist.compliant
    assert not reachy_sdk.head.neck.compliant

    reachy_sdk.turn_off()
    time.sleep(0.5)
    assert reachy_sdk.r_arm.shoulder.compliant
    assert reachy_sdk.r_arm.elbow.compliant
    assert reachy_sdk.r_arm.wrist.compliant
    assert reachy_sdk.l_arm.shoulder.compliant
    assert reachy_sdk.l_arm.elbow.compliant
    assert reachy_sdk.l_arm.wrist.compliant
    assert reachy_sdk.head.neck.compliant

    reachy_sdk.r_arm.turn_on()
    time.sleep(0.5)
    assert not reachy_sdk.r_arm.shoulder.compliant
    assert not reachy_sdk.r_arm.elbow.compliant
    assert not reachy_sdk.r_arm.wrist.compliant
    assert reachy_sdk.l_arm.shoulder.compliant
    assert reachy_sdk.l_arm.elbow.compliant
    assert reachy_sdk.l_arm.wrist.compliant
    assert reachy_sdk.head.neck.compliant

    reachy_sdk.l_arm.turn_on()
    time.sleep(0.5)
    assert not reachy_sdk.r_arm.shoulder.compliant
    assert not reachy_sdk.r_arm.elbow.compliant
    assert not reachy_sdk.r_arm.wrist.compliant
    assert not reachy_sdk.l_arm.shoulder.compliant
    assert not reachy_sdk.l_arm.elbow.compliant
    assert not reachy_sdk.l_arm.wrist.compliant
    assert reachy_sdk.head.neck.compliant

    reachy_sdk.head.turn_on()
    reachy_sdk.r_arm.turn_off()
    time.sleep(0.5)
    assert reachy_sdk.r_arm.shoulder.compliant
    assert reachy_sdk.r_arm.elbow.compliant
    assert reachy_sdk.r_arm.wrist.compliant
    assert not reachy_sdk.l_arm.shoulder.compliant
    assert not reachy_sdk.l_arm.elbow.compliant
    assert not reachy_sdk.l_arm.wrist.compliant
    assert not reachy_sdk.head.neck.compliant

    reachy_sdk.r_arm.shoulder.compliant = False
    time.sleep(0.5)
    assert not reachy_sdk.r_arm.shoulder.compliant
    assert reachy_sdk.r_arm.elbow.compliant
    assert reachy_sdk.r_arm.wrist.compliant
    assert not reachy_sdk.l_arm.shoulder.compliant
    assert not reachy_sdk.l_arm.elbow.compliant
    assert not reachy_sdk.l_arm.wrist.compliant
    assert not reachy_sdk.head.neck.compliant

    reachy_sdk.l_arm.wrist.compliant = True
    time.sleep(0.5)
    assert not reachy_sdk.r_arm.shoulder.compliant
    assert reachy_sdk.r_arm.elbow.compliant
    assert reachy_sdk.r_arm.wrist.compliant
    assert not reachy_sdk.l_arm.shoulder.compliant
    assert not reachy_sdk.l_arm.elbow.compliant
    assert reachy_sdk.l_arm.wrist.compliant
    assert not reachy_sdk.head.neck.compliant


@pytest.mark.online
def test_torque_limit(reachy_sdk: ReachySDK) -> None:
    reachy_sdk.r_arm.shoulder.set_torque_limit(50)
    time.sleep(0.5)
    assert reachy_sdk.r_arm.shoulder.get_torque_limit()["motor_1"] == 50
    assert reachy_sdk.r_arm.shoulder.get_torque_limit()["motor_2"] == 50

    reachy_sdk.l_arm.wrist.set_torque_limit(80)
    reachy_sdk.r_arm.elbow.set_torque_limit(60)
    time.sleep(0.5)
    assert reachy_sdk.l_arm.wrist.get_torque_limit()["motor_1"] == 80
    assert reachy_sdk.l_arm.wrist.get_torque_limit()["motor_2"] == 80
    assert reachy_sdk.l_arm.wrist.get_torque_limit()["motor_3"] == 80
    assert reachy_sdk.r_arm.elbow.get_torque_limit()["motor_1"] == 60
    assert reachy_sdk.r_arm.elbow.get_torque_limit()["motor_2"] == 60

    reachy_sdk.head.neck.set_torque_limit(20)
    reachy_sdk.l_arm.elbow.set_torque_limit(90)
    time.sleep(0.5)
    assert reachy_sdk.l_arm.wrist.get_torque_limit()["motor_1"] == 80
    assert reachy_sdk.l_arm.wrist.get_torque_limit()["motor_2"] == 80
    assert reachy_sdk.l_arm.wrist.get_torque_limit()["motor_3"] == 80
    assert reachy_sdk.r_arm.elbow.get_torque_limit()["motor_1"] == 60
    assert reachy_sdk.r_arm.elbow.get_torque_limit()["motor_2"] == 60

    assert reachy_sdk.head.neck.get_torque_limit()["motor_1"] == 20
    assert reachy_sdk.head.neck.get_torque_limit()["motor_2"] == 20
    assert reachy_sdk.head.neck.get_torque_limit()["motor_3"] == 20
    assert reachy_sdk.l_arm.elbow.get_torque_limit()["motor_1"] == 90
    assert reachy_sdk.l_arm.elbow.get_torque_limit()["motor_2"] == 90


@pytest.mark.online
def test_speed_limit(reachy_sdk: ReachySDK) -> None:
    reachy_sdk.r_arm.shoulder.set_speed_limit(50)
    time.sleep(0.5)
    assert reachy_sdk.r_arm.shoulder.get_speed_limit()["motor_1"] == 50
    assert reachy_sdk.r_arm.shoulder.get_speed_limit()["motor_2"] == 50

    reachy_sdk.l_arm.wrist.set_speed_limit(80)
    reachy_sdk.r_arm.elbow.set_speed_limit(60)
    time.sleep(0.5)
    assert reachy_sdk.l_arm.wrist.get_speed_limit()["motor_1"] == 80
    assert reachy_sdk.l_arm.wrist.get_speed_limit()["motor_2"] == 80
    assert reachy_sdk.l_arm.wrist.get_speed_limit()["motor_3"] == 80
    assert reachy_sdk.r_arm.elbow.get_speed_limit()["motor_1"] == 60
    assert reachy_sdk.r_arm.elbow.get_speed_limit()["motor_2"] == 60

    reachy_sdk.head.neck.set_speed_limit(20)
    reachy_sdk.l_arm.elbow.set_speed_limit(90)
    time.sleep(0.5)
    assert reachy_sdk.l_arm.wrist.get_speed_limit()["motor_1"] == 80
    assert reachy_sdk.l_arm.wrist.get_speed_limit()["motor_2"] == 80
    assert reachy_sdk.l_arm.wrist.get_speed_limit()["motor_3"] == 80
    assert reachy_sdk.r_arm.elbow.get_speed_limit()["motor_1"] == 60
    assert reachy_sdk.r_arm.elbow.get_speed_limit()["motor_2"] == 60

    assert reachy_sdk.head.neck.get_speed_limit()["motor_1"] == 20
    assert reachy_sdk.head.neck.get_speed_limit()["motor_2"] == 20
    assert reachy_sdk.head.neck.get_speed_limit()["motor_3"] == 20
    assert reachy_sdk.l_arm.elbow.get_speed_limit()["motor_1"] == 90
    assert reachy_sdk.l_arm.elbow.get_speed_limit()["motor_2"] == 90
