import pytest
from logging import getLogger
from reachy2_sdk_api.goto_pb2 import GoToId

from reachy2_sdk.reachy_sdk import ReachySDK
import logging


@pytest.mark.offline
def test_multiple_connections() -> None:
    rsdk = ReachySDK(host="dummy")
    rsdk2 = ReachySDK(host="dummy2")
    assert not rsdk.is_connected()
    assert not rsdk2.is_connected()
    rsdk.disconnect()
    rsdk2.disconnect()


@pytest.mark.offline
def test_unconnected() -> None:
    rsdk = ReachySDK(host="dummy")

    assert str(rsdk) == "Reachy is not connected"

    assert rsdk._grpc_connected is False
    assert not rsdk.is_connected()

    with pytest.raises(ConnectionError):
        rsdk._get_info()

    assert len(rsdk.joints) == 0

    assert len(rsdk._actuators) == 0

    assert rsdk.turn_on() is False

    assert rsdk.turn_off() is False

    rsdk.disconnect()

@pytest.mark.offline
def test_hostname(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        rsdk = ReachySDK(host=10)
    assert "The IP address needs to be in string format." in caplog.text
    assert any(record.message == "The IP address needs to be in string format." for record in caplog.records)
        

@pytest.mark.offline
def test_getters_setters() -> None:
    rsdk = ReachySDK(host="dummy")

    assert rsdk.r_arm is None

    assert rsdk.l_arm is None

    assert rsdk.head is None

    assert rsdk.mobile_base is None

    assert rsdk.cameras is None

    assert rsdk.joints == {}

    rsdk.disconnect()


@pytest.mark.offline
def test_unconnected() -> None:
    rsdk = ReachySDK(host="dummy")

    assert rsdk.is_connected() == False

    assert rsdk.info is None

    assert rsdk.turn_on() == False
    assert rsdk.turn_off() == False
    assert rsdk.turn_off_smoothly() == False
    assert rsdk.is_on() == False
    assert rsdk.is_off() == True

    assert rsdk.cancel_all_moves() is None
    assert rsdk.cancel_move_by_id(GoToId(id=1)) is None
    assert rsdk.get_move_joints_request(GoToId(id=1)) is None

    assert rsdk.is_move_finished(GoToId(id=1)) is False
    assert rsdk.is_move_playing(GoToId(id=1)) is False
