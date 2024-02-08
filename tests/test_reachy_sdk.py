import pytest

from src.reachy2_sdk.reachy_sdk import ReachySDK


@pytest.mark.offline
def test_singleton() -> None:
    rsdk = ReachySDK(host="dummy")
    with pytest.raises(ConnectionError):
        rsdk2 = ReachySDK(host="dummy2")
    rsdk.disconnect()
    ReachySDK.clear()


@pytest.mark.offline
def test_unconnected() -> None:
    rsdk = ReachySDK(host="dummy")

    assert rsdk._grpc_connected is False
    assert not rsdk.is_connected()

    with pytest.raises(ConnectionError):
        rsdk._get_info()

    assert len(rsdk.joints) == 0

    assert len(rsdk._actuators) == 0

    assert rsdk.turn_on() is False

    assert rsdk.turn_off() is False

    rsdk.disconnect()
    ReachySDK.clear()


@pytest.mark.offline
def test_getters_setters() -> None:
    rsdk = ReachySDK(host="dummy")

    with pytest.raises(AttributeError):
        rsdk.r_arm

    with pytest.raises(AttributeError):
        rsdk.l_arm

    with pytest.raises(AttributeError):
        rsdk.head

    rsdk.disconnect()
    ReachySDK.clear()
