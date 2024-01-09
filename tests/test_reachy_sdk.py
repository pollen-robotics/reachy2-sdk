import pytest

from src.reachy2_sdk.reachy_sdk import ReachySDK


def test_unconnected() -> None:
    rsdk = ReachySDK(host="dummy")
    with pytest.raises(ConnectionError):
        rsdk2 = ReachySDK(host="dummy2")

    assert rsdk._grpc_connected is False
    assert rsdk.grpc_status == "disconnected"

    with pytest.raises(ValueError):
        rsdk._grpc_status = "test_test"

    with pytest.raises(ConnectionError):
        rsdk._get_info()

    assert len(rsdk.enabled_parts) == 0

    assert len(rsdk.disabled_parts) == 0

    assert len(rsdk.joints) == 0

    assert len(rsdk.actuators) == 0

    assert rsdk.turn_on() is False

    assert rsdk.turn_off() is False
