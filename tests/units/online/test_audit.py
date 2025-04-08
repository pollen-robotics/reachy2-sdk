import pytest

from reachy2_sdk.reachy_sdk import ReachySDK


@pytest.mark.online
def test_audit(reachy_sdk_zeroed: ReachySDK) -> None:
    assert reachy_sdk_zeroed.r_arm.shoulder.status == "Ok"
    assert reachy_sdk_zeroed.r_arm.elbow.status == "Ok"
    assert reachy_sdk_zeroed.r_arm.wrist.status == "Ok"
    assert reachy_sdk_zeroed.l_arm.shoulder.status == "Ok"
    assert reachy_sdk_zeroed.l_arm.elbow.status == "Ok"
    assert reachy_sdk_zeroed.l_arm.wrist.status == "Ok"
    assert reachy_sdk_zeroed.head.neck.status == "Ok"

    assert reachy_sdk_zeroed.r_arm.audit == {
        "shoulder": reachy_sdk_zeroed.r_arm.shoulder.status,
        "elbow": reachy_sdk_zeroed.r_arm.elbow.status,
        "wrist": reachy_sdk_zeroed.r_arm.wrist.status,
        "gripper": None,
    }
    assert reachy_sdk_zeroed.l_arm.audit == {
        "shoulder": reachy_sdk_zeroed.l_arm.shoulder.status,
        "elbow": reachy_sdk_zeroed.l_arm.elbow.status,
        "wrist": reachy_sdk_zeroed.l_arm.wrist.status,
        "gripper": None,
    }
    assert reachy_sdk_zeroed.head.audit == {"neck": reachy_sdk_zeroed.head.neck.status, "l_antenna": None, "r_antenna": None}

    assert reachy_sdk_zeroed.audit["r_arm"] == reachy_sdk_zeroed.r_arm.audit
    assert reachy_sdk_zeroed.audit["l_arm"] == reachy_sdk_zeroed.l_arm.audit
    assert reachy_sdk_zeroed.audit["head"] == reachy_sdk_zeroed.head.audit
