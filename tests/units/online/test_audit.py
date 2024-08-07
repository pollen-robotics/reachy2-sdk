import time

import pytest

from reachy2_sdk.reachy_sdk import ReachySDK


@pytest.mark.online
def test_audit(reachy_sdk_zeroed: ReachySDK) -> None:
    assert reachy_sdk_zeroed.r_arm.shoulder.audit == "Ok"
    assert reachy_sdk_zeroed.r_arm.elbow.audit == "Ok"
    assert reachy_sdk_zeroed.r_arm.wrist.audit == "Ok"
    assert reachy_sdk_zeroed.l_arm.shoulder.audit == "Ok"
    assert reachy_sdk_zeroed.l_arm.elbow.audit == "Ok"
    assert reachy_sdk_zeroed.l_arm.wrist.audit == "Ok"
    assert reachy_sdk_zeroed.head.neck.audit == "Ok"

    assert reachy_sdk_zeroed.r_arm.audit == ["Ok", "Ok", "Ok"]
    assert reachy_sdk_zeroed.l_arm.audit == ["Ok", "Ok", "Ok"]
    assert reachy_sdk_zeroed.head.audit == ["Ok"]

    assert reachy_sdk_zeroed.audit["r_arm"] == reachy_sdk_zeroed.r_arm.audit
    assert reachy_sdk_zeroed.audit["l_arm"] == reachy_sdk_zeroed.l_arm.audit
    assert reachy_sdk_zeroed.audit["head"] == reachy_sdk_zeroed.head.audit
