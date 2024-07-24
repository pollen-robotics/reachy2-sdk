from reachy2_sdk import ReachySDK

reachy = ReachySDK("localhost")

if reachy.mobile_base is not None:
    reachy.mobile_base.goto(0.0, 0.0, -45, timeout=10.0)
