from reachy2_sdk import ReachySDK
from reachy2_sdk_api.goto_pb2 import GoalStatus

reachy = ReachySDK("localhost")

reachy.mobile_base.goto(0.0,0.0,-45,timeout=10.0)
