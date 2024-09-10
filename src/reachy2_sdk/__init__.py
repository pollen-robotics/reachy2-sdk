"""
# ReachySDK package.

This package provides remote access (via socket) to a Reachy robot.
It automatically handles the synchronization with the robot.
In particular, you can easily get an always up-to-date robot state (joint positions, sensors value).
You can also send joint commands, compute forward or inverse kinematics.

Simply do
```python
from reachy2_sdk.reachy_sdk import ReachySDK
reachy = ReachySDK(host="ip_address")
```

And you're ready to use Reachy!

*Examples and tutorials are available [here](../src/examples/)!*

"""

from .reachy_sdk import ReachySDK  # noqa: F401

__version__ = "1.0.4"
