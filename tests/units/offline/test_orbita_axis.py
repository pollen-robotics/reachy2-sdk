import numpy as np
import pytest
from google.protobuf.wrappers_pb2 import FloatValue

from reachy2_sdk.orbita.orbita_axis import OrbitaAxis


@pytest.mark.offline
def test_class() -> None:
    speed = FloatValue(value=1)
    load = FloatValue(value=2)
    state = {"present_speed": speed, "present_load": load}
    axis = OrbitaAxis(initial_state=state)
    assert axis.present_speed == np.rad2deg(1)
    assert axis.present_load == 2
