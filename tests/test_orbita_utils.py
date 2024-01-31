import pytest

from reachy2_sdk.orbita.utils import _to_internal_position


@pytest.mark.offline
def test_internal_pos() -> None:
    with pytest.raises(TypeError):
        _to_internal_position("2.3")
