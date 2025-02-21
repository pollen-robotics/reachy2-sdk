import pytest

from reachy2_sdk.utils import custom_dict


@pytest.mark.offline
def test_custom_dict() -> None:
    cd = custom_dict.CustomDict({"1": "2", "3": "4"})
    assert str(cd) == "{'1': 2,\n'3': 4}"
