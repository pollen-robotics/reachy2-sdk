import os
import tempfile
from typing import List

import numpy as np
import pytest
from reachy2_sdk_api.audio_pb2 import AudioFile

from reachy2_sdk.media.audio import Audio
from reachy2_sdk.reachy_sdk import ReachySDK


@pytest.mark.audio
def test_audio(reachy_sdk: ReachySDK) -> None:
    file = ""
    assert not reachy_sdk.audio.upload_audio_file(file)

    file = "badextension.mp4"
    assert not reachy_sdk.audio.upload_audio_file(file)

    file = "doesnotexist.mp3"
    assert not reachy_sdk.audio.upload_audio_file(file)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp_file:
        tmp_file.write(b"\0" * (1024 * 1024))
        tmp_file_path = tmp_file.name

        assert reachy_sdk.audio.upload_audio_file(tmp_file_path)

        files = reachy_sdk.audio.get_audio_files()

        assert os.path.basename(tmp_file_path) in files

    assert not reachy_sdk.audio.remove_audio_file("doesnotexist.mp3")

    assert reachy_sdk.audio.remove_audio_file(os.path.basename(tmp_file_path))


@pytest.mark.audio
def test_recording(reachy_sdk: ReachySDK) -> None:
    file = "badextension.mp4"
    assert not reachy_sdk.audio.record_audio(file, duration_secs=2.0)


@pytest.mark.audio
def test_connect_disconnect(reachy_sdk: ReachySDK) -> None:
    reachy_sdk.audio.disconnect()
    assert reachy_sdk._grpc_connected
    assert not reachy_sdk.audio._grpc_connected
    assert not reachy_sdk.audio.play_audio_file("test.mp3")
    assert not reachy_sdk.audio.stop_playing()
    assert not reachy_sdk.audio.record_audio("test.mp3", duration_secs=2.0)
    assert not reachy_sdk.audio.get_audio_files()
    assert not reachy_sdk.audio.remove_audio_file("test.mp3")

    reachy_sdk.audio.connect()
    assert reachy_sdk.audio._grpc_connected

    reachy_sdk.disconnect()
    assert not reachy_sdk._grpc_connected
    assert not reachy_sdk.audio._grpc_connected
    assert reachy_sdk.audio

    reachy_sdk.audio.connect()
    assert reachy_sdk.audio._grpc_connected
    assert not reachy_sdk._grpc_connected

    reachy_sdk.connect()
    assert reachy_sdk._grpc_connected
    assert reachy_sdk.audio._grpc_connected
