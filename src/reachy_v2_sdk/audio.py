"""Reachy Arm module.

Handles all specific method to an Arm (left and/or right) especially:
- the forward kinematics
- the inverse kinematics
"""
import grpc

from reachy_sdk_api_v2.sound_pb2_grpc import SoundServiceStub
from reachy_sdk_api_v2.sound_pb2 import (
    VolumeRequest,
    SoundId,
    SoundRequest,
    RecordingRequest,
)
from google.protobuf.empty_pb2 import Empty

from typing import List


class Audio:
    """Arm abstract class used for both left/right arms.

    It exposes the kinematics of the arm:
    - you can access the joints actually used in the kinematic chain,
    - you can compute the forward and inverse kinematics
    """

    def __init__(self, host: str, port: int) -> None:
        """Set up the arm with its kinematics."""
        self._grpc_audio_channel = grpc.insecure_channel(f"{host}:{port}")

        self._audio_stub = SoundServiceStub(self._grpc_audio_channel)
        self._setup_microphones()
        self._setup_speakers()

    def _setup_microphones(self) -> None:
        micro_info = self._audio_stub.GetAllMicrophone(Empty())
        self._microphone_id = micro_info.microphone_info[0].id

    def _setup_speakers(self) -> None:
        speaker_info = self._audio_stub.GetAllSpeaker(Empty())
        self._speaker_id = speaker_info.speaker_info[0].id

    def testing(self) -> None:
        self._audio_stub.TestSpeaker(self._speaker_id)

    def get_sounds_list(self) -> List[str]:
        sounds = self._audio_stub.GetSoundsList(Empty())
        return [soundId.id for soundId in sounds.sounds]

    def play(self, sound_name: str, volume: int = 100) -> None:
        available_sounds = self.get_sounds_list()
        if sound_name not in available_sounds:
            raise ValueError(f"Sound to play not available! Sounds available are {available_sounds}")
        self._audio_stub.PlaySound(SoundRequest(speaker=self._speaker_id, sound=SoundId(id=sound_name, volume=volume)))

    def stop(self) -> None:
        self._audio_stub.StopSound(ComponentId=self._speaker_id)

    def start_recording(self, sound_name: str) -> None:
        self._audio_stub.StartRecording(RecordingRequest(micro=self._microphone_id, recording_id=SoundId(id=sound_name)))

    def stop_recording(self) -> None:
        self._audio_stub.StopRecording(self._microphone_id)

    def set_audio_volume(self, volume: int) -> None:
        self._audio_stub.ChangeVolume(VolumeRequest(id=self._speaker_id, volume=volume))
