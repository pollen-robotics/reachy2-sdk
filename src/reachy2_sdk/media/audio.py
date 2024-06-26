"""Reachy Audio module.

Handles all specific method related to audio especially:
- playing sounds
- recording sounds
- text-to-speech
"""
from typing import List

import grpc
from google.protobuf.empty_pb2 import Empty
from reachy2_sdk_api.sound_pb2 import (
    RecordingRequest,
    SoundId,
    SoundRequest,
    TextRequest,
    VolumeRequest,
)
from reachy2_sdk_api.sound_pb2_grpc import SoundServiceStub


class Audio:
    """Audio class used for microphone and speakers.

    It exposes functions to:
    - play / stop sounds with defined speakers,
    - test the speakers,
    - record tracks with the stereo microphones,
    - synthezise audio from text (text-to-speech)
    """

    def __init__(self, host: str, port: int) -> None:
        """Set up audio module, along with microphone and speakers."""
        self._grpc_audio_channel = grpc.insecure_channel(f"{host}:{port}")

        self._audio_stub = SoundServiceStub(self._grpc_audio_channel)
        self._setup_microphones()
        self._setup_speakers()

    def _setup_microphones(self) -> None:
        """Internal fonction to set up the microphone."""
        micro_info = self._audio_stub.GetAllMicrophone(Empty())
        self._microphone_id = micro_info.microphone_info[0].id

    def _setup_speakers(self) -> None:
        """Internal fonction to set up the speakers."""
        speaker_info = self._audio_stub.GetAllSpeaker(Empty())
        self._speaker_id = speaker_info.speaker_info[0].id

    def testing(self) -> None:
        """Play a sound to test the speaker."""
        self._audio_stub.TestSpeaker(self._speaker_id)

    def get_sounds_list(self) -> List[str]:
        """Get the list of .wav and .ogg files from the robot sound folder."""
        sounds = self._audio_stub.GetSoundsList(Empty())
        return [soundId.id for soundId in sounds.sounds]

    def play(self, sound_name: str) -> None:
        """Play a .wav or .ogg file from the list of available sounds"""
        available_sounds = self.get_sounds_list()
        if sound_name not in available_sounds:
            raise ValueError(f"Sound to play not available! Sounds available are {available_sounds}")
        self._audio_stub.PlaySound(SoundRequest(speaker=self._speaker_id, sound=SoundId(id=sound_name)))

    def stop(self) -> None:
        """Stop playing current file"""
        self._audio_stub.StopSound(self._speaker_id)

    def start_recording(self, sound_name: str) -> None:
        """Start recording an .ogg file. No need to provide the .ogg extension to sound_name argument."""
        ack = self._audio_stub.StartRecording(RecordingRequest(micro=self._microphone_id, recording_id=SoundId(id=sound_name)))
        if ack.ack.success is False:
            raise RuntimeError("Failed to starting recording.")

    def stop_recording(self) -> None:
        """Stop the recording, and save the .ogg file"""
        self._audio_stub.StopRecording(self._microphone_id)

    def set_audio_volume(self, volume: float) -> None:
        """Set speakers volume from 0 to 1.0"""
        if not 0 <= volume <= 1:
            raise ValueError(f"Volume should be between 0 and 1, got {volume}")
        self._audio_stub.ChangeVolume(VolumeRequest(id=self._speaker_id, volume=volume))

    def say_text(self, text: str) -> None:
        """Synthezise and play the given english text"""
        self._audio_stub.SayText(TextRequest(text=text))
