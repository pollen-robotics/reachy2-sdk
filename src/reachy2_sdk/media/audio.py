"""Reachy Audio module.

Handles all specific method related to audio especially.
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
    """Audio class for managing microphone and speaker functionalities.

    This class provides an interface to control audio features, including:
    - Testing the speakers.
    - Retrieving the list of available sound files.
    - Playing and stopping audio files.
    - Recording audio using stereo microphones.
    - Setting the speaker volume.
    - Synthesizing and playing speech from text (text-to-speech).
    """

    def __init__(self, host: str, port: int) -> None:
        """Initialize the audio module.

        Sets up the gRPC channel to communicate with the audio services, initializes
        the audio stub for managing sounds, and configures the microphone and speaker.

        Args:
            host: The hostname or IP address of the audio service.
            port: The port number on which the audio service is running.
        """
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
        """Retrieve the list of available sound files.

        Gets the list of .wav and .ogg files located in the robot's sound folder.

        Returns:
            A list of sound file names as strings.
        """
        sounds = self._audio_stub.GetSoundsList(Empty())
        return [soundId.id for soundId in sounds.sounds]

    def play(self, sound_name: str) -> None:
        """Play a specified sound file.

        Plays a .wav or .ogg file from the available sounds list.

        Args:
            sound_name: The name of the sound file to play.

        Raises:
            ValueError: If the specified sound is not available in the list of sounds.
        """
        available_sounds = self.get_sounds_list()
        if sound_name not in available_sounds:
            raise ValueError(f"Sound to play not available! Sounds available are {available_sounds}")
        self._audio_stub.PlaySound(SoundRequest(speaker=self._speaker_id, sound=SoundId(id=sound_name)))

    def stop(self) -> None:
        """Stop playing current file."""
        self._audio_stub.StopSound(self._speaker_id)

    def start_recording(self, sound_name: str) -> None:
        """Start recording audio.

        Begins recording an .ogg audio file using the configured microphone.
        The sound_name parameter does not require the .ogg extension.

        Args:
            sound_name: The name to assign to the recorded audio file.

        Raises:
            RuntimeError: If the recording could not be started successfully.
        """
        ack = self._audio_stub.StartRecording(RecordingRequest(micro=self._microphone_id, recording_id=SoundId(id=sound_name)))
        if ack.ack.success is False:
            raise RuntimeError("Failed to starting recording.")

    def stop_recording(self) -> None:
        """Stop the recording, and save the .ogg file."""
        self._audio_stub.StopRecording(self._microphone_id)

    def set_audio_volume(self, volume: float) -> None:
        """Adjust the speaker volume.

        Sets the volume level for the speakers, ranging from 0.0 to 1.0.

        Args:
            volume: The desired volume level, where 0.0 is mute and 1.0 is maximum.

        Raises:
            ValueError: If the volume is not within the range of 0.0 to 1.0.
        """
        if not 0 <= volume <= 1:
            raise ValueError(f"Volume should be between 0 and 1, got {volume}")
        self._audio_stub.ChangeVolume(VolumeRequest(id=self._speaker_id, volume=volume))

    def say_text(self, text: str) -> None:
        """Convert text to speech and play it.

        Uses text-to-speech to synthesize and play the specified English text.

        Args:
            text: The text to convert to speech and play through the speakers.
        """
        self._audio_stub.SayText(TextRequest(text=text))
