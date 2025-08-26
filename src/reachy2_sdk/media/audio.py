"""Reachy Audio module.

Enable access to the microphones and speaker.
"""

import logging
import os
from io import BytesIO
from typing import Generator, List

import grpc
from google.protobuf.empty_pb2 import Empty
from reachy2_sdk_api.audio_pb2 import AudioFile, AudioFileRequest
from reachy2_sdk_api.audio_pb2_grpc import AudioServiceStub


class Audio:
    """Audio class manages the microhpones and speaker on the robot.

    It allows to play audio files, and record audio. Please note that the audio files are stored in a
    temporary folder on the robot and are deleted when the robot is turned off.
    """

    def __init__(self, host: str, port: int) -> None:
        """Set up the audio module.

        This initializes the gRPC channel for communicating with the audio service.

        Args:
            host: The host address for the gRPC service.
            port: The port number for the gRPC service.
        """
        self._logger = logging.getLogger(__name__)
        self._grpc_connected = False
        self._host = host
        self._port = port
        self.connect()

    def connect(self) -> None:
        """Connect to the audio service.

        This method establishes a gRPC channel to the audio service.
        """
        try:
            self._grpc_audio_channel = grpc.insecure_channel(f"{self._host}:{self._port}")
            self._audio_stub = AudioServiceStub(self._grpc_audio_channel)
            self._grpc_connected = True
            self._logger.debug("Audio gRPC channel established.")
        except Exception as e:
            self._grpc_connected = False
            raise ConnectionError(f"Failed to connect to audio service: {e}")

    def _validate_extension(self, path: str, valid_extensions: List[str]) -> bool:
        """Validate the file type and return the file name if valid.

        Args:
            path: The path to the audio file.

        Returns:
            The file name if the file type is valid, otherwise None.
        """
        return path.lower().endswith(tuple(valid_extensions))

    def upload_audio_file(self, path: str) -> bool:
        """Upload an audio file to the robot.

        This method uploads an audio file to the robot. The audio file is stored in a temporary folder on the robot
        and is deleted when the robot is turned off.

        Args:
            path: The path to the audio file to upload.
        """
        if not self._grpc_connected:
            self._logger.error("Not connected to the audio service.")
            return False

        if not self._validate_extension(path, [".wav", ".ogg", ".mp3"]):
            self._logger.error("Invalid file type. Supported file types are .wav, .ogg, .mp3")
            return False

        if not os.path.exists(path):
            self._logger.error(f"File does not exist: {path}")
            return False

        def generate_requests(file_path: str) -> Generator[AudioFileRequest, None, None]:
            yield AudioFileRequest(info=AudioFile(path=os.path.basename(file_path)))

            # 64KiB seems to be the size limit. see https://github.com/grpc/grpc.github.io/issues/371
            CHUNK_SIZE = 64 * 1024  # 64 KB

            with open(file_path, "rb") as file:
                while True:
                    chunk = file.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    yield AudioFileRequest(chunk_data=chunk)

        response = self._audio_stub.UploadAudioFile(generate_requests(path))
        if response.success.value:
            return True
        else:
            self._logger.error(f"Failed to upload file: {response.error}")
            return False

    def download_audio_file(self, name: str, path: str) -> bool:
        """Download an audio file from the robot.

        Args:
            name: The name of the audio file to download.
            path: The folder to save the downloaded audio file.
        """
        if not self._grpc_connected:
            self._logger.error("Not connected to the audio service.")
            return False

        response_iterator = self._audio_stub.DownloadAudioFile(AudioFile(path=name))

        file_name = None
        buffer = BytesIO()

        for response in response_iterator:
            if response.WhichOneof("data") == "info":
                file_name = response.info.path
            elif response.WhichOneof("data") == "chunk_data":
                buffer.write(response.chunk_data)

        if file_name:
            file_path = os.path.join(path, file_name)
            with open(file_path, "wb") as file:
                file.write(buffer.getvalue())
            return os.path.exists(file_path)
        else:
            return False

    def get_audio_files(self) -> List[str]:
        """Get audio files from the robot.

        This method retrieves the list of audio files stored on the robot.
        """
        if not self._grpc_connected:
            self._logger.error("Not connected to the audio service.")
            return []

        files = self._audio_stub.GetAudioFiles(request=Empty())

        return [file.path for file in files.files]

    def remove_audio_file(self, name: str) -> bool:
        """Remove an audio file from the robot.

        This method removes an audio file from the robot.

        Args:
            name: The name of the audio file to remove.
        """
        if not self._grpc_connected:
            self._logger.error("Not connected to the audio service.")
            return False

        response = self._audio_stub.RemoveAudioFile(request=AudioFile(path=name))
        if response.success.value:
            return True
        else:
            self._logger.error(f"Failed to remove file: {response.error}")
            return False

    def play_audio_file(self, name: str) -> None:
        """Play an audio file on the robot.

        This method plays an audio file on the robot.

        Args:
            name: The name of the audio file to play.
        """
        if not self._grpc_connected:
            self._logger.error("Not connected to the audio service.")
            return

        self._audio_stub.PlayAudioFile(request=AudioFile(path=name))

    def stop_playing(self) -> None:
        """Stop playing audio on the robot.

        This method stops the audio that is currently playing on the robot.
        """
        if not self._grpc_connected:
            self._logger.error("Not connected to the audio service.")
            return

        self._audio_stub.StopPlaying(Empty())

    def record_audio(self, name: str, duration_secs: float) -> bool:
        """Record audio on the robot.

        This method records audio on the robot.

        Args:
            name: name of the audio file. The extension defines the encoding. Ony ogg is supported.
            duration_secs: duration of the recording in seconds.
        """
        if not self._grpc_connected:
            self._logger.error("Not connected to the audio service.")
            return False

        if not self._validate_extension(name, [".ogg"]):
            self._logger.error("Invalid file type. Supported file type is .ogg")
            return False

        self._audio_stub.RecordAudioFile(request=AudioFile(path=name, duration=duration_secs))
        return True

    def stop_recording(self) -> None:
        """Stop recording audio on the robot.

        This method stops the audio recording on the robot.
        """
        if not self._grpc_connected:
            self._logger.error("Not connected to the audio service.")
            return

        self._audio_stub.StopRecording(Empty())

    def disconnect(self) -> None:
        """Disconnect the audio service.

        This method closes the gRPC channel to the audio service.
        """
        if self._grpc_connected:
            self._grpc_audio_channel.close()
            self._grpc_audio_channel = None
            self._grpc_connected = False
            self._logger.debug("Audio gRPC channel closed.")
