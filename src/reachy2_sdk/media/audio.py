"""Reachy Audio module.

Enable access to the microphones and speaker.
"""

import logging
import os
from typing import Generator, List

import grpc
from google.protobuf.empty_pb2 import Empty
from reachy2_sdk_api.audio_pb2 import AudioFile, UploadAudioFileRequest
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
        self._grpc_audio_channel = grpc.insecure_channel(f"{host}:{port}")
        self._host = host

        self._audio_stub = AudioServiceStub(self._grpc_audio_channel)

    def _validate_extension(self, path: str) -> bool:
        """Validate the file type and return the file name if valid.

        Args:
            path: The path to the audio file.

        Returns:
            The file name if the file type is valid, otherwise None.
        """
        valid_extensions = (".wav", ".ogg", ".mp3")
        return path.lower().endswith(valid_extensions)

    def upload_audio_file(self, path: str) -> bool:
        """Upload an audio file to the robot.

        This method uploads an audio file to the robot. The audio file is stored in a temporary folder on the robot
        and is deleted when the robot is turned off.

        Args:
            path: The path to the audio file to upload.
        """

        if not self._validate_extension(path):
            self._logger.error("Invalid file type. Supported file types are .wav, .ogg, .mp3")
            return False

        if not os.path.exists(path):
            self._logger.error(f"File does not exist: {path}")
            return False

        def generate_requests(file_path: str) -> Generator[UploadAudioFileRequest, None, None]:
            yield UploadAudioFileRequest(info=AudioFile(path=os.path.basename(file_path)))

            # 64KiB seems to be the size limit. see https://github.com/grpc/grpc.github.io/issues/371
            CHUNK_SIZE = 64 * 1024  # 64 KB

            with open(file_path, "rb") as file:
                while True:
                    chunk = file.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    yield UploadAudioFileRequest(chunk_data=chunk)

        response = self._audio_stub.UploadAudioFile(generate_requests(path))
        if response.success.value:
            return True
        else:
            self._logger.error(f"Failed to upload file: {response.error}")
            return False

    def get_audio_files(self) -> List[str]:
        """Get audio files from the robot.

        This method retrieves the list of audio files stored on the robot.
        """
        files = self._audio_stub.GetAudioFiles(request=Empty())

        return [file.path for file in files.files]

    def remove_audio_file(self, name: str) -> bool:
        """Remove an audio file from the robot.

        This method removes an audio file from the robot.

        Args:
            name: The name of the audio file to remove.
        """
        response = self._audio_stub.RemoveAudioFile(request=AudioFile(path=name))
        if response.success.value:
            return True
        else:
            self._logger.error(f"Failed to remove file: {response.error}")
            return False
