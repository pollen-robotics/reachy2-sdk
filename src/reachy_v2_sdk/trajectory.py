from pathlib import Path

import time
from threading import Thread
from reachy_v2_sdk import ReachySDK


class TrajectoryManager:
    def __init__(
            self,
            reachy: ReachySDK,
            records_folder: str = str(Path.cwd() / 'records')
            ) -> None:
        self.reachy = reachy
        self._records_folder = records_folder
        self.recorder = Recorder(
            reachy=reachy,
            trajectory_manager=self,
            records_folder=self._records_folder
            )
        self.replayer = Replayer(reachy, self._records_folder)

    def get_records(self):
        pass

class Recorder:
    def __init__(
            self,
            reachy: ReachySDK,
            trajectory_manager: TrajectoryManager,
            records_folder: str = str(Path.cwd() / 'records')
            ) -> None:
        self.reachy = reachy
        self._trajectory_manager = trajectory_manager
        self._records_folder = records_folder
        self.last_record = None

    @property
    def available_records(self):
        return self._available_records

    def _update_available_records(self):
        pass

    def start(self, record_hand: bool = False, sampling_frequency: int = 100) -> None:
        self.__joints_record = []
        self.is_recording = True
        self.__record_thread = Thread(
            target=self.__record,
            args=(
                record_hand,
                sampling_frequency,
            ),
        )
        self.__record_thread.start()

    def __record(self, sampling_frequency: int = 100) -> None:
        recorded_joints = []
        for actuator in self._actuators.values():
            for joint in getattr(actuator, "_joints").values():
                recorded_joints.append(joint)

        while self.is_recording:
            current_point = [joint.present_position for joint in recorded_joints]
            self.__joints_record.append(current_point)
            time.sleep(1 / sampling_frequency)

    def stop(self):
        self.is_recording = False
        self.__record_thread.join()

    def save(self):
        pass


class Replayer:
    def __init__(
        self,
        reachy: ReachySDK,
        trajectory_manager: TrajectoryManager,
        records_folder: str = str(Path.cwd() / 'records')
        ) -> None:
        self.reachy = reachy
        self._trajectory_manager = trajectory_manager
        self._records_folder = records_folder

    def is_playing(self):
        pass

    def load(self):
        pass

    def play(self):
        pass

    def stop(self):
        pass