import sys
import time
from reachy2_sdk import ReachySDK

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from numpy.typing import NDArray
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QSlider, QVBoxLayout, QWidget

from scipy.spatial.transform import Rotation


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Matplotlib Figure
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.layout.addWidget(self.canvas)

        # Sliders
        self.azimut_slider = self.create_slider("azimut")
        self.inclinaison_slider = self.create_slider("inclinaison")
        self.torsion_slider = self.create_slider("torsion")

        print("Trying to connect on localhost Reachy...")
        time.sleep(1.0)
        self.reachy = ReachySDK(host="localhost")

        time.sleep(1.0)
        if self.reachy._grpc_status == "disconnected":
            print("Failed to connect to Reachy, exiting...")
            return

        self.reachy.turn_on()
        print("Putting each joint at 0 degrees angle")
        time.sleep(0.5)
        for joint in self.reachy.joints.values():
            joint.goal_position = 0
        time.sleep(1.0)

        self.update_plot()

    def create_slider(self, name, min_val=-360, max_val=360):
        # Créer un layout horizontal pour le slider et ses labels
        slider_layout = QHBoxLayout()

        # Label pour le nom du slider
        name_label = QLabel(name)
        slider_layout.addWidget(name_label)

        # Créer le slider
        slider = QSlider(self.central_widget)
        slider.setOrientation(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(0)
        slider.valueChanged.connect(self.update_plot)
        slider_layout.addWidget(slider)

        # Label pour la valeur du slider
        value_label = QLabel("0")
        slider.valueChanged.connect(lambda value, label=value_label: label.setText(str(value)))
        slider_layout.addWidget(value_label)

        # Ajouter le layout du slider au layout principal
        self.layout.addLayout(slider_layout)

        return slider

    def update_plot(self):
        azimut = self.azimut_slider.value()
        inclinaison = self.inclinaison_slider.value()
        torsion = self.torsion_slider.value()
        # This is the natural convention used on the wrist
        R = Rotation.from_euler("xyz", [azimut, inclinaison, torsion], degrees=True)

        # Going to a rotation matrix and then back to euler angles in the zyz convention where we can limit the inclination
        rotation = Rotation.from_euler("xyz", [azimut, inclinaison, torsion], degrees=True)
        arm_joints = rotation.as_euler("zyz", degrees=True)
        # Limiting the inclination
        arm_joints[1] = min(45, max(-45, arm_joints[1]))
        # Going back to the rotation matrix
        rotation = Rotation.from_euler("zyz", arm_joints, degrees=True)
        # Going back to the euler angles in the xyz convention
        arm_joints = rotation.as_euler("xyz", degrees=True)

        r_arm_joints = [180, 0, 0, 0, arm_joints[0], arm_joints[1], arm_joints[2]]
        for joint, goal_pos in zip(self.reachy.r_arm.joints.values(), r_arm_joints):
            joint.goal_position = goal_pos

        self.ax.clear()
        self.plot_3d_coordinate_frame(R.as_matrix(), alpha=True)
        self.plot_3d_coordinate_frame(rotation.as_matrix(), alpha=True)
        self.canvas.draw()

    def plot_3d_coordinate_frame(self, R, alpha=False):
        i, j, k = np.eye(3)
        i_rotated, j_rotated, k_rotated = R @ np.eye(3)

        self.ax.quiver(0, 0, 0, i[0], i[1], i[2], color="k", length=1.0)
        self.ax.quiver(0, 0, 0, j[0], j[1], j[2], color="k", length=1.0)
        self.ax.quiver(0, 0, 0, k[0], k[1], k[2], color="k", length=1.0)

        if alpha:
            self.ax.quiver(
                0,
                0,
                0,
                i_rotated[0],
                i_rotated[1],
                i_rotated[2],
                color="r",
                length=1.0,
                alpha=0.5,
            )
            self.ax.quiver(
                0,
                0,
                0,
                j_rotated[0],
                j_rotated[1],
                j_rotated[2],
                color="g",
                length=1.0,
                alpha=0.5,
            )
            self.ax.quiver(
                0,
                0,
                0,
                k_rotated[0],
                k_rotated[1],
                k_rotated[2],
                color="b",
                length=1.0,
                alpha=0.5,
            )

        else:
            self.ax.quiver(0, 0, 0, i_rotated[0], i_rotated[1], i_rotated[2], color="r", length=1.0)
            self.ax.quiver(0, 0, 0, j_rotated[0], j_rotated[1], j_rotated[2], color="g", length=1.0)
            self.ax.quiver(0, 0, 0, k_rotated[0], k_rotated[1], k_rotated[2], color="b", length=1.0)

        self.ax.set_xlim([-1.5, 1.5])
        self.ax.set_ylim([-1.5, 1.5])
        self.ax.set_zlim([-1.5, 1.5])
        self.ax.set_xlabel("X Axis")
        self.ax.set_ylabel("Y Axis")
        self.ax.set_zlabel("Z Axis")


def build_pose_matrix(x: float, y: float, z: float):
    # The effector is always at the same orientation in the world frame
    return np.array(
        [
            [0, 0, -1, x],
            [0, 1, 0, y],
            [1, 0, 0, z],
            [0, 0, 0, 1],
        ]
    )


def get_homogeneous_matrix_msg_from_euler(
    position: tuple = (0, 0, 0),  # (x, y, z)
    euler_angles: tuple = (0, 0, 0),  # (roll, pitch, yaw)
    degrees: bool = False,
):
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = Rotation.from_euler("xyz", euler_angles, degrees=degrees).as_matrix()
    homogeneous_matrix[:3, 3] = position
    return homogeneous_matrix


def angle_diff(a: float, b: float) -> float:
    """Returns the smallest distance between 2 angles"""
    d = a - b
    d = ((d + np.pi) % (2 * np.pi)) - np.pi
    return d


if __name__ == "__main__":
    # Start the application
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
