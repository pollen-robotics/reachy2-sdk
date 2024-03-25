import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from numpy.typing import NDArray
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QSlider, QVBoxLayout, QWidget
from reachy2_sdk import ReachySDK
from scipy.spatial.transform import Rotation

DEGREES = True


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
        self.azimut_slider = self.create_slider("roll")
        self.inclinaison_slider = self.create_slider("pitch")
        self.torsion_slider = self.create_slider("yaw")

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

    def restrict_rotation_to_cone(self, euler_angles, cone_axis, max_angle, degrees=True):
        """
        Restrict a rotation given in Euler angles to a cone.

        :param euler_angles: The rotation in Euler angles (XYZ) as a numpy array.
        :param cone_axis: The axis of the cone as a unit vector.
        :param max_angle: The half-angle of the cone in radians.

        :return: A numpy array of the adjusted Euler angles.
        """

        # Convert Euler angles to a quaternion
        rotation = Rotation.from_euler("xyz", euler_angles, degrees=degrees)
        rotation_vector = rotation.as_rotvec()

        # Extract the rotation axis and angle
        rotation_angle = np.linalg.norm(rotation_vector)
        rotation_axis = rotation_vector / rotation_angle if rotation_angle != 0 else rotation_vector

        # Check if the rotation is within the cone
        angle_with_cone_axis = np.arccos(np.clip(np.dot(rotation_axis, cone_axis), -1.0, 1.0))
        # print(f"Angle with cone axis: {np.degrees(angle_with_cone_axis)}")

        if angle_with_cone_axis > max_angle:
            print("NOT IN THE CONE!")
            # Adjust the rotation to be within the cone
            # Project the rotation axis onto the plane orthogonal to the cone axis
            projected_axis = rotation_axis - np.dot(rotation_axis, cone_axis) * cone_axis
            projected_axis /= np.linalg.norm(projected_axis)

            # Adjust the rotation axis and angle
            adjusted_rotation_axis = np.cos(max_angle) * cone_axis + np.sin(max_angle) * projected_axis
            rotation_vector = adjusted_rotation_axis * rotation_angle

        # Convert the adjusted rotation vector back to Euler angles
        adjusted_rotation = Rotation.from_rotvec(rotation_vector)
        adjusted_euler_angles = adjusted_rotation.as_euler("xyz", degrees=degrees)

        return adjusted_euler_angles

    def update_plot(self):
        azimut = self.azimut_slider.value()
        inclinaison = self.inclinaison_slider.value()
        torsion = self.torsion_slider.value()
        # This is the natural convention used on the wrist
        # adjusted_euler_angles = self.restrict_rotation_to_cone(
        #     [azimut, inclinaison, torsion], [1, 0, 0], np.deg2rad(20), degrees=DEGREES
        # )
        R = Rotation.from_euler("xyz", [azimut, inclinaison, torsion], degrees=DEGREES)
        # Rrestricted = Rotation.from_euler("xyz", adjusted_euler_angles, degrees=DEGREES)
        # RZYZ = Rotation.from_euler("ZYZ", [azimut, inclinaison, torsion], degrees=DEGREES)

        # Going to a rotation matrix and then back to euler angles in the zyz convention where we can limit the inclination
        rotation = Rotation.from_euler("xyz", [azimut, inclinaison, torsion], degrees=DEGREES)
        arm_joints = rotation.as_euler("ZYZ", degrees=DEGREES)
        # Limiting the inclination
        arm_joints[1] = min(45, max(-45, arm_joints[1]))
        # arm_joints[2] = 0
        # Going back to the rotation matrix
        rotation = Rotation.from_euler("ZYZ", arm_joints, degrees=DEGREES)
        # Going back to the euler angles in the xyz convention
        arm_joints = rotation.as_euler("xyz", degrees=DEGREES)
        R2 = Rotation.from_euler("xyz", arm_joints, degrees=DEGREES)

        r_arm_joints = [180, 0, 0, 0, arm_joints[0], arm_joints[1], arm_joints[2]]
        for joint, goal_pos in zip(self.reachy.r_arm.joints.values(), r_arm_joints):
            joint.goal_position = goal_pos

        self.ax.clear()
        self.plot_3d_coordinate_frame(R.as_matrix(), alpha=1.0)
        self.plot_3d_coordinate_frame(R2.as_matrix(), alpha=0.5)
        self.plot_3d_coordinate_frame(np.eye(3), alpha=1.0, length=0.5)
        # self.plot_3d_coordinate_frame(RZYZ.as_matrix(), alpha=1.0, length=2.0)
        # self.plot_3d_coordinate_frame(Rrestricted.as_matrix(), alpha=1.0, length=2.0)

        self.canvas.draw()

    def plot_3d_coordinate_frame(self, R, alpha=0.5, length=1.0):
        # Standard basis vectors
        i = np.array([1, 0, 0])
        j = np.array([0, 1, 0])
        k = np.array([0, 0, 1])

        # Rotate basis vectors
        i_rotated = R @ i
        j_rotated = R @ j
        k_rotated = R @ k

        # Plotting each rotated vector
        # Red for i, Green for j, Blue for k
        self.ax.quiver(
            0,
            0,
            0,
            i_rotated[0],
            i_rotated[1],
            i_rotated[2],
            color="r",
            length=length,
            alpha=alpha,
        )
        self.ax.quiver(
            0,
            0,
            0,
            j_rotated[0],
            j_rotated[1],
            j_rotated[2],
            color="g",
            length=length,
            alpha=alpha,
        )
        self.ax.quiver(
            0,
            0,
            0,
            k_rotated[0],
            k_rotated[1],
            k_rotated[2],
            color="b",
            length=length,
            alpha=alpha,
        )
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
    homogeneous_matrix[:3, :3] = Rotation.from_euler("xyz", euler_angles, degrees=DEGREES).as_matrix()
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
