import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Any

from reachy2_sdk import ReachySDK
from reachy2_sdk.utils.utils import get_pose_matrix
from skimage.morphology import skeletonize

import time


def get_oriented_pose_matrix(
    point: Tuple[int, int],
    origin: List[float],
    orientation: str,
    pen_up: bool = False,
) -> Any:
    """Go to a specific pose with the arm."""
    if orientation == "horizontal":
        x = origin[0] + point[0] * 0.0002
        y = origin[1] - point[1] * 0.0002
        z = origin[2]
        roll = 0.0
        pitch = -90.0
        yaw = ((y + 0.6) * 80) / 0.7 - 20
        rotation = [roll, pitch, yaw]
        if pen_up:
            return get_pose_matrix([x, y, z + 0.02], rotation)
        return get_pose_matrix([x, y, z], rotation)
    elif orientation == "vertical":
        x = origin[0]
        y = origin[1] - point[1] * 0.0002
        z = origin[2] + point[0] * 0.0002
        roll = ((y + 0.6) * 80) / 0.7 - 20
        pitch = -180.0
        yaw = 0.0
        rotation = [roll, pitch, yaw]
        if pen_up:
            return get_pose_matrix([x - 0.02, y, z], rotation)
        return get_pose_matrix([x, y, z], rotation)
    else:
        raise ValueError("Invalid orientation")


def extract_trajectories(image_path: str) -> List[List[Tuple[int, int]]]:
    # Charger l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Appliquer un seuillage binaire
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    # Afficher l'image binaire intermédiaire
    cv2.imshow("Binary Image", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Appliquer une opération morphologique pour connecter les segments discontinus
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Réduction à la structure squelettique
    skeleton = skeletonize(binary // 255).astype(np.uint8) * 255
    
    # Afficher l'image squelettique
    cv2.imshow("Skeleton Image", skeleton)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    ## METHODE 1

    # Trouver les contours
    # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours), "contours trouvés")

    # # Extraire et lisser les trajectoires
    # trajectories = []
    # for contour in contours:
    #     trajectory = [(point[0][0], point[0][1]) for point in contour]
    #     smoothed_trajectory = smooth_trajectory(trajectory, shape=image.shape)
    #     trajectories.append(smoothed_trajectory)


    ## METHODE 2 
    trajectories = trace_skeleton(skeleton)

    # Filtrage des trajectoires pour simplifier les lignes droites
    filtered_trajectories = filter_straight_segments(trajectories)

    return filtered_trajectories


def smooth_trajectory(trajectory: List[Tuple[int, int]], shape: Tuple[int, int], epsilon: float = 5.0) -> List[Tuple[int, int]]:
    """Lisse une trajectoire en réduisant le nombre de points avec l'algorithme de Douglas-Peucker."""
    if len(trajectory) < 3:
        return trajectory

    contour_array = np.array(trajectory, dtype=np.float32)
    smoothed_contour = cv2.approxPolyDP(contour_array, epsilon, True)
    # return [(point[0][0], point[0][1]) for point in smoothed_contour]
    return [(shape[0] - point[0][1], point[0][0]) for point in smoothed_contour]

def trace_skeleton(skeleton: Any) -> List[List[Tuple[int, int]]]:
    """
    Génère des trajectoires à partir du squelette en suivant les lignes.
    """
    visited = np.zeros_like(skeleton)
    trajectories = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def is_valid(x, y):
        return 0 <= x < skeleton.shape[0] and 0 <= y < skeleton.shape[1] and skeleton[x, y] == 255 and visited[x, y] == 0

    def follow_line(x, y):
        trajectory = [(y, x)]
        visited[x, y] = 1
        while True:
            found = False
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if is_valid(nx, ny):
                    trajectory.append((ny, nx))
                    visited[nx, ny] = 1
                    x, y = nx, ny
                    found = True
                    break
            if not found:
                break
        return trajectory

    for i in range(skeleton.shape[0]):
        for j in range(skeleton.shape[1]):
            if skeleton[i, j] == 255 and visited[i, j] == 0:
                trajectory = follow_line(i, j)
                if len(trajectory) > 1:  # Éviter les points isolés
                    trajectories.append(trajectory)

    return trajectories


def filter_straight_segments(trajectories: List[List[Tuple[int, int]]], tolerance: float = 2.0) -> List[List[Tuple[int, int]]]:
    """
    Filtre les trajectoires pour ne garder que les segments de ligne droite.
    Utilise l'algorithme de régression linéaire pour détecter les droites.
    """
    filtered_trajectories = []
    for trajectory in trajectories:
        if len(trajectory) < 3:  # Pas besoin de filtrer si moins de 3 points
            filtered_trajectories.append(trajectory)
            continue

        # Ajuster une droite avec régression linéaire
        x, y = zip(*trajectory)
        x = np.array(x)
        y = np.array(y)
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]  # Pente et ordonnée à l'origine

        # Calculer la distance de chaque point à la droite ajustée
        distances = np.abs(y - (m * x + c)) / np.sqrt(m**2 + 1)

        # Si toutes les distances sont en dessous du seuil de tolérance, garder seulement 2 points
        if np.all(distances <= tolerance):
            filtered_trajectories.append([(x[0], y[0]), (x[-1], y[-1])])
        else:
            filtered_trajectories.append(trajectory)
    return filtered_trajectories


def plot_trajectories(trajectories: List[List[Tuple[int, int]]]) -> None:
    plt.figure(figsize=(8, 8))
    for trajectory in trajectories:
        x, y = zip(*trajectory)
        plt.plot(x, y, marker="o")
    plt.gca().invert_yaxis()
    plt.show()


def draw_trajectories(
    reachy: ReachySDK, trajectories: List[List[Tuple[int, int]]], sheet_origin: List[float], orientation: str
) -> None:
    for trajectory in trajectories:
        for point in range(len(trajectory)):
            if point == 0:
                reachy.r_arm.goto(
                    get_oriented_pose_matrix(trajectory[point], sheet_origin, orientation=orientation, pen_up=True),
                    duration=1.0,
                )
            reachy.r_arm.goto(
                get_oriented_pose_matrix(trajectory[point], sheet_origin, orientation=orientation),
                interpolation_space="cartesian_space",
                duration=1.0,
            )
        reachy.r_arm.goto(
            get_oriented_pose_matrix(trajectory[0], sheet_origin, orientation=orientation),
            interpolation_space="cartesian_space",
            duration=1.0,
        )
        reachy.r_arm.goto(
            get_oriented_pose_matrix(trajectory[0], sheet_origin, orientation=orientation, pen_up=True),
            interpolation_space="cartesian_space",
            duration=1.0,
        )


if __name__ == "__main__":
    print("Reachy SDK example: drawing")

    reachy = ReachySDK(host="localhost")

    if not reachy.is_connected:
        exit("Reachy is not connected.")

    print("Turning on Reachy")
    reachy.turn_on()
    reachy.goto_posture()

    time.sleep(0.2)

    print("Set to Elbow 120 pose ...")
    r_arm_120 = reachy.r_arm.goto([35, -15, -15, -120, 0, 0, 0], wait=True)

    image_path = "House.jpg"  # Remplace par le chemin de ton image
    print(f"Reading image {image_path}")
    trajectories = extract_trajectories(image_path)

    plot_trajectories(trajectories)
    origin = [0.3, 0.1, -0.3]
    draw_trajectories(reachy, trajectories, origin, orientation="horizontal")

    print("Set back to Elbow 90 pose ...")
    head_move = reachy.head.goto_posture("default")
    r_arm_120 = reachy.r_arm.goto([35, -15, -15, -120, 0, 0, 0])
    while not reachy.is_goto_finished(r_arm_120):
        time.sleep(0.1)
