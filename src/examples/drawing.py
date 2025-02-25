import time
from typing import Any, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from skimage.morphology import skeletonize

from reachy2_sdk import ReachySDK
from reachy2_sdk.utils.utils import get_pose_matrix


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


def extract_trajectories(image_path: str) -> Tuple[List[List[Tuple[int, int]]], List[bool]]:
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

    # METHODE 1

    # Trouver les contours
    # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours), "contours trouvés")

    # # Extraire et lisser les trajectoires
    # trajectories = []
    # for contour in contours:
    #     trajectory = [(point[0][0], point[0][1]) for point in contour]
    #     smoothed_trajectory = smooth_trajectory(trajectory)
    #     trajectories.append(smoothed_trajectory)

    # METHODE 2
    trajectories = trace_skeleton(skeleton)
    print(len(trajectories), "trajectoires trouvées")

    # ellipses = detect_half_ellipses(trajectories)
    # print("Demi-ellipses détectées :", ellipses)

    # Filtrage des trajectoires pour simplifier les lignes droites
    # filtered_trajectories = filter_straight_segments(trajectories)
    # filtered_trajectories = filter_straight_segments_recursive(trajectories)

    # filtered_trajectories = [smooth_trajectory(trajectory) for trajectory in trajectories]
    # # Vérification des trajectoires fermées
    closed_status = check_closed_trajectories(trajectories)
    # print("Trajectoires fermées :", closed_status)

    # filtered_trajectories = [change_coordinate_system(trajectory, image.shape) for trajectory in filtered_trajectories]

    # TEST ellipses :
    # Détection des demi-ellipses

    return trajectories, closed_status


def change_coordinate_system(trajectory: List[Tuple[int, int]], shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Change le système de coordonnées de l'image à l'origine en bas à gauche."""
    return [(shape[0] - point[1], point[0]) for point in trajectory]


def smooth_trajectory(trajectory: List[Tuple[int, int]], epsilon: float = 5.0) -> List[Tuple[int, int]]:
    """Lisse une trajectoire en réduisant le nombre de points avec l'algorithme de Douglas-Peucker."""
    if len(trajectory) < 3:
        return trajectory

    contour_array = np.array(trajectory, dtype=np.float32)
    smoothed_contour = cv2.approxPolyDP(contour_array, epsilon, True)
    # return [(point[0][0], point[0][1]) for point in smoothed_contour]
    return [(point[0][0], point[0][1]) for point in smoothed_contour]


def check_closed_trajectories(trajectories: List[List[Tuple[int, int]]], tolerance: float = 2.0) -> List[bool]:
    """
    Vérifie si chaque trajectoire est fermée en comparant le premier et le dernier point.
    Retourne une liste de booléens indiquant si chaque trajectoire est fermée.
    """
    closed_status = []
    for trajectory in trajectories:
        if len(trajectory) < 3:
            closed_status.append(False)
            continue
        start = trajectory[0]
        end = trajectory[-1]
        distance = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)
        closed_status.append(distance <= tolerance)  # Tolérance pour considérer comme fermé
    return closed_status


def trace_skeleton(skeleton: Any) -> List[List[Tuple[int, int]]]:
    """
    Génère des trajectoires à partir du squelette en suivant les lignes.
    """
    visited = np.zeros_like(skeleton)
    trajectories = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def is_valid(x, y):
        return 0 <= x < skeleton.shape[0] and 0 <= y < skeleton.shape[1] and skeleton[x, y] == 255 and visited[x, y] == 0

    def valid_but_already_visited(x, y):
        return 0 <= x < skeleton.shape[0] and 0 <= y < skeleton.shape[1] and skeleton[x, y] == 255 and visited[x, y] == 1

    def merge_trajectories(trajectories, index_traj1, index_traj2):
        if trajectories[index_traj1][-1] == trajectories[index_traj2][0]:
            return trajectories[index_traj1] + trajectories[index_traj2][1:], index_traj1
        elif trajectories[index_traj1][0] == trajectories[index_traj2][-1]:
            return trajectories[index_traj2] + trajectories[index_traj1][1:], index_traj2
        return None, None

    def follow_line(x, y):
        # print(f"follow_line: x: {x}, y: {y}")
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
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if valid_but_already_visited(nx, ny) and not ((ny, nx) in trajectory):
                        trajectory.append((ny, nx))
                        break
                break
        return trajectory

    for i in range(skeleton.shape[0]):
        for j in range(skeleton.shape[1]):
            if skeleton[i, j] == 255 and visited[i, j] == 0:
                trajectory = follow_line(i, j)
                if len(trajectory) > 1:  # Éviter les points isolés
                    trajectories.append(trajectory)
    print(f"Nombre de trajectoires: {len(trajectories)}")

    while True:
        modif = False
        for traj1_ind in range(len(trajectories)):
            for traj2_ind in range(len(trajectories)):
                if traj1_ind != traj2_ind:
                    merged, ind = merge_trajectories(trajectories, traj1_ind, traj2_ind)
                    if merged:
                        modif = True
                        trajectories[ind] = merged
                        if ind == traj1_ind:
                            del trajectories[traj2_ind]
                        else:
                            del trajectories[traj1_ind]
                        break
            break
        if not modif:
            break
    print(f"Nombre de trajectoires 2 : {len(trajectories)}")

    return trajectories


def detect_half_ellipses(trajectories, tolerance=2.0, min_points=5):
    """
    Détecte les demi-ellipses dans les trajectoires en utilisant un ajustement par courbe.
    Retourne une liste de demi-ellipses avec leurs paramètres.
    """
    ellipses = []
    for trajectory in trajectories:
        # Vérifier si la trajectoire a suffisamment de points pour l'ajustement
        if len(trajectory) < min_points:
            continue
        x, y = zip(*trajectory)
        x = np.array(x)
        y = np.array(y)

        # Ajuster une demi-ellipse par régression non-linéaire
        def ellipse_model(x, a, b, h, k):
            return k + b * np.sqrt(1 - ((x - h) / a) ** 2)

        try:
            params, _ = curve_fit(ellipse_model, x, y, maxfev=10000)
            a, b, h, k = params

            # Calcul des résidus pour vérifier l'ajustement
            y_fit = ellipse_model(x, a, b, h, k)
            residuals = np.abs(y - y_fit)

            if np.all(residuals <= tolerance):
                ellipses.append(
                    {
                        "a": a,  # Grand rayon (horizontal)
                        "b": b,  # Petit rayon (vertical)
                        "h": h,  # Centre en X
                        "k": k,  # Centre en Y
                        "trajectory": trajectory,
                    }
                )
        except RuntimeError:
            continue

    return ellipses


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


def filter_straight_segments_recursive(
    trajectories: List[List[Tuple[int, int]]], tolerance: float = 2.0
) -> List[List[Tuple[int, int]]]:
    """
    Filtre récursivement les trajectoires pour ne garder que les segments de ligne droite.
    Segmente les portions de trajectoire en lignes droites plus courtes si nécessaire.
    """

    def recursive_filter(trajectory: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(trajectory) < 3:
            return trajectory

        # Ajuster une droite sur tous les points du segment
        x, y = zip(*trajectory)
        x = np.array(x)
        y = np.array(y)

        # Vérifier si tous les x sont (quasi) égaux pour détecter les lignes verticales
        if np.max(np.abs(x - np.mean(x))) <= tolerance:
            return [(x[0], y[0]), (x[-1], y[-1])]

        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]  # Pente et ordonnée à l'origine

        # Calculer la distance de chaque point à la droite ajustée
        distances = np.abs(y - (m * x + c)) / np.sqrt(m**2 + 1)

        # Si toutes les distances sont en dessous du seuil, simplifier à 2 points
        if np.all(distances <= tolerance):
            return [(x[0], y[0]), (x[-1], y[-1])]
        else:
            # Diviser la trajectoire en deux et traiter récursivement
            mid = len(trajectory) // 2
            left = recursive_filter(trajectory[: mid + 1])
            right = recursive_filter(trajectory[mid:])
            return left[:-1] + right  # Éviter la duplication du point intermédiaire

    filtered_trajectories = [recursive_filter(trajectory) for trajectory in trajectories]
    return filtered_trajectories


def plot_trajectories(trajectories: List[List[Tuple[int, int]]]) -> None:
    plt.figure(figsize=(8, 8))
    for trajectory in trajectories:
        x, y = zip(*trajectory)
        plt.plot(x, y, marker="o")
    plt.gca().invert_yaxis()
    plt.show()


def draw_trajectories(
    reachy: ReachySDK,
    trajectories: List[List[Tuple[int, int]]],
    closed_status: List[bool],
    sheet_origin: List[float],
    orientation: str,
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
        if closed_status[trajectories.index(trajectory)]:
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
        else:
            reachy.r_arm.goto(
                get_oriented_pose_matrix(trajectory[-1], sheet_origin, orientation=orientation, pen_up=True),
                interpolation_space="cartesian_space",
                duration=1.0,
            )


if __name__ == "__main__":
    print("Reachy SDK example: drawing")

    reachy = ReachySDK(host="localhost")
    image_path = "Dessins2.jpg"  # Remplace par le chemin de ton image

    if not reachy.is_connected:
        exit("Reachy is not connected.")

    print("Turning on Reachy")
    reachy.turn_on()
    reachy.goto_posture()

    time.sleep(0.2)

    print("Set to Elbow 120 pose ...")
    r_arm_120 = reachy.r_arm.goto([35, -15, -15, -120, 0, 0, 0], wait=True)

    print(f"Reading image {image_path}")
    trajectories, closed_status = extract_trajectories(image_path)

    plot_trajectories(trajectories)
    origin = [0.3, 0.1, -0.3]
    # draw_trajectories(reachy, trajectories, closed_status, origin, orientation="horizontal")

    print("Set back to Elbow 90 pose ...")
    head_move = reachy.head.goto_posture("default")
    r_arm_120 = reachy.r_arm.goto([35, -15, -15, -120, 0, 0, 0])
    while not reachy.is_goto_finished(r_arm_120):
        time.sleep(0.1)
