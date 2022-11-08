import numpy as np


def get_moore_neighbors(image: np.ndarray, x: int, y: int) -> np.ndarray:
    return image[
        max(0, x - 1) : min(image.shape[0], x + 2),
        max(0, y - 1) : min(image.shape[1], y + 2),
    ]


MOORE_MASK = np.array(
    [[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]
)


def get_fitting_moore_mask(
    image_shape: tuple[int, int], pixel_coords: tuple[int, int]
) -> np.ndarray:
    moore_mask = MOORE_MASK + np.array(pixel_coords)
    return moore_mask[
        (np.min(moore_mask, axis=1) >= 0)
        & (moore_mask[:, 0] < image_shape[0])
        & (moore_mask[:, 1] < image_shape[1])
    ]


def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def cosine_distance_mat_vec(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    return 1 - np.dot(M, v) / (np.linalg.norm(M, axis=2) * np.linalg.norm(v))


def explore_pixel(
    image: np.ndarray,
    labels: np.ndarray,
    entry_pixel_coords: tuple[int, int],
    threshold: float,
) -> None:
    pixels_to_explore = [entry_pixel_coords]
    while pixels_to_explore:
        pixel_coords = pixels_to_explore.pop()
        pixel_neighbors = get_moore_neighbors(image, *pixel_coords)
        pixel = image[pixel_coords].astype(np.float32)
        distances = cosine_distance_mat_vec(pixel_neighbors, pixel).flatten()
        neighbors_cords = get_fitting_moore_mask(image.shape, pixel_coords)
        for neighbor_cord, distance in zip(neighbors_cords, distances):
            neighbor_cord_as_tuple = tuple(neighbor_cord)
            if distance < threshold and labels[neighbor_cord_as_tuple] == 0:
                labels[neighbor_cord_as_tuple] = labels[pixel_coords]
                pixels_to_explore.append(neighbor_cord_as_tuple)


def labelize(image: np.ndarray, threshold: float) -> np.ndarray:
    labels = np.zeros(image.shape[:2])
    last_free_label = 1
    while True:
        unlabeled_pixels = np.argwhere(labels == 0)
        if not unlabeled_pixels.size:
            break
        unlabeled_pixel = tuple(unlabeled_pixels[0])
        labels[unlabeled_pixel] = last_free_label
        explore_pixel(image, labels, unlabeled_pixel, threshold)
        last_free_label += 1
    return labels
