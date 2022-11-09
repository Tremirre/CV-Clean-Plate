import numpy as np


def get_median_image(images: list[np.ndarray]) -> np.ndarray:
    """
    Get the median image from a list of images

    :param images: list/array of images
    :return: median image
    """
    return np.median(images, axis=0).astype(np.uint8)


def get_mean_pixel(images: list[np.ndarray]) -> np.ndarray:
    """
    From a list of images, get the mean pixel (as uint8)

    :param images: list/array of images
    :return: mean pixel
    """
    return np.mean(images, axis=(0, 1, 2)).astype(np.uint8)


def get_normalized_distance_to_vector(
    image: np.ndarray, pixel: np.ndarray
) -> np.ndarray:
    """
    Get the normalized distance of each pixel in an image to a vector (pixel).
    Normalization occurs by dividing by the maximal distance value in the matrix.

    :param image: image on which the distance is calculated
    :param pixel: pixel to which the distance is calculated
    :return: array of distances in range [0, 1]
    """
    distances = np.linalg.norm(image.astype(np.float32) - pixel, axis=2)
    return distances / np.max(distances)


def create_image_histogram(image: np.ndarray) -> np.ndarray:
    """
    Cretes histogram of each color channel of an image.

    :param image: image for which histogram is to be computed
    :return: histogram of each color channel
    """
    histogram = np.zeros((256, 3), dtype=np.uint32)
    for i in range(3):
        histogram[:, i] = np.bincount(image[:, :, i].flatten(), minlength=256)
    return histogram
