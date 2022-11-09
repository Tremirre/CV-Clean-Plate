import cv2
import numpy as np

from . import stat


def get_changing_mask(images: list[np.ndarray], tolerance: int = 2) -> np.ndarray:
    """
    Returns a mask of pixels that are changing between images.

    :param images: list of images from which to extract changing mask
    :param tolerance: maximal difference between the two pixel channel values for them to be considered unchanged, defaults to 2
    :return: mask denoting the changing pixels by True
    """
    result = np.zeros(images[0].shape[:2])
    for i, image_a in enumerate(images):
        for _, image_b in enumerate(images[i + 1 :]):
            result += ~np.all(np.isclose(image_a, image_b, atol=tolerance), axis=2)
    return result > 0


def get_overly_red_mask(image: np.ndarray, tolerance: int) -> np.ndarray:
    """
    Returns a mask of pixels that are overly red.

    :param image: image on which mask is to be computed
    :param tolerance: minimal difference between the red channel and other channel values for pixel to be considered overly red
    :return: mask denoting overly red pixels by True
    """
    red_mask = (image[:, :, 2] > (image[:, :, 1] + tolerance)) & (
        image[:, :, 2] > (image[:, :, 0] + tolerance)
    )
    return red_mask


def get_overly_blue_mask(image: np.ndarray, tolerance: int) -> np.ndarray:
    """
    Returns a mask of pixels that are overly blue.

    :param image: image on which mask is to be computed
    :param tolerance: minimal difference between the blue channel and other channel values for pixel to be considered overly blue
    :return: mask denoting overly blue pixels by True
    """
    blue_mask = (image[:, :, 1] > (image[:, :, 2] + tolerance)) & (
        image[:, :, 1] > (image[:, :, 0] + tolerance)
    )
    return blue_mask


def get_overly_green_mask(image: np.ndarray, tolerance: int) -> np.ndarray:
    """
    Returns a mask of pixels that are overly green.

    :param image: image on which mask is to be computed
    :param tolerance: minimal difference between the green channel and other channel values for pixel to be considered overly green
    :return: mask denoting overly green pixels by True
    """
    green_mask = (image[:, :, 0] > (image[:, :, 2] + tolerance)) & (
        image[:, :, 0] > (image[:, :, 1] + tolerance)
    )
    return green_mask


def get_high_single_channel_intensity_mask(
    image: np.ndarray, tolerance: int
) -> np.ndarray:
    """
    Returns a mask of pixels that have a high intensity in any single channel.

    :param image: image on which mask is to be computed
    :param tolerance: minimal difference between the channel values for pixel to be considered too high for any channel
    :return: mask denoting overly high intensity pixels by True
    """
    red_mask = get_overly_red_mask(image, tolerance)
    blue_mask = get_overly_blue_mask(image, tolerance)
    green_mask = get_overly_green_mask(image, tolerance)
    return np.logical_or(np.logical_or(red_mask, blue_mask), green_mask)


def get_distance_to_vector_mask(
    image: np.ndarray, vector: np.ndarray, threshold: float
) -> np.ndarray:
    """
    Returns a mask of pixels which distance to a vector is over a given threshold.

    :param image: image on which mask is to be computed
    :param vector: pixel/vector to which the distance is calculated
    :param threshold: threshold for the normalized distance to the vector for pixel to be considered too far
    :return: mask denoting pixels further from the vector than given threshold by True
    """
    return stat.get_normalized_distance_to_vector(image, vector) > threshold


def erode_dilate_mask(
    mask: np.ndarray, erode_iter: int, dilate_iter: int
) -> np.ndarray:
    """
    Perform erosion and dilation specified number of times on a mask converted to uint8.
    Uses 3x3 ones kernel.

    :param mask: mask which is to be morphed
    :param erode_iter: number of applied erosions
    :param dilate_iter: number of applied dilations
    :return: morphed boolean mask
    """
    eroded = cv2.erode(
        mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=erode_iter
    )
    eroded_and_dilated = cv2.dilate(
        eroded, np.ones((3, 3), np.uint8), iterations=dilate_iter
    )
    return eroded_and_dilated.astype(bool)
