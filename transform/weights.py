import cv2
import numpy as np

from .stat import create_image_histogram


def get_inverse_edge_weight(image: np.ndarray) -> np.ndarray:
    """
    Returns a weight array where edges have low values and flat areas have high values.

    :param image: image on which weight map is to be computed
    :return: array of weights
    """
    edges = cv2.Canny(image, 100, 300) / 255
    edges = cv2.dilate(edges, np.ones((3, 3)), iterations=1)
    edges = cv2.blur(edges, (3, 3))
    return 1 - edges


def get_histogram_weight(image: np.ndarray) -> np.ndarray:
    """
    Returns a weight array where pixels with high probability of being in the image have high values.
    Probability is calculated from color histogram of an image.
    Three color channels are treated as independent variables.

    :param image: image for which weight is calculated
    :return: array of weights
    """
    histogram = create_image_histogram(image).T
    densities = histogram / np.repeat(np.sum(histogram, axis=1), 256).reshape(3, 256)
    pixel_probabilities = np.prod(densities[range(3), image], axis=2)
    return pixel_probabilities / np.max(pixel_probabilities)


def get_standard_deviation_weight(image: np.ndarray) -> np.ndarray:
    """
    Returns a weight array where pixels with high standard deviation have high values.

    :param image: image for which the standard deviation is calculated
    :return: array of weights in the range [0, 1]
    """
    std_img = np.var(np.array(image), axis=2) ** 0.5
    std_img = std_img / np.max(std_img)
    return std_img


def merge_images_by_max_weight(
    images: list[np.ndarray], weights: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merges images by taking the pixel with the highest weight from each image.

    :param images: list of images to be merged
    :param weights: list of corresponding weights
    :return: max merged image
    """
    weights_stacked = np.dstack(weights)
    max_mask = (weights_stacked.max(axis=2, keepdims=1) == weights_stacked).astype(
        np.float32
    )
    max_mask_norm = max_mask / np.atleast_3d(np.sum(max_mask, axis=2))
    max_mask_destacked = np.array([max_mask_norm]).swapaxes(0, 3)
    result = np.sum(np.array(images) * max_mask_destacked, axis=0).astype(np.uint8)

    return result


def merge_images_by_weighted_average(
    images: list[np.ndarray], weights: list[np.ndarray]
) -> np.ndarray:
    """
    Merges images by taking the weighted mean of pixels from each image.

    :param images: list of images to be merged
    :param weights: list of corresponding weights
    :return: max merged image
    """
    weights_array = np.array(weights)
    weights_expanded = np.expand_dims(weights_array, axis=3)
    weights_broadcasted = np.broadcast_to(weights_expanded, (*weights_array.shape, 3))
    summed_weights = np.sum(weights_broadcasted, axis=0)
    result = (
        np.sum(np.array(images) * weights_broadcasted, axis=0) / summed_weights
    ).astype(np.uint8)
    final = np.zeros(result.shape)
    final[summed_weights == 0] = images[0][summed_weights == 0]
    final[summed_weights > 0] = result[summed_weights > 0]
    return final
