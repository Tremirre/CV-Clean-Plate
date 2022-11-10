import cv2
import numpy as np

from . import masks, weights, stat


def extract_changing_region(
    images: list[np.ndarray], tolerance: int = 2, erosions: int = 3, dilations: int = 3
) -> np.ndarray:
    """
    Extract the changing region from a list of images, erodes noise and dilates final result.

    :param images: list of images from which to extract the changing region
    :param tolerance: tolerance for changing mask extraction, defaults to 2
    :param erosions: number of applied erosions, defaults to 3
    :param dilations: number of applied dilations, defaults to 3
    :return: mask denoting the changing region by True
    """
    changing = masks.get_changing_mask(images, tolerance).astype(np.uint8)
    eroded = cv2.erode(changing, np.ones((3, 3)), iterations=erosions)
    return cv2.dilate(eroded, np.ones((3, 3)), iterations=dilations).astype(np.bool)


class ImagesPurifier:
    """
    Class for purifying a list of images with predified algorithms.
    """

    def __init__(
        self,
        images: list[np.ndarray],
        use_max_merge: bool = True,
        changing_region: np.ndarray = None,
    ) -> None:
        self.images = images
        self.merge_func = (
            weights.merge_images_by_max_weight
            if use_max_merge
            else weights.merge_images_by_weighted_average
        )
        self.changing_region = changing_region
        if changing_region is None:
            self.changing_region = extract_changing_region(
                images, tolerance=15, erosions=3, dilations=5
            )

    def merge_on_changing_region(self, image_weights: list[np.ndarray]) -> np.ndarray:
        """
        Merge a list of images on a changing region mask.

        :param image_weights: weights of each image
        :return: max merged image
        """
        result = self.images[0].copy()
        merged = self.merge_func(self.images, image_weights)
        result[self.changing_region] = merged[self.changing_region]
        return result

    def purify_with_edge_weight(self) -> np.ndarray:
        """
        Purify images with edge weight.

        :return: image max merged with edge weights
        """
        edge_weights = [weights.get_inverse_edge_weight(image) for image in self.images]
        return self.merge_on_changing_region(edge_weights)

    def purify_with_pixel_probability(self) -> np.ndarray:
        """
        Purify images with pixel probabilities and standard deviation.

        :return: image max merged with pixel probabilities.
        """
        prob_weights = [weights.get_histogram_weight(image) for image in self.images]
        for weight, image in zip(prob_weights, self.images):
            weight[:] = np.clip(
                weight - weights.get_standard_deviation_weight(image), 0, 1
            )
        return self.merge_on_changing_region(prob_weights)

    def purify_with_mean_pixel_distance_and_channel_intensity(
        self, mean_threshold: float = 0.4, intensity_tolerance: int = 8
    ) -> np.ndarray:
        """
        Purify images with mean pixel distance and channel intensity.

        :param mean_threshold: threshold for normalized distance from the mean, defaults to 0.4
        :param intensity_tolerance: tolerance for highly-intensive images, defaults to 8
        :return: image max merged with mean pixel distance and channel intensity
        """
        mean_pixel = stat.get_mean_pixel(self.images)
        distance_masks = [
            masks.erode_dilate_mask(
                masks.get_distance_to_vector_mask(image, mean_pixel, mean_threshold),
                2,
                2,
            )
            for image in self.images
        ]
        distance_intensity_masks = [
            np.logical_or(
                mask,
                masks.get_high_single_channel_intensity_mask(
                    image, intensity_tolerance
                ),
            )
            for mask, image in zip(distance_masks, self.images)
        ]
        distance_intensity_weights = [
            1 - cv2.GaussianBlur(mask.astype(np.float32), (9, 9), 3)
            for mask in distance_intensity_masks
        ]
        return self.merge_on_changing_region(distance_intensity_weights)
