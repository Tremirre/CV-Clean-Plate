import enum

import cv2
import numpy as np


class Channel(enum.Enum):
    RED = 2
    GREEN = 1
    BLUE = 0


def extract_channel(image: np.ndarray, channel: Channel) -> np.ndarray:
    channelled_img = np.zeros(image.shape)
    channelled_img[:, :, channel.value] = image[:, :, channel.value]
    return channelled_img.astype(np.uint8)


def zero_out_channel(image: np.ndarray, channel: Channel) -> np.ndarray:
    result_img = np.zeros(image.shape)
    for available_channel in Channel:
        if available_channel != channel:
            result_img[:, :, available_channel.value] = image[
                :, :, available_channel.value
            ]
    return result_img.astype(np.uint8)


def distance_from_average_color(image: np.ndarray) -> np.ndarray:
    average_top_color = np.mean(np.mean(image, axis=0), axis=0)
    distance_from_average = np.max((image - average_top_color) ** 2, axis=2)
    distance_from_average /= np.max(distance_from_average)
    return distance_from_average


def reduce_to_dominating_color(image: np.ndarray) -> np.ndarray:
    images_maxed = np.zeros(image.shape)
    max_mask = image == np.max(image, axis=2).reshape(image.shape[0], image.shape[1], 1)
    images_maxed[max_mask] += image[max_mask]
    return images_maxed.astype(np.uint8)


def increase_saturation(image: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def get_changing_pixels(images: list[np.ndarray], tolerance: int = 2) -> np.ndarray:
    result = np.zeros(images[0].shape[:2])
    for i, image_a in enumerate(images):
        for j, image_b in enumerate(images[i + 1 :]):
            result += ~np.all(np.isclose(image_a, image_b, atol=tolerance), axis=2)
    return ((result > 0) * 255).astype(np.uint8)


def extract_common_region(images: list[np.ndarray], tolerance: int = 2) -> np.ndarray:
    changing = get_changing_pixels(images, tolerance)
    eroded = cv2.erode(changing, np.ones((3, 3)), iterations=3)
    return cv2.dilate(eroded, np.ones((3, 3)), iterations=5)


def get_overly_red_mask(
    image: np.ndarray, people_region: np.ndarray, tolerance: int
) -> np.ndarray:
    red_mask = (image[:, :, 2] > (image[:, :, 1] + tolerance)) & (
        image[:, :, 2] > (image[:, :, 0] + tolerance)
    )
    red_mask[people_region == 0] = 0
    return red_mask


def get_overly_blue_mask(
    image: np.ndarray, people_region: np.ndarray, tolerance: int
) -> np.ndarray:
    blue_mask = (image[:, :, 1] > (image[:, :, 2] + tolerance)) & (
        image[:, :, 1] > (image[:, :, 0] + tolerance)
    )
    blue_mask[people_region == 0] = 0
    return blue_mask


def get_overly_green_mask(
    image: np.ndarray, people_region: np.ndarray, tolerance: int
) -> np.ndarray:
    blue_mask = (image[:, :, 0] > (image[:, :, 2] + tolerance)) & (
        image[:, :, 0] > (image[:, :, 1] + tolerance)
    )
    blue_mask[people_region == 0] = 0
    return blue_mask


def get_extreme_mask(
    image: np.ndarray, people_region: np.ndarray, tolerance: int
) -> np.ndarray:
    red_mask = get_overly_red_mask(image, people_region, tolerance)
    blue_mask = get_overly_blue_mask(image, people_region, tolerance)
    green_mask = get_overly_green_mask(image, people_region, tolerance)
    return np.logical_or(np.logical_or(red_mask, blue_mask), green_mask)


def get_distance_to_vector(image: np.ndarray, pixel=np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(image.astype(np.float32) - pixel, axis=2)
    return distances / np.max(distances)


def get_people_mask(
    image: np.ndarray,
    people_region: np.ndarray,
    distance_pixels=list[np.ndarray],
    thresholds=list[float],
    struct: np.ndarray = np.ones((3, 3), np.uint8),
    perform_final_morph: bool = True,
) -> np.ndarray:
    assert len(distance_pixels) == len(thresholds)
    masks = [
        get_distance_to_vector(image, pixel) > threshold
        for pixel, threshold in zip(distance_pixels, thresholds)
    ]
    img_copy = image.copy()
    img_copy[people_region == 0] = 0
    for mask in masks:
        img_copy[mask == 0] = 0
    grayscaled = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    if perform_final_morph:
        grayscaled = cv2.erode(grayscaled, struct, iterations=2)
        grayscaled = cv2.dilate(grayscaled, struct, iterations=2)
    return grayscaled > 0


def merge_use_max(images: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights_stacked = np.dstack(weights)
    max_mask = (weights_stacked.max(axis=2, keepdims=1) == weights_stacked).astype(
        np.float32
    )
    max_maks_norm = max_mask / np.atleast_3d(np.sum(max_mask, axis=2))
    max_maks_destacked = np.array([max_maks_norm]).swapaxes(0, 3)
    result = np.sum(np.array(images) * max_maks_destacked, axis=0).astype(np.uint8)

    return result


def get_pure_image(
    impure_images: np.ndarray, from_pixels: list[np.ndarray], thresholds: list[float]
) -> np.ndarray:
    common_region = extract_common_region(impure_images, 15)
    people_masks = [
        get_people_mask(
            image, common_region, from_pixels, thresholds, perform_final_morph=True
        )
        for image in impure_images
    ]

    for mask, image in zip(people_masks, impure_images):
        mask |= get_extreme_mask(image, common_region, 8)
    final_masks_blurred = [
        cv2.GaussianBlur(mask.astype(np.float32), (9, 9), 3) for mask in people_masks
    ]
    final_image = impure_images[0].copy()

    inversed_weights = [1 - weight for weight in final_masks_blurred]
    maxed = merge_use_max(impure_images, inversed_weights)
    final_image[common_region == 255] = maxed[common_region == 255]
    return final_image
