import numpy as np
import cv2
import os

from typing import NoReturn

data_dir: str = 'photos/second/'


def _modify_saturation(pixels: np.array, ratio: float) -> np.array:
    h, w = pixels.shape
    for i in range(h):
        for k in range(w):
            pixels[i, k] = min(255, pixels[i, k] * ratio)
    return pixels


def _modify_brightness(pixels: np.array, delta: float) -> np.array:
    h, w = pixels.shape
    for i in range(h):
        for k in range(w):
            pixels[i, k] = min(255, pixels[i, k] + delta)
    return pixels


def modify_saturation(image_name: str, ratio: float = 1.2) -> NoReturn:
    bgr_image: np.array = cv2.imread(os.path.join(data_dir, image_name), 1)
    lab_image: np.array = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)

    l: np.array = np.array(lab_image[:, :, 0], np.float32)
    a: np.array = _modify_saturation(np.array(lab_image[:, :, 1], np.float32), ratio)
    b: np.array = _modify_saturation(np.array(lab_image[:, :, 2], np.float32), ratio)
    res: np.array = (cv2
                     .cvtColor(cv2.merge([l, a, b])
                               .astype('uint8'),
                               cv2.COLOR_LAB2BGR))

    modifyed_image: str = f"saturation_{image_name}"

    cv2.imwrite(modifyed_image, res)


def modify_brightness(image_name: str, delta: float = 30) -> NoReturn:
    image: np.array = cv2.imread(os.path.join(data_dir, image_name), 1)
    lab_image: np.array = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l: np.array = _modify_brightness(np.array(lab_image[:, :, 0], np.float32), delta)
    a: np.array = np.array(lab_image[:, :, 1], np.float32)
    b: np.array = np.array(lab_image[:, :, 2], np.float32)
    res: np.array = (cv2
                     .cvtColor(cv2.merge([l, a, b])
                               .astype('uint8'),
                               cv2.COLOR_LAB2BGR))

    modifyed_image: str = f"brightness_{image_name}"

    cv2.imwrite(modifyed_image, res)


if __name__ == "__main__":
    for filename in os.listdir(data_dir):
        modify_saturation(filename)
        modify_brightness(filename)
