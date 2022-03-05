import os
import numpy as np
import cv2

from typing import NoReturn

data_dir: str = 'photos/third/'


def histogram(image_name: str) -> NoReturn:
    image: np.array = cv2.imread(os.path.join(data_dir, image_name))
    image_yuv: np.array = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    modifyed_image: str = f"histogram_{image_name}"

    cv2.imwrite(modifyed_image,
                cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR))


def clahe(image_name: str) -> NoReturn:
    image: np.array = cv2.imread(os.path.join(data_dir, image_name))
    image_yuv: np.array = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_yuv[:, :, 0] = (cv2
                          .createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                          .apply(image_yuv[:, :, 0]))
    modifyed_image: str = f"clahe_{image_name}"

    cv2.imwrite(modifyed_image,
                cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR))


def blur(image_name: str) -> NoReturn:
    image: np.array = cv2.imread(os.path.join(data_dir, image_name))
    modifyed_image: str = f"gaussian_blur_{image_name}"

    cv2.imwrite(modifyed_image,
                cv2.GaussianBlur(image, (21, 21), sigmaX=0))


def _sobel_deriv_x(image_name: str, image: np.array) -> NoReturn:
    modifyed_image: str = f"obel_deriv_x_{image_name}"

    cv2.imwrite(modifyed_image,
                cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5))


def _sobel_deriv_y(image_name: str, image: np.array) -> NoReturn:
    modifyed_image: str = f"obel_deriv_y_{image_name}"

    cv2.imwrite(modifyed_image,
                cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5))

def sobel(image_name: str) -> NoReturn:
    image: np.array = cv2.imread(os.path.join(data_dir, image_name), 0)
    _sobel_deriv_x(image_name, image)
    _sobel_deriv_y(image_name, image)

def laplas(image_name: str) -> NoReturn:
    image: np.array = cv2.imread(os.path.join(data_dir, image_name), 0)
    modifyed_image: str = f"laplas_{image_name}"

    cv2.imwrite(modifyed_image,
                cv2.Laplacian(image, cv2.CV_64F))


def _gauss_pyramid(image: np.array) -> list:
    gauss_pyr: list = [image.copy()]
    gauss_pyr.append(cv2.pyrDown(gauss_pyr[0]))
    for i in range(1, 16):
        gauss_pyr.append(cv2.pyrDown(gauss_pyr[i - 1]))
    return gauss_pyr


def _laplas_pyramid(gauss_pyr: list) -> list:
    laplas_pyr = [gauss_pyr[len(gauss_pyr) - 2]]
    for i in range(len(gauss_pyr) - 2, 0, -1):
        gauss = cv2.pyrUp(gauss_pyr[i])
        gauss = gauss[:gauss_pyr[i - 1].shape[0],
                :gauss_pyr[i - 1].shape[1]]
        laplas_pyr.append(cv2.subtract(gauss_pyr[i - 1], gauss))
    return laplas_pyr


def pyramid_blending(first_image_name: str, second_image_name: str) -> NoReturn:
    first_image: np.array = cv2.imread(os.path.join(data_dir, first_image_name))
    second_image: np.array = cv2.imread(os.path.join(data_dir, second_image_name))

    gauss_first: list = _gauss_pyramid(first_image)
    gauss_second: list = _gauss_pyramid(second_image)

    laplas_first: list = _laplas_pyramid(gauss_first)
    laplas_second: list = _laplas_pyramid(gauss_second)

    merged_pyr: list = []
    for first, second in zip(laplas_first, laplas_second):
        merged_pyr.append(np.hstack((first[:, :int(first.shape[1] / 2)],
                                     second[:, int(first.shape[1] / 2):])))

    blending: np.array = merged_pyr[0]
    for i in range(1, 16):
        blending = cv2.pyrUp(blending)[:merged_pyr[i].shape[0],
                                       :merged_pyr[i].shape[1]]
        blending = cv2.add(blending, merged_pyr[i])

    direct_blending: np.array = np.hstack((first_image[:, :int(merged_pyr[15].shape[1] / 2)],
                                           second_image[:, int(merged_pyr[15].shape[1] / 2):]))

    cv2.imwrite('pyramid_blending.jpg', blending)
    cv2.imwrite('direct_blending.jpg', direct_blending)


def alpha_blending(foreground_image: str, background_image: str, alpha_mask_image: str) -> NoReturn:
    foreground: float = cv2.imread(os.path.join(data_dir, foreground_image)).astype(float)
    background: float = cv2.imread(os.path.join(data_dir, background_image)).astype(float)
    alpha: float = cv2.imread(os.path.join(data_dir, alpha_mask_image)).astype(float)

    foreground = cv2.multiply(alpha / 255, foreground)
    background = cv2.multiply(1.0 - alpha / 255, background)

    cv2.imwrite("alpha_blending.jpg",
                cv2.add(foreground, background))


if __name__ == "__main__":
    images: list = os.listdir(data_dir)

    for i in range(len(images) - 2):
        histogram(images[i])
        clahe(images[i])
        blur(images[i])
        sobel(images[i])
        laplas(images[i])

        pyramid_blending(images[i], images[i + 1])
        alpha_blending(images[i], images[i + 1], images[i + 2])
