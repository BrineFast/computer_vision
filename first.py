from typing import NoReturn

import numpy as np
import cv2
import os
from matplotlib import pyplot as plt


def get_avg(img: object, left_bound: int, right_bound: int, bottom_bound: int, top_bound: int):
    gray_image: np.array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cropped_gray_image: np.array = gray_image[bottom_bound:top_bound, left_bound:right_bound]
    height, width = cropped_gray_image.shape
    nominator: int = 0
    for h in range(height):
        for w in range(width):
            nominator += cropped_gray_image[h, w]
    return nominator / (height * width)


def print_plt(ox: list, oy: list, n: int) -> NoReturn:
    plt.figure(n)
    plt.scatter(ox, oy)
    plt.xticks(sorted(ev))
    plt.xlabel('EV')
    plt.ylabel('Avg brightness')
    plt.grid()


if __name__ == "__main__":
    data_dir: str = 'photos/first/'
    ev: list = []
    avg_brightness: list = []
    for filename in os.listdir(data_dir):
        ev.append(int(filename.replace(".jpg", "")))
        avg_brightness.append(get_avg(plt.imread(os.path.join(data_dir, filename)), 900, 1028, 3000, 3128))
    print_plt(ev, avg_brightness, 1)
    print_plt(ev, np.log10(avg_brightness), 2)
    plt.show()
