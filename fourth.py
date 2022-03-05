import os
import numpy as np
import cv2

from typing import NoReturn

data_dir: str = 'photos/fourth/'


def haar(image_name: str) -> NoReturn:
    cascade: cv2.CascadeClassifier = cv2.CascadeClassifier("utils/haarcascade.xml")
    image: np.array = cv2.imread(os.path.join(data_dir, image_name))
    faces: np.array = cascade.detectMultiScale(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 1.3, 5, minSize=(10, 10))
    for face in faces:
        cv2.rectangle(image,
                      (face[0], face[1]),
                      (face[0] + face[2], face[1] + face[3]),
                      (200, 255, 100),
                      3)

    cv2.imwrite(f"haar_{image_name}", image)


if __name__ == "__main__":
    for filename in os.listdir(data_dir):
        haar(filename)
