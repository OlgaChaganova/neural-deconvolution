import cv2
import numpy as np


def read_image(img_path: str) -> np.array:
    image = cv2.imread(img_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)