import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import matplotlib

matplotlib.use("TkAgg")  # No Linux
import matplotlib.pyplot as plt

from aula01.aula01 import showImage
from aula06.aula06 import convolute_image


def erode_image(image, mask: list[list[int]]):
    convoluted_image = convolute_image(image, mask, erosion)
    showImage(convoluted_image, title="Eroded Image")
    return convoluted_image


def erosion(mask: list[list[int]], image_patch) -> bool:
    mask_h = len(mask)
    mask_w = len(mask[0])

    for y in range(mask_h):
        for x in range(mask_w):
            # onde a mascara e 1, o pixel do patch precisa ser preto (objeto)
            if mask[y][x] == 1 and image_patch[y][x] != 0:
                return False
    return True


def main():
    image = cv2.imread("moedas.jpg")
    erode_image(image, [[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
