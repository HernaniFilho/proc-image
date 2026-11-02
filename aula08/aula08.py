import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import matplotlib

matplotlib.use("TkAgg")  # No Linux
import matplotlib.pyplot as plt

from aula01.aula01 import showImage
from aula04.aula04 import enhance_image
from aula06.aula06 import convolute_image
from aula07.aula07 import erode_image


def dilate_image(image, mask: list[list[int]]):
    dilated_image = convolute_image(image, mask, dilation)
    # showImage(dilated_image, title="Dilated Image")
    return dilated_image.copy()


def dilation(mask: list[list[int]], image_patch) -> bool:
    mask_h = len(mask)
    mask_w = len(mask[0])

    for y in range(mask_h):
        for x in range(mask_w):
            # onde a mascara e 1, o pixel do patch precisa ser preto (objeto)
            if mask[y][x] == 1 and image_patch[y][x] == 0:
                return True
    return False


# Concertar
def aperture(image, erosion_mask: list[list[int]], dilation_mask: list[list[int]]):
    black_white_image = enhance_image(image, r1=154, r2=155)
    showImage(black_white_image, title="Black n White")
    eroded_image = erode_image(image, erosion_mask)
    showImage(eroded_image, title="Eroded Image")
    dilated_image = dilate_image(eroded_image, dilation_mask)
    showImage(dilated_image, title="Dilated Image")
    return dilated_image


def main():
    image = cv2.imread("stars.jpg")
    # dilate_image(image, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    # Máscara de erosão 13x13 em formato de estrela/diamante
    erosion_mask_13x13 = [
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    ]

    # Máscara de dilatação 13x13 full (preenchida com 1s)
    dilation_mask_13x13 = [[1 for _ in range(13)] for _ in range(13)]

    aperture(image, erosion_mask_13x13, dilation_mask_13x13)

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
