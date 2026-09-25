import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import matplotlib

matplotlib.use("TkAgg")  # No Linux
from typing import Callable

import matplotlib.pyplot as plt

from aula01.aula01 import greyImage
from aula04.aula04 import enhance_image


def convolute_image(
    image, mask: list[list[int]], operation: Callable, black_white: bool = True
):
    if black_white:
        black_white_image = enhance_image(image, r1=154, r2=155)
    else:
        black_white_image = greyImage(image)

    mask_h = len(mask)  # linhas
    mask_w = len(mask[0])  # colunas
    img_h = black_white_image.shape[0]
    img_w = black_white_image.shape[1]

    # copia (nao e so outra referencia) e zera: o resultado comeca todo branco
    result = black_white_image.copy()
    for i in range(img_h):
        for j in range(img_w):
            result[i, j] = 255

    center_y = mask_h // 2
    center_x = mask_w // 2

    for i in range(img_h - mask_h + 1):
        for j in range(img_w - mask_w + 1):
            # o patch vem da imagem original, nunca do resultado
            image_patch = black_white_image[i : i + mask_h, j : j + mask_w]

            operation_result = operation(mask, image_patch)

            if operation_result and isinstance(operation_result, bool):
                result[i + center_y, j + center_x] = 0  # Pintar de preto

            if isinstance(operation_result, int) and not isinstance(
                operation_result, bool
            ):
                result[i, j] = operation_result

    return result


def main():
    image = cv2.imread("museu.jpg")
    convoluted = convolute_image(
        image, [[0, 1, 0], [1, -4, 1], [0, 1, 0]], edge_detection, False
    )
    cv2.imshow("Convoluted", convoluted)

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def edge_detection(mask: list[list[int]], image_patch: list[list[int]]) -> int:
    mask_x_size = len(mask[0])
    mask_y_size = len(mask)

    result = 0

    for x in range(mask_x_size):
        for y in range(mask_y_size):
            result += mask[y][x] * int(image_patch[y][x])
    return min(abs(result), 255)


if __name__ == "__main__":
    main()
