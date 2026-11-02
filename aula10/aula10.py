import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import matplotlib

matplotlib.use("TkAgg")  # No Linux
import matplotlib.pyplot as plt
import numpy as np

from aula01.aula01 import showImage
from aula04.aula04 import enhance_image
from aula07.aula07 import erode_image
from aula08.aula08 import dilate_image


def frontier(image, erosion_mask: list[list[int]]):
    black_white_image = enhance_image(image, r1=154, r2=155)
    # howImage(black_white_image, title="Original Black and White")
    eroded_image = erode_image(image, erosion_mask)
    # showImage(eroded_image, title="Eroded Image")
    frontier_image = black_white_image.copy()
    for i in range(black_white_image.shape[0]):
        for j in range(black_white_image.shape[1]):
            if (
                black_white_image[i][j] == eroded_image[i][j]
                and black_white_image[i][j] == 0
            ):
                frontier_image[i][j] = 255

    showImage(frontier_image, title="Frontier Image")
    return frontier_image


def fill(
    image,
    initial_point: list[int],
    erosion_mask: list[list[int]],
    dilation_mask: list[list[int]],
):
    # Imagem binarizada (objeto = 0, fundo = 255)
    black_white_image = enhance_image(image, r1=154, r2=155)

    # Obter a imagem de fronteira
    frontier_image = frontier(image, erosion_mask)
    inverted_frontier = 255 - frontier_image

    height, width = black_white_image.shape

    # Criar a imagem marcador (X0): fundo branco (255) e ponto preto (0)
    area_image = np.full_like(black_white_image, 255)
    area_image[initial_point[0], initial_point[1]] = 0

    # Inicializa temp_image com zeros para garantir a entrada no loop
    temp_image = np.zeros_like(area_image)

    # Iteração
    while not np.array_equal(temp_image, area_image):
        temp_image = area_image.copy()

        # Dilatação morfológica
        dilated = dilate_image(temp_image, dilation_mask)

        # Garante que o fundo branco (255) barre a dilatação e o objeto (0) permita crescer.
        for i in range(height):
            for j in range(width):
                if dilated[i][j] == 0 and inverted_frontier[i][j] == 255:
                    dilated[i][j] = 255

        area_image = dilated.copy()

    # união final com a fronteira
    final_filled_image = np.zeros_like(area_image)
    for i in range(height):
        for j in range(width):
            if area_image[i][j] == 0:
                final_filled_image[i][j] = 0
            if frontier_image[i][j] == 0:
                final_filled_image[i][j] = 0

    showImage(final_filled_image, title="Filled Image")
    return final_filled_image


def main():
    image = cv2.imread("moedas.jpg")

    # frontier(image, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    fill(
        image,
        [110, 110],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[1, 1, 0], [1, 1, 1], [1, 1, 1]],
    )

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
