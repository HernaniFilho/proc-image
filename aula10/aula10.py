import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import matplotlib

matplotlib.use("TkAgg")  # No Linux
import matplotlib.pyplot as plt

from aula01.aula01 import showImage
from aula04.aula04 import enhance_image
from aula07.aula07 import erode_image
from aula08.aula08 import dilate_image


def frontier(image, erosion_mask: list[list[int]]):
    black_white_image = enhance_image(image, r1=154, r2=155)
    # showImage(black_white_image, title="Original Black and White")
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

    # showImage(frontier_image, title="Frontier Image")
    return frontier_image


def fill(
    image,
    initial_point: list[int],
    erosion_mask: list[list[int]],
    dilation_mask: list[list[int]],
    max_iterations: int = 5000,
):
    # Imagem binarizada (objeto = 0, fundo = 255)
    black_white_image = enhance_image(image, r1=154, r2=155)

    # Obter a imagem de fronteira (fronteira = 0, resto = 255)
    frontier_image = frontier(black_white_image, erosion_mask)

    height, width = black_white_image.shape
    y0 = initial_point[0]
    x0 = initial_point[1]

    if frontier_image[y0][x0] == 0:
        raise ValueError("Ponto inicial caiu na fronteira; escolha um ponto interno")

    # Criar a imagem marcador (X0): fundo branco (255) e ponto preto (0)
    area_image = black_white_image.copy()
    for i in range(height):
        for j in range(width):
            area_image[i][j] = 255
    area_image[y0][x0] = 0

    # A dilatacao (convolucao em Python puro) na imagem INTEIRA a cada iteracao
    # e o que fazia parecer loop infinito. Como a area so cresce ao redor do
    # ponto inicial, dilatamos apenas a caixa que contem a area + uma margem.
    margin = max(len(dilation_mask), len(dilation_mask[0])) + 1
    top, bottom, left, right = y0, y0, x0, x0

    changed = True
    iteration = 0
    while changed and iteration < max_iterations:
        iteration += 1

        y1 = max(0, top - margin)
        y2 = min(height, bottom + margin + 1)
        x1 = max(0, left - margin)
        x2 = min(width, right + margin + 1)

        # Dilatacao morfologica (aula08) so na regiao de interesse
        region = area_image[y1:y2, x1:x2].copy()
        dilated = dilate_image(region, dilation_mask)

        changed = False
        for i in range(y2 - y1):
            for j in range(x2 - x1):
                y = y1 + i
                x = x1 + j
                if area_image[y][x] == 0:
                    continue  # X_k sempre contem X_{k-1}
                # dilatou e NAO e fronteira (a fronteira barra o crescimento)
                if dilated[i][j] == 0 and frontier_image[y][x] != 0:
                    area_image[y][x] = 0
                    changed = True
                    top = min(top, y)
                    bottom = max(bottom, y)
                    left = min(left, x)
                    right = max(right, x)

        if iteration % 10 == 0:
            print(
                f"iteracao {iteration}: caixa linhas {top}-{bottom}, colunas {left}-{right}"
            )

        # Se a area encostou na borda da imagem, a fronteira esta aberta (vazou)
        if top <= 1 or left <= 1 or bottom >= height - 2 or right >= width - 2:
            raise ValueError(
                "O preenchimento vazou ate a borda da imagem: a fronteira nao esta "
                "fechada ou o ponto inicial esta fora do objeto. Veja a Frontier Image."
            )

    # Uniao final: area preenchida + fronteira (comeca toda branca)
    final_filled_image = black_white_image.copy()
    for i in range(height):
        for j in range(width):
            if area_image[i][j] == 0 or frontier_image[i][j] == 0:
                final_filled_image[i][j] = 0
            else:
                final_filled_image[i][j] = 255
    showImage(final_filled_image, title="Filled Image")

    # Coloca o resultado de volta na imagem preto e branca original
    result = black_white_image.copy()
    for i in range(height):
        for j in range(width):
            if final_filled_image[i][j] == 0:
                result[i][j] = 0
    showImage(result, title="Imagem P&B Original com Preenchimento")
    return result


def main():
    image = cv2.imread("moedas.jpg")

    # frontier(image, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    fill(
        image,
        [110, 110],  # [linha, coluna] dentro de uma moeda
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],  # erosao -> fronteira
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],  # dilatacao em cruz (simetrica)
    )

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
