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


def closure(image, erosion_mask: list[list[int]], dilation_mask: list[list[int]]):
    black_white_image = enhance_image(image, r1=154, r2=155)
    showImage(black_white_image, title="Black n White")
    dilated_image = dilate_image(image, dilation_mask)
    showImage(dilated_image, title="Dilated Image")
    eroded_image = erode_image(dilated_image, erosion_mask)
    showImage(eroded_image, title="Eroded Image")
    return eroded_image


def pattern_finder(image, erosion_mask: list[list[int]]):
    # Primeiro passo: Fazer erosão da imagem preto e branca
    ## Imagem original
    black_white_image = enhance_image(image, r1=154, r2=155)
    showImage(black_white_image, title="Imagem Preto e Branco")

    eroded_image_normal = erode_image(black_white_image, erosion_mask)
    showImage(eroded_image_normal, title="Primeira Erosao")

    # Segundo Passo: Fazer erosão do complemento
    inverted_black_white_image = 255 - black_white_image
    showImage(inverted_black_white_image, title="Complemento da Imagem Preto e Branco")

    ## Inverso da mascara de erosão
    inverted = [[1 - val for val in row] for row in erosion_mask]
    num_cols = len(inverted[0])
    inverted_erosion_mask = []
    inverted_erosion_mask.append([1] * (num_cols + 2))
    for row in inverted:
        new_line = [1] + row + [1]
        inverted_erosion_mask.append(new_line)
    inverted_erosion_mask.append([1] * (num_cols + 2))

    ## Erosão com o inverso
    eroded_image_inverted = erode_image(
        inverted_black_white_image, inverted_erosion_mask
    )
    showImage(eroded_image_inverted, title="Segunda Erosao")

    # TERCEIRO PASSO: Encontrar a intersecção dos pixels pretos
    ## Intersecção
    bool_normal = eroded_image_normal == 0
    bool_inverted = eroded_image_inverted == 255
    intersection_bool = np.logical_and(bool_normal, bool_inverted)

    ## Cria a imagem final começando toda branca (255) e pinta a intersecção de preto (0)
    intersection_image = np.full(black_white_image.shape, 255, dtype=np.uint8)
    intersection_image[intersection_bool] = 0

    # Exibe o resultado final da intersecção
    showImage(intersection_image, title="Terceiro Passo - Interseccao das Erosoes")


def main():
    image = cv2.imread("stars.jpg")
    # dilate_image(image, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    # Máscara de erosão 13x13 em formato de estrela/diamante
    # erosion_mask_13x13 = [
    #    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    #    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    #    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    #    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    #    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    #    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    #    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    #    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    # ]

    # Máscara de dilatação 13x13 full (preenchida com 1s)
    dilation_mask_13x13 = [[1 for _ in range(13)] for _ in range(13)]

    # closure(image, erosion_mask_13x13, dilation_mask_13x13)

    erosion_mask_12x11 = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]

    erosion_mask_13x13 = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]

    # Achar tamanho exato do quadrado da imagem stars para ser o elemento estruturante

    pattern_finder(image, erosion_mask_13x13)

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
