import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import matplotlib

matplotlib.use("TkAgg")  # No Linux
import matplotlib.pyplot as plt
import numpy as np

from aula01.aula01 import greyImage, histogram, histogram_graph, showImage


def enhance_image(image, r1: int = 160, r2: int = 240):
    # histogram_graph(image, "GREY", "Fig 1")  # use para descobrir r1 e r2
    enhanced_image = greyImage(image)

    for i in range(enhanced_image.shape[0]):
        for j in range(enhanced_image.shape[1]):
            old_pixel = enhanced_image[i, j]

            if old_pixel <= r1:
                new_pixel = 0
            elif old_pixel >= r2:
                new_pixel = 255
            else:
                new_pixel = 255 * ((old_pixel - r1) / (r2 - r1))

            enhanced_image[i, j] = int(round(new_pixel))

    # showImage(title="Enhanced Image", image=enhanced_image)

    # histogram_graph(enhanced_image, "GREY", "Fig 2")  # use para mostrar a diferenca
    return enhanced_image


def normalized_histogram(histogram: list[int], image_size: int | float):
    prob = 256 * [0]
    acc = 256 * [0]
    pixel_result = 256 * [0]

    for i in range(0, 256):
        prob[i] = histogram[i] / image_size

        if i == 0:
            acc[i] = prob[i]
        else:
            acc[i] = acc[i - 1] + prob[i]

        min_dif = 999  # qualquer numero maior que 1
        # j = 0
        # target = j
        # while True:
        #    if j >= 255:
        #        break
        #
        #    # j / 255 -> 0/7...1/7...2/7...
        #    calc = abs(acc[i] - (j/255)) # 0.123...0.234...0.456...
        #
        #    if calc < min_dif:
        #        target = j
        #        j += 1
        #        min_dif = calc
        #    else:
        #        break

        target = 0
        for j in range(256):
            calc = abs(acc[i] - (j / 255.0))
            if calc < min_dif:
                min_dif = calc
                target = j

        pixel_result[i] = target

    return prob, acc, pixel_result


def equalizate_image(image):
    hist = histogram(image, "GREY")
    height, width, channels = image.shape
    size = height * width

    prob, acc, pixel_result = normalized_histogram(hist, size)

    # equalize pixel
    equalized_image = greyImage(image)
    for i in range(equalized_image.shape[0]):
        for j in range(equalized_image.shape[1]):
            old_pixel = equalized_image[i, j]
            new_pixel = pixel_result[old_pixel]

            equalized_image[i, j] = int(new_pixel)

    pixels = np.arange(0, 256)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    ax1.bar(pixels, hist, color="blue")
    ax1.set_xlabel("Pixel")
    ax1.set_ylabel("Quantity")
    ax1.set_title("Num de Pixel no original Cinza")

    # print(f"Diferenca entre hist e prob")
    # for i in range(0, 256):
    #    prob[i] = prob[i] * size
    #    print(f"[{i}] {hist[i]} {prob[i]} = {hist[i] - prob[i]}")

    ax2.bar(pixels, prob, color="blue")
    ax2.set_xlabel("Pixel")
    ax2.set_ylabel("Quantity")
    ax2.set_title("Distr de probabilidade")

    ax3.bar(pixels, acc, color="blue")
    ax3.set_xlabel("Pixel")
    ax3.set_ylabel("Quantity")
    ax3.set_title("Acumulado normalizado de distr prob")

    hist_equalized = histogram(equalized_image, "GREY")
    ax4.bar(pixels, hist_equalized, color="blue")
    ax4.set_xlabel("Pixel")
    ax4.set_ylabel("Quantity")
    ax4.set_title("Num de Pixel na equalizada")

    return equalized_image


def main():
    image = cv2.imread("neve.jpg")
    showImage(image, "Original Image")
    # enhance_image(image)

    equalized_image = equalizate_image(image)
    showImage(equalized_image, "Equalized Image")

    grey_image = greyImage(image)
    showImage(grey_image, "Original Grey Image")

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
