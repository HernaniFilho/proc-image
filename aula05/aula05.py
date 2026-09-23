import os
import sys
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import matplotlib
matplotlib.use("TkAgg")  # No Linux
import matplotlib.pyplot as plt
import numpy as np

from aula01.aula01 import showImage, greyImage, histogram_graph, histogram
from aula04.aula04 import normalized_histogram

def specify_histogram(image, custom_histogram: list[int]):
    original_hist = histogram(image, "GREY")
    height, width, channels = image.shape
    size = height * width

    prob1, acc1, original_pixel_result = normalized_histogram(original_hist, size)
    prob2, acc2, custom_pixel_result = normalized_histogram(custom_histogram, size)

    
    pixels = np.arange(0, 256)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12,10))

    ax1.bar(pixels, original_pixel_result, color="blue")
    ax1.set_xlabel("Pixel")
    ax1.set_ylabel("Quantity")
    ax1.set_title("Num de Pixel na equalizada original SEM IMAGEM")

    ax2.bar(pixels, custom_pixel_result, color="blue")
    ax2.set_xlabel("Pixel")
    ax2.set_ylabel("Quantity")
    ax2.set_title("Num de Pixel na equalizada Customizada SEM IMAGEM")

    ## equalize pixel Original
    #equalized_original_image = greyImage(image)
    #for i in range(equalized_original_image.shape[0]):
    #        for j in range(equalized_original_image.shape[1]):
    #            old_pixel = equalized_original_image[i, j]
    #            new_pixel = original_pixel_result[old_pixel]
    #
    #            equalized_original_image[i, j] = int(new_pixel)

    ## equalize pixel Custom
    #equalized_custom_image = greyImage(image)
    #for i in range(equalized_custom_image.shape[0]):
    #        for j in range(equalized_custom_image.shape[1]):
    #            old_pixel = equalized_custom_image[i, j]
    #            new_pixel = custom_pixel_result[old_pixel]
    #
    #            equalized_custom_image[i, j] = int(new_pixel)

    #pixels = np.arange(0, 256)
    #fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12,10))

    #hist_original_equalized = histogram(equalized_original_image, "GREY")
    #ax1.bar(pixels, hist_original_equalized, color="blue")
    #ax1.set_xlabel("Pixel")
    #ax1.set_ylabel("Quantity")
    #ax1.set_title("Num de Pixel na equalizada original")

    #hist_custom_equalized = histogram(equalized_custom_image, "GREY")
    #ax2.bar(pixels, hist_custom_equalized, color="blue")
    #ax2.set_xlabel("Pixel")
    #ax2.set_ylabel("Quantity")
    #ax2.set_title("Num de Pixel na equalizada Customizada")

def main():
    original_image = cv2.imread("neve.jpg")
    height, width, _ = original_image.shape
    size = height * width

    raw_custom = [random.randint(1, 1000) for _ in range(256)]
    total_raw = sum(raw_custom)

    custom_histogram = [int((val / total_raw) * size) for val in raw_custom]

    diff = size - sum(custom_histogram)
    custom_histogram[0] += diff

    #custom_histogram = 256 * [0]
    #custom_histogram = [random.randint(0, 25000) for _ in range(256)]

    print(f"Valores custom gerados: {custom_histogram}")
    print(f"Soma do histograma customizado: {sum(custom_histogram)}")
    specify_histogram(original_image, custom_histogram)

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()