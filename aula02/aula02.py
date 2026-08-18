import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import matplotlib
matplotlib.use("TkAgg")  # No Linux
import matplotlib.pyplot as plt

from aula01.aula01 import showImage, histogram_graph, greyImage

def cut_image(image, cutting_point=40, above=False, cutting_range: list = None | list, inside_range=True):
    if len(cutting_range) != 2:
        raise ValueError("Cutting_range must be 2 values only.")
    if cutting_range[0] < 0 or cutting_range[0] > 255:
        raise ValueError("Cutting_range value must be >= 0 or <=255")
    if cutting_range[1] < 0 or cutting_range[1] > 255:
            raise ValueError("Cutting_range value must be >= 0 or <=255")
    if cutting_point < 0 or cutting_point > 255:
        raise ValueError("Cutting_point must be >=0 and <= 255")
    
    histogram_graph(image, graph_color="GREY")
    
    new_image = greyImage(image)
    print("Image grey done")

    showImage(new_image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # A single color
            if cutting_range is None:
                if new_image[i, j] > cutting_point and above:
                    image[i, j] = 255
                if new_image[i, j] < cutting_point and not above:
                    image[i, j] = 255

            # A range of color
            elif len(cutting_range) == 2:
                if inside_range:
                    if new_image[i, j] >= cutting_range[0] and new_image[i, j] <= cutting_range[1]:
                        image[i, j] = 255
                else:
                    if new_image[i, j] < cutting_range[0] or new_image[i, j] > cutting_range[1]:
                        image[i, j] = 255
                

    
    print("image cutted")

    showImage(image, "Cut image")


def main():
    image = cv2.imread("franca.jpg")
    print(image.shape)
    showImage(image, "Original Image")
    cut_image(image, cutting_range=[0, 50], inside_range=True)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
