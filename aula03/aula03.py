import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import matplotlib
matplotlib.use("TkAgg")  # No Linux
import matplotlib.pyplot as plt
import numpy as np

from aula01.aula01 import showImage, greyImage

#
# Fazer as funções aplicarem num pixel apenas e reaproveitar
#

def apply_contrast_brightness(image, contrast_level: float = 2.0, brightness_level: float = 100):
    contrasted_image = greyImage(image)
    old_pixels = np.arange(0, 256)
    new_pixels = np.clip(old_pixels * contrast_level + brightness_level, 0, 255)

    for i in range(contrasted_image.shape[0]):
        for j in range(contrasted_image.shape[1]):
            old_pixel = float(contrasted_image[i, j])
            new_pixel = old_pixel * contrast_level + brightness_level
            
            # Clip of values to avoid overflow in contrast image
            if new_pixel > 255:
                new_pixel = 255
            elif new_pixel < 0:
                new_pixel = 0

            contrasted_image[i, j] = int(new_pixel)
            
    showImage(title="Contrasted Image", image=contrasted_image)
    plt.title("Difference in contrast and brightness")
    plt.xlabel("Original Pixel")
    plt.ylabel("Contrast Pixel")
    plt.xticks(np.arange(min(old_pixels), max(old_pixels) + 1, step=25.0))
    plt.plot(new_pixels, linewidth=2)

    return contrasted_image

def negative_image(image):
    neg_image = greyImage(image)
    old_pixels = np.arange(0, 256)
    new_pixels = np.clip(255 - old_pixels, 0, 255)

    for i in range(neg_image.shape[0]):
        for j in range(neg_image.shape[1]):
            neg_image[i, j] = 255 - neg_image[i, j]

    showImage(title="Negative Image",image=neg_image)
    plt.title("Difference in Negative")
    plt.xlabel("Original Pixel")
    plt.ylabel("Negative Pixel")
    plt.xticks(np.arange(min(old_pixels), max(old_pixels) + 1, step=25.0))
    plt.plot(new_pixels, linewidth=2)
    
    return neg_image

def parabolic_tone(image):
    parab_image = greyImage(image)
    old_pixels = np.arange(0, 256)
    new_pixels = np.clip(((old_pixels/256) ** 2) * 255, 0, 255)

    for i in range(parab_image.shape[0]):
        for j in range(parab_image.shape[1]):
            old_pixel = parab_image[i, j]
            new_pixel = ((old_pixel/256) ** 2) * 255

            # Clip of values to avoid overflow in contrast image
            if new_pixel > 255:
                new_pixel = 255
            elif new_pixel < 0:
                new_pixel = 0

            parab_image[i, j] = int(new_pixel)

    showImage(title="Parabolic Image",image=parab_image)
    plt.title("Difference in Parabolic Tone")
    plt.xlabel("Original Pixel")
    plt.ylabel("Parabolic Pixel")
    plt.xticks(np.arange(min(old_pixels), max(old_pixels) + 1, step=25.0))
    plt.plot(new_pixels, linewidth=2)

    return parab_image

def zig_zag(image):
    zz_image = greyImage(image)

    lim1 = 256/4
    lim2 = 256/2

    for i in range(zz_image.shape[0]):
        for j in range(zz_image.shape[1]):
            old_pixel = zz_image[i, j]

            if old_pixel <= lim1:
                new_pixel = (old_pixel/ lim1) * 255
            elif old_pixel > lim1 and old_pixel < lim2:
                new_pixel = 255 - ((old_pixel - lim1) / (lim2 - lim1)) * 255
            else:
                new_pixel = ((old_pixel - lim2) / (255 - lim2)) * 255


            zz_image[i, j] = int(round(new_pixel))

    showImage(title="Zig Zag Image",image=zz_image)
    plt.title("Difference in Zig Zag")
    plt.xlabel("Original Pixel")
    plt.ylabel("Zig Zag Pixel")

    old_pixels = np.arange(0, 256)
    plt.xticks(np.arange(min(old_pixels), max(old_pixels) + 1, step=25.0))
    
    temp1 = np.arange(0, lim1)
    temp2 = np.arange(lim1, lim2)
    temp3 = np.arange(lim2, 256)
  
    new_pixels = np.concatenate((
        (temp1 / lim1) * 255,
        255 - ((temp2 - lim1) / (lim2 - lim1)) * 255,
        ((temp3 - lim2) / (255 - lim2)) * 255
    ))

    plt.plot(new_pixels, linewidth=2)

    return zz_image



def main():
    image = cv2.imread("morgan.jpg")
    showImage(image)
    #apply_contrast_brightness(image, contrast_level=2, brightness_level=0)
    #negative_image(image)
    #parabolic_tone(image)
    zig_zag(image)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()