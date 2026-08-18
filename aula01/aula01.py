from enum import Enum

import cv2
import matplotlib

matplotlib.use("TkAgg")  # No Linux
import matplotlib.pyplot as plt
import numpy as np


def splitBGRChannels(image, colors=True):
    if colors:
        blue_channel = np.zeros(
            (image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint8
        )
        green_channel = np.zeros(
            (image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint8
        )
        red_channel = np.zeros(
            (image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint8
        )

        blue_channel[:, :, 0] = image[:, :, 0]
        green_channel[:, :, 1] = image[:, :, 1]
        red_channel[:, :, 2] = image[:, :, 2]

    else:
        blue_channel = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        green_channel = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        red_channel = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        blue_channel[:, :] = image[:, :, 0]
        green_channel[:, :] = image[:, :, 1]
        red_channel[:, :] = image[:, :, 2]

    return blue_channel, green_channel, red_channel


def greyImage(image):
    grey_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            grey_image[i, j] = np.mean(image[i, j])
    return grey_image


def showImage(image, title="Image"):
    cv2.imshow(title, image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


def showBGRChannels(image):
    blue_channel, green_channel, red_channel = splitBGRChannels(image)
    showImage(blue_channel, "Blue Channel")
    showImage(green_channel, "Green Channel")
    showImage(red_channel, "Red Channel")


def histogram(image, color_space="B"):
    class ColorSpace(Enum):
        BLUE = "B"
        GREEN = "G"
        RED = "R"
        GREY = "GREY"

    channel = None
    match color_space:
        case ColorSpace.BLUE.value:
            blue_channel, _, _ = splitBGRChannels(image)
            channel = blue_channel
        case ColorSpace.GREEN.value:
            _, green_channel, _ = splitBGRChannels(image)
            channel = green_channel
        case ColorSpace.RED.value:
            _, _, red_channel = splitBGRChannels(image)
            channel = red_channel
        case ColorSpace.GREY.value:
            channel = greyImage(image)
        case _:
            raise ValueError("Invalid color space")

    histogram = 256 * [0]
    for i in range(channel.shape[0]):
        for j in range(channel.shape[1]):
            match color_space:
                case ColorSpace.BLUE.value:
                    pixel_color = channel[i, j, 0]
                case ColorSpace.GREEN.value:
                    pixel_color = channel[i, j, 1]
                case ColorSpace.RED.value:
                    pixel_color = channel[i, j, 2]
                case ColorSpace.GREY.value:
                    pixel_color = channel[i, j]
            histogram[pixel_color] += 1

    return histogram


def histogram_graph(image, graph_color="B", graph_title=None):
    class ColorSpace(Enum):
        BLUE = "B"
        GREEN = "G"
        RED = "R"
        GREY = "GREY"

    hist = None
    match graph_color:
        case ColorSpace.BLUE.value:
            graph_color = "blue"
            hist = histogram(image, "B")
        case ColorSpace.GREEN.value:
            graph_color = "green"
            hist = histogram(image, "G")
        case ColorSpace.RED.value:
            graph_color = "red"
            hist = histogram(image, "R")
        case ColorSpace.GREY.value:
            graph_color = "grey"
            hist = histogram(image, "GREY")
        case _:
            graph_color = "grey"
            hist = histogram(image, "GREY")

    pixel = np.arange(0, 256)
    plt.bar(pixel, hist, color=graph_color)
    plt.xlabel("Pixel")
    plt.ylabel("Quantity")
    if graph_title is not None:
        plt.title(graph_title)

    #plt.show()


def main():
    image = cv2.imread("morgan.jpg")
    print(image.shape)
    showImage(image)
    showBGRChannels(image)
    histogram_graph(image, graph_color="B", graph_title="Blue Channel")
    histogram_graph(image, graph_color="G", graph_title="Green Channel")
    histogram_graph(image, graph_color="R", graph_title="Red Channel")
    histogram_graph(image, graph_color="GREY", graph_title="Grey Channel")
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
