#!/usr/bin/env python3
import cv2
import sys
import math
import numpy as np
from matplotlib import pyplot as plt


def main():
    image_to_20x20_gray(sys.argv[1])


def image_to_20x20_gray(image_name):
    '''
    Convert rectangular image to 20x20 Gray
    '''
    # Read as RGB image
    image = cv2.imread(image_name) / 255.0

    # Convert image to grays by keeping luminance (Y) only
    brightness = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]
    image_BW = np.zeros(image.shape)
    for i in range(3):
        image_BW[:, :, i] = brightness

    # Converting pixels to pure Black(0) or White(1)
    # image_BW = image_BW >= 0.5

    # Set the size of the short edge as new square image's size
    height, width = brightness.shape
    size = min(height, width)
    # Crop the image to square
    crop_h = (height - size) // 2
    crop_w = (width - size) // 2
    cropped_image = brightness[crop_h:(crop_h+size), crop_w:(crop_w+size)]

    # Change resolution to 20x20 by selecting points from the original image
    new_brightness = cv2.resize(brightness, (20,20), interpolation=cv2.INTER_CUBIC)
    new_image = np.zeros((20, 20, 3))
    for i in range(3):
        new_image[:, :, i] = new_brightness

    # Plot
    original = plt.imread(image_name)
    plt.figure(1)
    plt.subplot(131)
    plt.imshow(np.uint8(np.round(image * 255.0)))
    plt.title('Original')
    plt.subplot(132)
    plt.imshow(image_BW)
    plt.title('Gray')
    plt.subplot(133)
    plt.imshow(new_image)
    plt.title('New')
    plt.show()


if __name__ == '__main__':
    main()
