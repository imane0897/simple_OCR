#!/use/bin/env python2
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt


image_to_20x20_gray(sys.argv[1])


def image_to_20x20_gray(image_name):
    '''
    Convert rectangular image to 20x20 Gray
    '''
    # Read as RGB image
    image = cv2.imread(image_name)

    # Convert image to grays by keeping luminance (Y) only
    brighness = np.round(0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2])
    image_BW = np.zeros(image.shape)
    for i in range(0, 3):
        image_BW[:, :, i] = brighness

    # Let numpy do the heavy lifting for converting pixels to pure black or white
    # image_BW = np.asarray(gray).copy()

    # Pixel range is 0...255, 256/2 = 128
    # image_BW[image_BW < 128] = 0    # Black
    # image_BW[image_BW >= 128] = 255 # White

    # Set the size of the short edge as new square image's size
    height, width = brighness.shape
    size = min(height, width)
    # Crop the image to square
    crop_h = (height - size) / 2
    crop_w = (width - size) / 2
    cropped_image = brighness[crop_h:(crop_h+size), crop_w:(crop_w+size)]

    # Change resolution to 20x20 by selecting points from the original image
    new_brightness = np.zeros((20, 20))
    # for i in range(0, 20):
    #     x = min((i+0.5) / 20 * size, size)
    #     for j in range(0, 20):
    #         y = min((j+0.5) / 20 * size, size)
    #         # new_image[i, j] = cropped_image[x, y]
    #         new_brightness[i,j] = cropped_image[int(x), int(y)]

    new_brightness = cv2.resize(brighness, (20,20), cv2.INTER_CUBIC)
    print(new_brightness.shape)
    new_image = np.zeros((20, 20, 3))
    for i in range(0, 3):
        new_image[:, :, i] = new_brightness

    # Plot
    original = plt.imread(image_name)
    plt.figure(1)
    plt.subplot(131)
    plt.imshow(original)
    plt.title('Original')
    plt.subplot(132)
    plt.imshow(image_BW)
    plt.title('Gray')
    plt.subplot(133)
    plt.imshow(new_image)
    plt.title('New')
    plt.show()
