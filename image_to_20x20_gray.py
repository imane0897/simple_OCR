import cv2
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def image_to_20x20_gray(image_name):
    '''
    Convert rectangular image to 20x20 Gray
    '''
    # Read as RGB image
    # image = cv2.imread(image_name)

    # Convert to NTSC image (YIQ) and then convert to grays by keeping luminance (Y) only
    # image_BW = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]
    # image_YCbCr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    # image_BW = image_YCbCr[:, :, 0]
    image = Image.open(image_name)
    gray = image.convert('L')

    # Let numpy do the heavy lifting for converting pixels to pure black or white
    image_BW = np.asarray(gray).copy()

    # Pixel range is 0...255, 256/2 = 128
    image_BW[image_BW < 128] = 0    # Black
    image_BW[image_BW >= 128] = 255 # White

    # Set the size of the short edge as new square image's size
    height, width = image.shape[ :2]
    size = min(height, width)
    # Crop the image to square
    crop_h = (height - size) / 2
    crop_w = (width - size) / 2
    cropped_image = image_BW[crop_h:(crop_h+size), crop_w:(crop_w+size)]

    # Change resolution to 20x20 by selecting points from the original image
    new_image = np.zeros((20, 20))
    for i in range(0, 20):
        x = min((i+0.5) / 20 * size, size)
        for j in range(0, 20):
            y = min((j+0.5) / 20 * size, size)
            # new_image[i, j] = cropped_image[x, y]
            new_image[i,j] = cropped_image[int(x), int(y)]

    # Plot
    original = plt.imread(image_name)
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(original)
    plt.title('Original')
    plt.subplot(122)
    plt.imshow(new_image)
    plt.title('New')
    plt.show()



image_to_20x20_gray('myDigit.png')
