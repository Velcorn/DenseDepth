import os
from PIL import Image, ImageEnhance
import numpy as np


# Resizes images in path to 640x320
def resize_640(path):
    items = os.listdir(path)
    for item in items:
        img = Image.open(f'{path}/{item}')
        imgr = img.resize((640, 480), Image.ANTIALIAS)
        imgr.save(f'{path}/{item}', 'JPEG', quality=100)


# Resizes images in path to 320x240
def resize_320(path):
    items = os.listdir(path)
    for item in items:
        img = Image.open(f'{path}/{item}')
        imgr = img.resize((320, 240), Image.ANTIALIAS)
        imgr.save(f'{path}/{item}', 'JPEG', quality=100)


# Gets filenames of items in path
def get_names(path):
    names = os.listdir(path)
    return names


# Convert numpy array to image and save it
def np2img(array):
    img = Image.fromarray(array)
    return img


# Calculate mean squared error
def mse(img1, img2):
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])
    return err


# Alter brightness of an image
def brightness(img, factor):
    enhancer = ImageEnhance.Brightness(img)
    enhanced_img = enhancer.enhance(factor)
    return enhanced_img


img1 = np.asarray(Image.open("frame0000_1.jpg"))
img2 = np.asarray(Image.open("frame0000_2.jpg"))
print(mse(img1, img2))
