import glob
from PIL import Image, ImageEnhance
import numpy as np
from tqdm import tqdm


# Resizes images in path to 480/360
def resize_480(path):
    for item in tqdm(glob.glob(f"{path}/*/*/*/*.jpg")):
        img = Image.open(item)
        imgr = img.resize((480, 360), Image.ANTIALIAS)
        imgr.save(item, "JPEG", quality=90)


# Resizes images in path to 320x240
def resize_32(path):
    for item in tqdm(glob.glob(f"{path}/*/*/*/*.jpg")):
        img = Image.open(item)
        imgr = img.resize((320, 240), Image.ANTIALIAS)
        imgr.save(item, "JPEG", quality=90)


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
# print(mse(img1, img2))

print(resize_480("chalearn-input"))
# print(resize_320("chalearn-input"))
