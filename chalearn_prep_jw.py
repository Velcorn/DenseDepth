"""
A script to prepare the ChaLearn IsoGD dataset, so the depth estimation network can be trained on it.

Jan Willruth
"""

import cv2
import os
from glob import glob
from tqdm import tqdm


# Saves images in path as .png
def img_to_png():
    print("Saving images as png...")
    for item in tqdm(glob("chalearn_data/*/*/*/*.jpg")):
        path = "/".join(item.replace("\\", "/").split("/")[:-1])
        name = item.replace("\\", "/").split("/")[-1:][0][:-4]
        if os.path.exists(f"{path}/{name}.png"):
            continue
        img = cv2.imread(item)
        cv2.imwrite(f"{path}/{name}.png", img)
    print("Finished saving images!")


if __name__ == "__main__":
    print(img_to_png())

    print("Cleaning up...")
    # Remove .jpg files
    for jpg in tqdm(glob("chalearn_data/*/*/*/*.jpg")):
        os.remove(jpg)

    print("All done!!!")
