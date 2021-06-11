"""
A script to post-process the pseudo-depth frames created using chalearn_pre-process_jw.py

Jan Willruth
"""

import cv2
import glob
import math
import os
import numpy as np
from tqdm import tqdm


def post_process():
    print("Getting images to post-process...")
    input = glob.glob("chalearn-output/*/*/*/*.jpg")
    output = set(x.replace("\\", "/") for x in glob.glob("chalearn-post-processed/*/*/*/*"))
    to_post_process = []
    for item in input:
        img = item.replace("\\", "/")
        if img.replace("output", "post-processed") not in output:
            to_post_process.append(img)
    num_of_images = len(to_post_process)

    # Exit program if no images to post-process, else continue
    if not to_post_process:
        return "No images need to be post-processed, exiting..."
    print(f"{num_of_images} images need to be post-processed, proceeding...")

    print("Post-processing images...")
    for item in tqdm(to_post_process):
        # Get path and name separately
        path = "/".join(item.replace("\\", "/").replace("output", "post-processed").split("/")[:-1])
        name = item.replace("\\", "/").split("/")[-1:][0][:-4]
        # Apply gamma correction to image
        img = cv2.imread(item)
        gamma = math.log(.5 * 255) / math.log(np.mean(img))
        corrected_img = np.power(img, gamma).clip(0, 255).astype(np.uint8)
        # Create target dir and save image
        os.makedirs(path, exist_ok=True)
        cv2.imwrite(f"{path}/{name}.jpg", corrected_img)

    return "All done!"


if __name__ == "__main__":
    print(post_process())
