import csv
import cv2
import os
import sys
import numpy as np
from glob import glob
from tqdm import tqdm
from skimage.metrics import structural_similarity, mean_squared_error


def best_worst_mean_ssim():
    print("Getting images to compare...")
    # Get depth and pdepth image paths and join them together in pairs.
    depth = glob("F:/Programming/BA/data-temp/chalearn/249-40/depth/*/*/*/*")
    pdepth = glob("F:/Programming/BA/data-temp/chalearn/249-40/pdepth/*/*/*/*")
    joint = [[x.replace("\\", "/"), y.replace("\\", "/")] for x, y in zip(depth, pdepth)]

    print("Calculating SSIM for images...")
    # Calculate SSIM for each image pair, save to dict and add to ssim_sum.
    ssims = {}
    ssim_sum = 0
    for p in tqdm(joint):
        name = "/".join(p[0].split("/")[-4:])
        d_img = cv2.imread(p[0])
        pd_img = cv2.imread(p[1])
        ssims[name] = structural_similarity(d_img, pd_img, multichannel=True)
        ssim_sum += ssims[name]

    # Save best/worst to CSV file.
    sorted_ssims = sorted(ssims.items(), key=lambda item: item[1])
    best_worst_20 = sorted_ssims[:20] + sorted_ssims[-20:]
    with open("ssim.csv", "w", newline="") as f:
        for t in best_worst_20:
            csv.writer(f).writerow(t)

    # Save mean SSIM to TXT file.
    num_of_images = len(depth)
    with open("ssim_mean.txt", "w") as f:
        f.write(str(ssim_sum / num_of_images))

    return "Finished calculating and saving worst/best SSIM images and mean SSIM!"


def mean_rmse():
    print("Getting images to compare...")
    # Get depth and pdepth image paths and join them together in pairs.
    depth = glob("F:/Programming/BA/data-temp/chalearn/249-40/depth/*/*/*/*")
    pdepth = glob("F:/Programming/BA/data-temp/chalearn/249-40/pdepth/*/*/*/*")
    joint = [[x.replace("\\", "/"), y.replace("\\", "/")] for x, y in zip(depth, pdepth)]

    print("Calculating RMSE for images...")
    # Calculate RMSE for each image pair, adding to a total sum of all RMSEs and calculate mean.
    rmse_sum = 0
    num_of_images = len(depth)
    for p in tqdm(joint):
        d_img = cv2.imread(p[0])
        pd_img = cv2.imread(p[1])
        rmse_sum += np.sqrt(mean_squared_error(d_img, pd_img))

    with open("rmse_mean.txt", "w") as f:
        f.write(str(rmse_sum / num_of_images))

    return "Finished calculating mean RMSE!"


if __name__ == "__main__":
    # Remove files if already existing
    files = ["best_worst_ssims.csv", "mean_ssim.txt", "mean_rmse.txt"]
    for file in files:
        try:
            if os.path.exists(file):
                os.remove(file)
        except OSError:
            print(f"{file} is still open, close it first!")
            sys.exit()

    print(best_worst_mean_ssim())
    print(mean_rmse())
