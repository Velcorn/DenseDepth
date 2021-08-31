"""
A script to prepare the AUTSL dataset, so the depth estimation network can be trained on it.

Expected procedure:
- Execute extract_frames to extract 40 frames from each video file and save them in target directories
- Execute create_csvs to generate CSV files in the format of: "RGB_name, depth_name"
- Zip the "autsl_data/data" folder, rename it to "autsl_data.zip" and move it into the root dir

Jan Willruth
"""
import cv2
import csv
import os
import re
import sys
import numpy as np
from glob import glob
from tqdm import tqdm


# Human sorting
# Implementation from: https://stackoverflow.com/questions/4836710/is-there-a-built-in-function-for-string-natural-sort
def digit(text):
    return int(text) if text.isdigit() else text.lower()


def human_sort(text):
    return [digit(c) for c in re.split(r'(\d+)', text)]


# Extract frame of part of AUTSL dataset to generate a train set for depth estimation.
def extract_frames():
    print("Extracting frames...")

    # Create base directory to move files to.
    os.makedirs("autsl_data", exist_ok=True)
    os.makedirs("autsl_data/data", exist_ok=True)
    os.makedirs("autsl_data/data/train", exist_ok=True)
    os.makedirs("autsl_data/data/test", exist_ok=True)

    # Get paths for subsets
    train_color = sorted(glob("AUTSL/train/*_color.mp4"), key=human_sort)
    train_depth = sorted(glob("AUTSL/train/*_depth.mp4"), key=human_sort)
    test_color = sorted(glob("AUTSL/valid/*_color.mp4"), key=human_sort)
    test_depth = sorted(glob("AUTSL/valid/*_depth.mp4"), key=human_sort)
    train_output = "autsl_data/data/train"
    test_output = "autsl_data/data/test"

    frames = 40
    for subset in [train_color, train_depth, test_color, test_depth]:
        # Set number according to train or test
        if subset in [train_color, train_depth]:
            number = 1250
        else:
            number = 125

        for vid in tqdm(subset[:number]):
            # Get path to video
            path = vid.replace("\\", "/")

            # Get name for saving
            if subset in [train_color, test_color]:
                name = path.split("/")[-1].replace("_color.mp4", "")
            else:
                name = path.split("/")[-1].replace("_depth.mp4", "")

            # Create video capture, calculate frames_step necessary to extract target frame number from video.
            vidcap = cv2.VideoCapture(path)
            if not vidcap.isOpened():
                raise ValueError("Error opening video file")

            total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
            frames_step = total_frames / frames
            for i in range(frames):
                vidcap.set(1, i * frames_step)
                success, image = vidcap.read()

                # Save as 384x384 JPEG when color, 192x192 PNG when depth
                if subset is train_color:
                    image = cv2.resize(image, (384, 384))
                    cv2.imwrite(f"{train_output}/{name}_{i:04n}.jpg", image)
                elif subset is train_depth:
                    image = cv2.resize(image, (192, 192))
                    cv2.imwrite(f"{train_output}/{name}_{i:04n}.png", np.clip(image[:, :, 1], 10, 255))
                elif subset is test_color:
                    image = cv2.resize(image, (384, 384))
                    cv2.imwrite(f"{test_output}/{name}_{i:04n}.jpg", image)
                else:
                    image = cv2.resize(image, (192, 192))
                    cv2.imwrite(f"{test_output}/{name}_{i:04n}.png", np.clip(image[:, :, 1], 10, 255))
            vidcap.release()

    return "Finished extracting frames!"


# Create CSV files analogous to those in the NYU dataset.
def create_csvs():
    print("Creating CSV files...")

    for data_set in ["train", "test"]:
        # Remove CSV file if it already exists.
        try:
            if os.path.exists(f"autsl_data/data/autsl_{data_set}.csv"):
                os.remove(f"autsl_data/data/autsl_{data_set}.csv")
        except OSError:
            print(f"The file 'autsl_{data_set}.csv' is still open, close it first!")
            sys.exit()

        # Get path of all images.
        rgb = glob(f"autsl_data/data/{data_set}/*.jpg")
        depth = glob(f"autsl_data/data/{data_set}/*.png")

        # Create joint set while removing topmost directory from strings.
        joint = [[x.replace("\\", "/").replace("autsl_data/", ""), y.replace("\\", "/").replace("autsl_data/", "")]
                 for x, y in zip(rgb, depth)]

        # Write to CSV file.
        with open(f"autsl_data/data/autsl_{data_set}.csv", "w", newline="") as f:
            for item in joint:
                csv.writer(f).writerow(item)

    return "Finished creating CSV files!"


if __name__ == "__main__":
    extract = False
    create = True

    if extract:
        print(extract_frames())

    if create:
        print(create_csvs())

    print('Please zip the "autsl_data/data" folder, so the content is "./data/...", '
          'rename the ZIP file to "autsl_data" and place it in the root folder!')
