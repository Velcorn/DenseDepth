"""
A script to prepare the ChaLearn IsoGD dataset, so the depth estimation network can be trained on it.

Expected procedure:
- Execute move_files to copy a subset of the Chalearn dataset
- Execute depth_to_png to convert depth images to 1-channel PNGs
- Manually move the depth data from "chalearn_data" into "chalearn_data/data" to merge (faced permission errors)
- Execute create_csvs to generate CSV files in the format of: "RGB_name, depth_name"
- Zip the "chalearn_data/data" folder, rename it to "chalearn_data.zip" and move it into the root dir

Jan Willruth
"""
import cv2
import csv
import os
import shutil
import sys
import numpy as np
from glob import glob
from tqdm import tqdm


# Copy part of Chalearn dataset to generate a train set for depth estimation.
def copy_files():
    print("Copying files...")

    # Source dirs
    rgb = "chalearn-input2"
    depth = "Z:/Documents/Programming/BA/data-temp/chalearn/249-40/depth"

    # Iterate over both source dirs and set variables.
    for dt in [rgb, depth]:
        if dt == rgb:
            target = "chalearn_data/data"
            replace = "M_"
            print("RGB:")
        else:
            target = "chalearn_data"
            replace = "K_"
            print("Depth:")

        # Iterate over desired subdirs and move first 1/5 dirs to new dir.
        # Rename target dir after copying to merge in the end.
        for d in os.listdir(dt):
            if d == "valid":
                continue
            elif d == "train":
                for sd in tqdm(os.listdir(f"{dt}/{d}")):
                    for ssd in os.listdir(f"{dt}/{d}/{sd}")[:5]:
                        target_dir = f"{target}/{d}/{sd}/{ssd}"
                        target_renamed = target_dir.replace(replace, "")

                        # Copy dir if target dir doesn't exist already.
                        if not os.path.exists(target_dir) and not os.path.exists(target_renamed):
                            shutil.copytree(f"{dt}/{d}/{sd}/{ssd}", target_dir)

                        # Rename dir by stripping M_/K_ prefix
                        if os.path.exists(target_dir):
                            os.rename(target_dir, target_renamed)
            elif d == "test":
                for sd in tqdm(os.listdir(f"{dt}/{d}")):
                    for ssd in os.listdir(f"{dt}/{d}/{sd}")[:1]:
                        target_dir = f"{target}/{d}/{sd}/{ssd}"
                        target_renamed = target_dir.replace(replace, "")

                        # Copy dir if target dir doesn't exist already.
                        if not os.path.exists(target_dir) and not os.path.exists(target_renamed):
                            shutil.copytree(f"{dt}/{d}/{sd}/{ssd}", target_dir)

                        # Rename dir by stripping M_/K_ prefix
                        if os.path.exists(target_dir):
                            os.rename(target_dir, target_renamed)

    return "Finished copying files!"


# Resize and save depth image JPEGs as 1-channel PNGs.
def depth_to_png():
    print("Resizing and saving images as 1-channel PNG...")
    for item in tqdm(glob("chalearn_data/*/*/*/*.jpg")):
        path = "/".join(item.replace("\\", "/").split("/")[:-1])
        name = item.replace("\\", "/").split("/")[-1:][0][:-4]
        img = cv2.imread(item)[:, :, 1]

        # Check if image already has desired shape.
        if img.shape[1] == 640:
            continue
        else:
            img = cv2.resize(img, (640, 480))

        # Clip to range 10-255
        img = np.clip(img, 10, 255)

        # Save as PNG and remove JPG.
        cv2.imwrite(f"{path}/{name}.png", img)
        os.remove(item)

    return "Finished resizing and saving images!"


# Create CSV files analogous to those in the NYU dataset.
def create_csvs():
    print("Creating CSV files...")

    for data_set in ["train", "test"]:
        # Remove CSV file if it already exists.
        try:
            if os.path.exists(f"chalearn_data/data/chalearn_{data_set}.csv"):
                os.remove(f"chalearn_data/data/chalearn_{data_set}.csv")
        except OSError:
            print(f"The file 'chalearn_{data_set}.csv' is still open, close it first!")
            sys.exit()

        # Get path of all images.
        rgb = glob(f"chalearn_data/data/{data_set}/*/*/*.jpg")
        depth = glob(f"chalearn_data/data/{data_set}/*/*/*.png")

        # Create joint set while removing topmost directory from strings.
        joint = []
        for x, y in zip(rgb, depth):
            joint.append((x.replace("\\", "/").replace("chalearn_data/", ""),
                          y.replace("\\", "/").replace("chalearn_data/", "")))

        # Write to CSV file.
        with open(f"chalearn_data/data/chalearn_{data_set}.csv", "w", newline="") as f:
            for item in joint:
                csv.writer(f).writerow(item)

    return "Finished creating CSV files!"


if __name__ == "__main__":
    copy = True
    convert = True
    move = True
    create = True

    if copy:
        print(copy_files())

    if convert:
        print(depth_to_png())

    if move:
        print('Please move "test" and "train" folders from "chalearn_data" into "chalearn_data/data" to merge! '
              'Afterwards, simply set "move" to "False" and proceed!')
        sys.exit()

    if create:
        print(create_csvs())

    print('Please zip the "chalearn_data/data" folder, so the content is "./data/...", '
          'rename the ZIP file to "chalearn_data" and place it in the root folder!')
