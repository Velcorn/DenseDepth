"""
A script to prepare the ChaLearn IsoGD dataset, so the depth estimation network can be trained on it.

Expected procedure:
- Place a subset from the RGB test and train set into ./chalearn_data/data, ./chalearn_data for depth data. Should
obviously be the same images for both
- Execute depth_to_png and rename_dirs
- Move the depth data into the same folder as the RGB data
- Execute create_csvs
- Create a ZIP file from the ./chalearn_data folder

Jan Willruth
"""
import cv2
import csv
import os
import shutil
import sys
from glob import glob
from tqdm import tqdm


# Move part of Chalearn dataset to generate a train set for depth estimation.
def move_files():
    print("Moving files...")

    rgb = "chalearn-input2"
    depth = "Z:/Documents/Programming/BA/data-temp/chalearn/249-40/depth"

    for dt in [rgb, depth]:
        if dt == rgb:
            target = "chalearn-data/data"
        else:
            target = "chalearn-data"

        for d in os.listdir(dt):
            if d == "valid":
                continue
            elif d == "train":
                for sd in tqdm(os.listdir(f"{dt}/{d}")):
                    for ssd in os.listdir(f"{dt}/{d}/{sd}")[:10]:
                        target_dir = f"{target}/{d}/{sd}/{ssd}"
                        try:
                            shutil.copytree(f"{dt}/{d}/{sd}/{ssd}", target_dir)
                        except FileExistsError:
                            pass
            elif d == "test":
                for sd in tqdm(os.listdir(f"{dt}/{d}")):
                    for ssd in os.listdir(f"{dt}/{d}/{sd}")[:1]:
                        target_dir = f"{target}/{d}/{sd}/{ssd}"
                        try:
                            shutil.copytree(f"{dt}/{d}/{sd}/{ssd}", target_dir)
                        except FileExistsError:
                            pass

    print("Finished moving files!")


# Resizes and saves depth image JPEGs as 1-channel PNGs.
def depth_to_png():
    print("Resizing and saving images as 1-channel PNG...")
    for item in tqdm(glob("chalearn_data/*/*/*/*.jpg")):
        path = "/".join(item.replace("\\", "/").split("/")[:-1])
        name = item.replace("\\", "/").split("/")[-1:][0][:-4]
        img = cv2.imread(item)[:, :, 1]
        if img.shape[1] == 640:
            continue
        img = cv2.resize(img, (640, 480))
        cv2.imwrite(f"{path}/{name}.png", img)

    # Remove .jpg files
    print("Cleaning up...")
    for jpg in tqdm(glob("chalearn_data/*/*/*/*.jpg")):
        os.remove(jpg)

    print("Finished resizing and saving images!")


# Rename video directories to strip them of M_/K_ prefix.
def rename_dirs():
    print("Renaming directories...")
    for item in glob("chalearn_data/*/*/[K_]*"):
        path = "/".join(item.replace("\\", "/").split("/")[:-1])
        name = item.replace("\\", "/").split("/")[-1:][0]
        os.rename(f"{path}/{name}", f"{path}/{name.replace('K_', '')}")
    for item in glob("chalearn_data/*/*/*/[M_]*"):
        path = "/".join(item.replace("\\", "/").split("/")[:-1])
        name = item.replace("\\", "/").split("/")[-1:][0]
        os.rename(f"{path}/{name}", f"{path}/{name.replace('M_', '')}")
    print("Finished renaming directories!")


# Create CSV files analogue to those in the NYU dataset.
def create_csvs():
    for data_set in ["train", "test"]:
        # Remove CSV file if it already exists.
        try:
            if os.path.exists(f"chalearn_data/data/chalearn_{data_set}.csv"):
                os.remove(f"chalearn_data/data/chalearn_{data_set}.csv")
        except OSError:
            print(f"The file 'chalearn_{data_set}.csv' is still open, close it first!")
            sys.exit()

        rgb = glob(f"chalearn_data/data/{data_set}/*/*/*.jpg")
        depth = glob(f"chalearn_data/data/{data_set}/*/*/*.png")

        # Create joint set while removing topmost directory from strings.
        joint = set()
        for x, y in zip(rgb, depth):
            joint.add((x.replace("\\", "/").replace("chalearn_data/", ""),
                       y.replace("\\", "/").replace("chalearn_data/", "")))

        # Write to CSV file.
        with open(f"chalearn_data/data/chalearn_{data_set}.csv", "w", newline="") as f:
            for item in joint:
                csv.writer(f).writerow(item)

    print("Finished creating CSV files!")


if __name__ == "__main__":
    move = True
    convert = False
    rename = False
    create = False

    if move:
        print(move_files())

    if convert:
        print(depth_to_png())

    if rename:
        print(rename_dirs())

    if create:
        print(create_csvs())

    print("All done!!!")
