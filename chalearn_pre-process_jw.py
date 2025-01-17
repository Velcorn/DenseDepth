"""
A script based on test.py to batch-wise process Chalearn RGB frames to pseudo-depth frames and normalize them for SLR.

Jan Willruth
"""

import argparse
import cv2
import gc
import os
import tifffile
import numpy as np
from glob import glob
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images

# JW: Limit memory usage to a fraction of total GPU memory.
total_memory = 8000
frac = 0.8
limit = int(total_memory * frac)
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(
        memory_limit=limit)])

# Argument Parser
parser = argparse.ArgumentParser(description="High Quality Monocular Depth Estimation via Transfer Learning")
parser.add_argument("--model", default="chalearn_249.h5", type=str, help="Trained Keras model file.")
parser.add_argument("--input", default="chalearn-input", type=str, help="Input folder.")
parser.add_argument("--output", default="chalearn-output", type=str,
                    help="Output folder.")
args = parser.parse_args()

# Custom objects needed for inference and training
custom_objects = {"BilinearUpSampling2D": BilinearUpSampling2D, "depth_loss_function": None}


# Resizes images in path to 640x480
def resize_640():
    print("Resizing images to 640x480...")
    input = glob(f"{args.input}/*/*/*/*.jpg")
    for item in tqdm(input):
        path = "/".join(item.replace("\\", "/").split("/")[:-1])
        name = item.replace("\\", "/").split("/")[-1:][0][:-4]
        img = cv2.imread(item)
        if img.shape[1] == 640:
            continue
        img = cv2.resize(img, (640, 480))
        cv2.imwrite(f"{path}/{name}.jpg", img)
    print("Finished resizing images!")


def estimate_depth():
    # Get images to process
    print("Getting images that need to be processed...")
    input = glob(f"{args.input}/*/*/*/*.jpg")
    output = set(x for x in glob(f"{args.output}/*/*/*/*.npy"))
    to_process = []
    for i in input:
        img = i.replace(args.input, args.output).replace("jpg", "npy")
        if img not in output:
            to_process.append(i.replace("\\", "/"))
    num_of_images = len(to_process)
    remaining = num_of_images

    # Exit program if no images to process, else continue
    if not to_process:
        return "No images need to be processed, skipping..."
    print(f"{num_of_images} images need to be processed, proceeding...")

    # Load model into GPU/CPU
    model = load_model(args.model, custom_objects=custom_objects, compile=False)
    print(f"Loaded model {args.model}")

    # Current batch, batch size, total number of batches, left and right "borders" for batch
    # NOTE: a batch size between 200-400 images seemed to be the sweet spot on my machine;
    # results may vary greatly depending on your hardware specifications (mostly RAM, I suppose)!
    current_batch = 1
    batch_size = 400
    # Calculate batch number based on number of images to process.
    if num_of_images % batch_size == 0:
        batch_number = num_of_images // batch_size
    else:
        batch_number = num_of_images // batch_size + 1
    l = (current_batch - 1) * batch_size
    r = current_batch * batch_size
    for _ in tqdm(range(batch_number - current_batch + 1)):
        print(f"\nProcessing batch {current_batch} of {batch_number}...")

        # Load images into memory
        print("Loading images into memory...")

        # Update right border if number of remaining images is smaller than batch size
        if remaining < batch_size:
            r = l + remaining
        names = to_process[l:r]
        inputs = load_images(names)
        print(f"Loaded {inputs.shape[0]} images of size {inputs.shape[1:]}.")

        # Compute results
        outputs = predict(model, inputs)

        # Save depth estimation to output folder
        for j, item in enumerate(outputs.copy()):
            # Get path to image and name
            path = "/".join(names[j].replace(args.input, args.output).split("/")[:-1])
            name = names[j].split("/")[4:][0][:-4]
            # Create dir
            os.makedirs(path, exist_ok=True)
            # Resize image
            img = cv2.resize(item, (320, 240))
            # Save image as numpy array of type float16 to save storage space
            '''
            NOTE: A float32 NPY file takes up 300 KB, resulting in ~600 GB with ~2 million images;
            a float16 NPY file still takes up 150 KB (~300 GB), but I had enough space available.
            Feel free to come up with a better solution ^^
            '''
            np.save(f"{path}/{name}", img.astype(np.float16))

            # Save as TIFF to view as image.
            # tifffile.imwrite(f"{path}/{name}.tiff", item.astype(np.float16))

        # Update variables
        current_batch += 1
        l += batch_size
        r += batch_size
        remaining -= batch_size

        # Manually invoke garbage collector to free RAM for next iteration
        gc.collect()

        print("\nFinished batch!\n")

    return "Finished all batches!"


def normalize_imgs():
    # Get images to normalize
    print("Getting images that need to be normalized...")
    input = glob(f"{args.output}/*/*/*/*.npy")
    output = set(x for x in glob(f"{args.output}/*/*/*/*.jpg"))
    to_normalize = []
    for i in input:
        if i.replace("npy", "jpg") not in output:
            to_normalize.append(i.replace("\\", "/"))
    num_of_images = len(to_normalize)

    # Exit program if no images to process, else continue
    if not to_normalize:
        return "No images need to be normalized, exiting..."
    print(f"{num_of_images} images need to be normalized, proceeding...")

    if not os.path.exists("min-max.txt"):
        print("Getting min-max across all images...")
        min_val = np.inf
        max_val = 0
        for npy in tqdm(to_normalize):
            arr = np.load(npy)
            min_img_val = np.min(arr)
            max_img_val = np.max(arr)
            if min_img_val < min_val:
                min_val = min_img_val
            if max_img_val > max_val:
                max_val = max_img_val
        # Write values to text file
        with open("min-max.txt", "w") as f:
            f.write(f"{min_val},{max_val}")

    # Get values from text file
    with open("min-max.txt", "r") as f:
        line = f.readline().split(",")
        min_val = float(line[0])
        max_val = float(line[1])

    print("Normalizing and saving images...")
    for npy in tqdm(to_normalize):
        # Min-max normalize image to 0-255 range,
        # see https://stackoverflow.com/questions/48178884/min-max-normalisation-of-a-numpy-array for details.
        img = np.load(npy)
        img = (255.0 * (img - min_val) / (max_val - min_val)).astype(np.uint8)
        # Save normalized image as JPG.
        cv2.imwrite(npy.replace("npy", "jpg"), img)

    print("Finished normalizing images!")


if __name__ == "__main__":
    # Booleans for execution of functions.
    resize = False
    estimate = True
    normalize = True
    remove = True
    clean = True

    if resize:
        print(resize_640())
    else:
        print("Skipping resizing of images...")

    if estimate:
        print(estimate_depth())

    if normalize:
        if remove:
            # Remove existing min-max.txt
            if os.path.exists("min-max.txt"):
                os.remove("min-max.txt")
        print(normalize_imgs())

    if clean:
        print("Cleaning up...")
        # Remove NPY files.
        for file in tqdm(glob("chalearn-output/*/*/*/*.npy")):
            os.remove(file)

    print("All done!!!")
