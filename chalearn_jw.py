"""
Modified test.py to batch-wise process Chalearn RGB frames to pseudo-depth frames for SLR.

Jan Willruth
"""

import argparse
import glob
import os
import sys

import cv2
from tqdm import tqdm
import gc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils_jw import predict, load_images, to_multichannel, resize_640, brightness

total_memory = 8000
frac = 0.8
limit = int(total_memory * frac)
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(
        memory_limit=limit)])


# Argument Parser
parser = argparse.ArgumentParser(description="High Quality Monocular Depth Estimation via Transfer Learning")
parser.add_argument("--model", default="nyu.h5", type=str, help="Trained Keras model file.")
parser.add_argument("--input", default="chalearn-input/*/*/*/*.jpg", type=str, help="Input folder.")
parser.add_argument("--output", default="chalearn-output/*/*/*/*.jpg", type=str, help="Output folder.")
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {"BilinearUpSampling2D": BilinearUpSampling2D, "depth_loss_function": None}

# Resize images if resize boolean True.
resize = False
if resize:
    print(resize_640("chalearn-input"))
else:
    print("Skipping resizing of images...")

# Get images to process
print("Getting images that need to be processed...")
chalearn_input = glob.glob(args.input)
chalearn_output = {x: 0 for x in glob.glob(args.output)}
to_process = []
for i in chalearn_input:
    if i.replace("input", "output") not in chalearn_output:
        to_process.append(i)
num_of_images = len(to_process)
remaining = num_of_images

# Exit program if no images to process, else continue
if not to_process:
    print(f"No images need to be processed, exiting...")
    sys.exit()
print(f"{num_of_images} images need to be processed, proceeding...")

# Load model into GPU/CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)
print(f"\nModel loaded {args.model}.")

# Current batch, batch size, total number of batches, left and right "borders" for batch
# NOTE: a batch size of 296 with a batch number of 5630 seemed to be the sweet spot on my machine,
# results may vary greatly depending on your hardware specs!
current_batch = 1
batch_size = 296
# Calculate batch number based on number of images to process.
if num_of_images % batch_size == 0:
    batch_number = num_of_images // batch_size
else:
    batch_number = num_of_images // batch_size + 1
l = (current_batch - 1) * batch_size
r = current_batch * batch_size
for i in tqdm(range(batch_number - current_batch + 1)):
    print(f"\nProcessing batch {current_batch} of {batch_number}...\n")

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

    # Save results as to output folder
    for j, item in enumerate(outputs.copy()):
        img = item*1000
        # Convert output to multichannel
        # img = to_multichannel(item*255)
        # Get path to image and name
        path = "/".join(names[j].replace("input", "output").split("\\")[:4])
        name = names[j].split("\\")[4:][0][:-4]
        # Create dir
        os.makedirs(path, exist_ok=True)
        # Resize image
        cv2.resize(img, (320, 240))
        # Save image
        cv2.imwrite(f"{path}/{name}.png", img)

    # Update variables
    current_batch += 1
    l += batch_size
    r += batch_size
    remaining -= batch_size

    # Manually invoke garbage collector to free RAM for next iteration
    gc.collect()

    print("\nFinished batch!\n")

print("\nAll done!!!")
