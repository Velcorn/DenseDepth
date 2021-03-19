import argparse
import glob
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils_jw import predict, load_images, to_multichannel, resize_640, np2img, brightness

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# Argument Parser
parser = argparse.ArgumentParser(description="High Quality Monocular Depth Estimation via Transfer Learning")
parser.add_argument("--model", default="nyu.h5", type=str, help="Trained Keras model file.")
parser.add_argument("--input", default="chalearn-input/*/*/*/*.jpg", type=str, help="Input filename or folder.")
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {"BilinearUpSampling2D": BilinearUpSampling2D, "depth_loss_function": None}

# Resize images if resize boolean True.
resize = False
if resize:
    print(resize_640("chalearn-input"))
else:
    print("\nSkipping resizing of images...")

# Load model into GPU/CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)
print(f"\nModel loaded {args.model}.")

# Current batch, batch size, total number of batches, left and right "borders" for batch
b = 333
bs = 740
bn = 2252
l = (b-1) * bs
r = b * bs
for i in tqdm(range(bn-b+1)):
    print(f"\nProcessing batch {b} of {bn}...")

    # Load images into memory
    print("\nLoading images into memory...")
    names = glob.glob(args.input)[l:r]
    inputs = load_images(names)
    print(f"Loaded {inputs.shape[0]} images of size {inputs.shape[1:]}.")

    # Compute results
    outputs = predict(model, inputs)
    # Free up RAM by clearing inputs
    inputs = None

    # Save results as JPEG to output folder
    for i, item in enumerate(outputs.copy()):
        # Convert output to multichannel
        img = to_multichannel(item)
        # Convert from np array to image
        img = np2img(np.uint8(img*255))
        # Brighten image
        # img = brightness(img, 1.5)
        # Get path to image and name
        path = "/".join(names[i].replace("input", "output").split("\\")[:4])
        name = names[i].split("\\")[4:][0]
        # Create dirs if not existing
        if not os.path.exists(path):
            os.makedirs(path)
        # Resize image
        img = img.resize((320, 240), Image.ANTIALIAS)
        # Save image
        img.save(f"{path}/{name}", "JPEG", quality=90)

    # Increment variables
    b += 1
    l += bs
    r += bs

    print("Finished batch!\n")

print("\nAll done!!!")
