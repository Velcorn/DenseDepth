import argparse
import glob
import os
import tensorflow as tf
import numpy as np
from PIL import Image

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Keras/TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils_jw import predict, load_images, display_images, to_multichannel, resize_640, np2img, brightness
from matplotlib import pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='chalearn-input/*/*/*/*.jpg', type=str, help='Input filename or folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

# Resize images if resize boolean True.
resize = False
if resize:
    print(resize_640("chalearn-input"))
else:
    print("Skipping resizing of images...")


def process_images(b, bn, l, r):
    print(f"\nProcessing batch {b} of {bn}")

    # Load model into GPU/CPU
    model = load_model(args.model, custom_objects=custom_objects, compile=False)
    print('\nModel loaded ({0}).'.format(args.model))

    # Load images into memory
    print("\nLoading images into memory...")
    names = glob.glob(args.input)[l:r]
    inputs = load_images(names)
    print('Loaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

    # Compute results
    outputs = predict(model, inputs)

    # Save results as JPEG to output folder
    # plasma = plt.get_cmap('plasma')
    for i, item in enumerate(outputs.copy()):
        '''a = item[:, :, 0]
        a -= np.min(a)
        a /= np.max(a)
        a = plasma(a)[:, :, :3]'''
        # Convert output to multichannel
        img = to_multichannel(item)
        # Convert from np array to image
        img = np2img(np.uint8(img*255))
        # Brighten image
        # img = brightness(img, 1.5)
        # Get path to image and name.
        path = "/".join(names[i].replace("input", "output").split("\\")[:4])
        name = names[i].split("\\")[4:][0]
        # Create dirs if not existing
        if not os.path.exists(path):
            os.makedirs(path)
        # Resize image
        img = img.resize((320, 240), Image.ANTIALIAS)
        # Save image
        img.save(f"{path}/{name}", "JPEG", quality=90)
    return "\nFinished batch!\n"
