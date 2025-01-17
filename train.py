import argparse
import os
import pathlib
import time
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

# Keras / TensorFlow
from loss import depth_loss_function
from utils import load_test_data
from model import create_model
from data import get_nyu_train_test_data, get_unreal_train_test_data, \
    get_chalearn_train_test_data, get_autsl_train_test_data
from callbacks import get_nyu_callbacks
from keras.optimizers import Adam

# JW: removed plotting and multi_gpu_model for now as I couldn't get it to work/had no use for it
# from keras.utils import multi_gpu_model
# from keras.utils.vis_utils import plot_model

# JW: TF memory allocation fix
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--data', default='autsl', type=str, help='Training dataset.')  # was default='nyu'
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')  # was default=0.0001
parser.add_argument('--bs', type=int, default=4, help='Batch size')  # was default=4
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')  # was default=20
parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to use')
parser.add_argument('--gpuids', type=str, default='0', help='IDs of GPUs to use')
parser.add_argument('--mindepth', type=float, default=10.0, help='Minimum of input depths')
parser.add_argument('--maxdepth', type=float, default=1000.0, help='Maximum of input depths')
parser.add_argument('--name', type=str, default='densedepth_autsl', help='A name to attach to the training session')
# JW: default='densedepth_nyu'
parser.add_argument('--checkpoint', type=str, default='', help='Start training from an existing model.')
parser.add_argument('--full', dest='full', action='store_true',
                    help='Full training with metrics, checkpoints, and image samples.')

args = parser.parse_args()

# Inform about multi-gpu training
if args.gpus == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuids
    print('Will use GPU ' + args.gpuids)
else:
    print('Will use ' + str(args.gpus) + ' GPUs.')

# Create the model
model = create_model(existing=args.checkpoint)

# Data loaders
if args.data == 'nyu':
    train_generator, test_generator = get_nyu_train_test_data(args.bs)
elif args.data == 'unreal':
    train_generator, test_generator = get_unreal_train_test_data(args.bs)
elif args.data == 'chalearn':
    train_generator, test_generator = get_chalearn_train_test_data(args.bs)
elif args.data == 'autsl':
    train_generator, test_generator = get_autsl_train_test_data(args.bs)
else:
    print("Wrong dataset selected, check your arguments for typos!")
    sys.exit()

# Training session details
# JW: Optimized runID a bit, using properly formatted localtime and f-strings now.
'''runID = str(int(time.time())) + '-n' + str(len(train_generator)) + '-e' + str(args.epochs) + '-bs' + str(
    args.bs) + '-lr' + str(args.lr) + '-' + args.name'''
runID = f"{time.strftime('%Y%m%d-%H%M', time.localtime())}-n{len(train_generator)}-e{args.epochs}-bs{args.bs}" \
        f"-lr{args.lr}-{args.name}"
outputPath = './models/'
runPath = outputPath + runID
pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
print('Output: ' + runPath)

# (optional steps)
'''if True:
    # Keep a copy of this training script and calling arguments
    with open(__file__, 'r') as training_script:
        training_script_content = training_script.read()
    training_script_content = '#' + str(sys.argv) + '\n' + training_script_content
    with open(__file__, 'w') as training_script:
        training_script.write(training_script_content)

    # Generate model plot
    plot_model(model, to_file=runPath + '/model_plot.svg', show_shapes=True, show_layer_names=True)

    # Save model summary to file
    from contextlib import redirect_stdout

    with open(runPath + '/model_summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()'''

# Multi-gpu setup:
basemodel = model
'''if args.gpus > 1:
    model = multi_gpu_model(model, gpus=args.gpus)'''

# Optimizer
optimizer = Adam(lr=args.lr, amsgrad=True)

# Compile the model
print('\n\n\n', 'Compiling model:', runID, '\n\n\tGPU ' + (str(args.gpus) + ' gpus' if args.gpus > 1 else args.gpuids)
      + '\t\tBatch size [ ' + str(args.bs) + ' ] ' + ' \n\n')
model.compile(loss=depth_loss_function, optimizer=optimizer)

print('Ready for training!\n')

# Callbacks
callbacks = []
if args.data == 'nyu':
    callbacks = get_nyu_callbacks(model, train_generator, test_generator,
                                  load_test_data() if args.full else None, runPath)
if args.data == 'unreal':
    callbacks = get_nyu_callbacks(model, train_generator, test_generator,
                                  load_test_data() if args.full else None, runPath)
if args.data == 'chalearn':
    callbacks = get_nyu_callbacks(model, train_generator, test_generator,
                                  load_test_data() if args.full else None, runPath)
if args.data == 'autsl':
    callbacks = get_nyu_callbacks(model, train_generator, test_generator,
                                  load_test_data() if args.full else None, runPath)

# Start training
model.fit(train_generator,
          callbacks=callbacks,
          validation_data=test_generator,
          epochs=args.epochs,
          shuffle=True)

# Save the final trained model:
basemodel.save(runPath + '/model.h5')
