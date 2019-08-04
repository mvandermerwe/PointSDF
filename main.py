# This file handles all our argument hyperparameters and setting up the training/predicting.
import argparse
import os
import sys

# Get our model functions:
from sdf_pointconv_model import get_pointconv_model

# Get running function.
from run_sdf_model import run
from mise import mesh_objects

import pdb

parser = argparse.ArgumentParser(description='Run SDF model.')
parser.add_argument('--learning_rate', type=float, help='Initial learning rate.', default=1e-5)
parser.add_argument('--optimizer', type=str, help='Optimizer to use [adam, momentum].', default='adam')
parser.add_argument('--model_func', type=str, help='Model function to call.', default='pointconv', required=True)

parser.add_argument('--model_name', type=str, help='Model name for logging/saving.', default='model_name', required=True)
parser.add_argument('--model_path', type=str, help='Path to save model to (full path is model_path/model_name). Note if warm starting or loading a saved model, will use this full path as well.', required=True)
parser.add_argument('--log_path', type=str, help='Path to save logs to (full path is log_path/model_name).', required=True)

parser.add_argument('--warm_start', dest='warm_start', action='store_true', help='Whether to continue training from the model of the given name.')
parser.set_defaults(warm_start=False)

parser.add_argument('--batch_size', type=int, help='Batch size to run.', default=16)
parser.add_argument('--epochs', type=int, help='Epochs to run.', default=1000)
parser.add_argument('--epoch_start', type=int, help='If continuing a run, the epoch number to start at.', default=0)

parser.add_argument('--training', dest='training', action='store_true', help='If training this run.')
parser.add_argument('--testing', dest='training', action='store_false', help='If testing this run.')
parser.set_defaults(training=True)

# Data inputs.
parser.add_argument('--train_path', type=str, help='Path to Training folder.')
parser.add_argument('--validation_path', type=str, help='Path to Validation folder.')

parser.add_argument('--alpha', type=float, help='Alpha for loss tradeoff between voxel and SDF.', default=0.5)
parser.add_argument('--loss_function', type=str, help='Loss function to use.', default='mse')

#parser.add_argument('--voxelize', dest='voxelize', action='store_true', help='If should create a voxel.')
#parser.set_defaults(voxelize=False)
parser.add_argument('--mesh', dest='mesh', action='store_true', help='If should mesh the voxelization.')
parser.set_defaults(mesh=False)
parser.add_argument('--mesh_folder', type=str, help='Folder to save meshes to.', default='./')
parser.add_argument('--pcd_folder', type=str, help='Folder holding generated point clouds for objects to test.', default='/dataspace/PyrenderData/Depth/')

parser.add_argument('--sdf_count', type=int, help='Number of SDF points to run together for each example. Points are randomly down sampled to this count.', default=64)

args = parser.parse_args()

# Set up model/logging folders as needed.
model_folder = os.path.join(args.model_path, args.model_name)
if not os.path.exists(model_folder):
   os.mkdir(model_folder)
logs_folder = os.path.join(args.log_path, args.model_name)
if not os.path.exists(logs_folder):
   os.mkdir(logs_folder)

model_ = args.model_func
if model_ == 'pointconv':
   model_func = get_pointconv_model
else:
   print("Unknown model requested.")
   sys.exit(0)
    
# Run!
if args.mesh:
   mesh_objects(model_func = model_func,
                model_path = model_folder,
                save_path = args.mesh_folder,
                pcd_folder = args.pcd_folder)
else:
   run(get_model=model_func,
       train_path=args.train_path,
       validation_path=args.validation_path,
       model_path=model_folder,
       logs_path=logs_folder,
       batch_size=args.batch_size,
       epoch_start=args.epoch_start,
       epochs=args.epochs,
       learning_rate=args.learning_rate,
       optimizer=args.optimizer,
       train=args.training,
       warm_start=args.warm_start,
       alpha=args.alpha,
       sdf_count=args.sdf_count,
       loss_function=args.loss_function)
