# Generate the train/validation/test splits for our partial view dataset.

import numpy as np
import os
import pdb

_GRASP_DATABASE = '/dataspace/GraspDatabaseMeshes'
_YCB_DATABASE = '/dataspace/YCBMeshes'
_NUM_POSES_PER_OBJ = 200
_SAVE_DIR = '/home/markvandermerwe/catkin_ws/src/ll4ma_3d_reconstruction/src/data_generation/data_split'

def generate_split(grasp_database_meshes, ycb_meshes, num_poses_per_obj, save_dir):
    '''
    For given list of object meshes, at the specified number of views per mesh, 
    generate a train/validate/test split of the data.

    For grasp database:
    0.8 of objects in training - some of these views (0.1) we'll also place in validation.
    0.1 of objects in validation.
    0.1 of objects in Test.

    For YCB:
    All go into Test.
    '''

    # Determine views/objects to be in each of training/validation.
    np.random.shuffle(grasp_database_meshes)
    training_size = int(len(grasp_database_meshes) * 0.8)
    valid_test_size = (len(grasp_database_meshes) - training_size) // 2

    training_objects = grasp_database_meshes[:training_size] # This we will further split into holdout/training views.
    validation_objects = grasp_database_meshes[training_size:-valid_test_size]
    test_objects = grasp_database_meshes[-valid_test_size:]
    test_objects.extend(ycb_meshes)

    # Create the actual lists of views.
    training_views = []
    validation_views = []
    test_views = []

    # Add the training objects.
    for training_object in training_objects:

        # Put 90 percent of the views into training, the other 10 into validation.
        training_count = int(num_poses_per_obj * 0.9)
        validation_count = num_poses_per_obj - training_count

        for pose_num in range(training_count):
            training_views.append(training_object + '_' + str(pose_num))
        for pose_num in range(num_poses_per_obj - validation_count, num_poses_per_obj):
            validation_views.append(training_object + '_' + str(pose_num))

    # Add validation objects.
    for validation_object in validation_objects:
        for pose_num in range(num_poses_per_obj):
            validation_views.append(validation_object + '_' + str(pose_num))

    # Test objects.
    for test_object in test_objects:
        for pose_num in range(num_poses_per_obj):
            test_views.append(test_object + '_' + str(pose_num))
            
    np.random.shuffle(training_views)
    np.random.shuffle(validation_views)
    np.random.shuffle(test_views)

    np.savetxt(os.path.join(save_dir, 'train_fold.txt'), training_views, fmt='%s')
    np.savetxt(os.path.join(save_dir, 'validation_fold.txt'), validation_views, fmt='%s')
    np.savetxt(os.path.join(save_dir, 'test_fold.txt'), test_views, fmt='%s')

def get_view_splits():
    '''
    Read in the view splits to use for data generation elsewhere.
    '''

    training_views = np.loadtxt(os.path.join(_SAVE_DIR, 'train_fold.txt'), dtype=str)
    validation_views = np.loadtxt(os.path.join(_SAVE_DIR, 'validation_fold.txt'), dtype=str)
    test_views = np.loadtxt(os.path.join(_SAVE_DIR, 'test_fold.txt'), dtype=str)

    return training_views, validation_views, test_views
    
def run():
    # Read in mesh names.
    # grasp_database_meshes = [filename.replace('.stl','') for filename in os.listdir(_GRASP_DATABASE) if ".stl" in filename]
    # ycb_meshes = [filename.replace('.stl', '') for filename in os.listdir(_YCB_DATABASE) if ".stl" in filename]
    
    # generate_split(grasp_database_meshes,
    #                ycb_meshes,
    #                _NUM_POSES_PER_OBJ,
    #                _SAVE_DIR)

    train, validation, test = get_view_splits()

if __name__ == '__main__':
    run()
