# PointSDF

![Results](images/results.png "PointSDF Qualitative Examples")

**See our preprint using PointSDF for real robotic grasping [here](https://sites.google.com/view/reconstruction-grasp/home)**

PointSDF is a novel, point cloud based, implicit surface reconstruction learning architecture based on recent advances in the Computer Vision community [1,2,3]. Our approach uses new advances in point cloud based learning [1] in an architecture designed similar to [2], but with the implicit surface representation introduced in [3]. Our approach allows for fast point cloud based, reconstructions, specifically to be utilized in grasp planning. 

## Design:

![Architecture](images/architecture.png "PointSDF Architecture")

PointSDF represents an object as the zero level set of a function that predicts the Signed Distance Function value for a given query point in arbitrary 3D space around an object. This is a regression problem, which we train using MSE loss.

We embed input point clouds using 4 PointConv embedding layers [1]. Query points are embedded using dropout FC layers. The point clouds are embedded and appended to each input query points's embedding. This joint embedding is passed through 8 FC layers with Batch Norm and ReLU activations (also fed in at layer 4).

## Usage:

### Dependencies:

Our code depends on the following python libraries:
1. Tensorflow (>=1.9.0)
2. numpy
3. mcubes
4. trimesh
5. matplotlib
 
We additionally rely on PointConv for our embedding layers. PointConv should be cloned from [here](https://github.com/DylanWusee/pointconv) and their instructions should be followed to compile their custom TF operations. Additionally, the environment variable `POINTCONV_HOME` should be set to point at the pointconv repository folder.

### Dataset:

You will need the following:
1. [Training Data](https://uofu.box.com/s/xz9nromkjick63kb4fhmx3routy2klsp) - train/validate/test folds of generated sdf values.
2. [PCD Files](https://uofu.box.com/s/nxhr26gyyiud9yi3xap6p9fh6stf32vh) - PCD files of partial views - used for meshing tests.
 
### Pretrained Models:

When extracted, pass name/enclosing folder as specified in the Usage section below:
1. `pointconv_mse` - [here](https://uofu.box.com/s/w3irnokcgflgst6e7qmsn6d2awm39ptv).

### Usage:

For full list of options run:
```
python main.py -h
```
Not all options are necessary each run and have reasonable defaults.

Training run example:
```
python main.py --learning_rate 1e-5 --optimizer adam --model_func pointconv --model_name test_training --model_path ~/models/sdf/ --log_path ~/logs/ --batch_size 8 --epochs 100 --training --train_path /dataspace/ReconstructionData/SDF_Full/Train/ --validation_path /dataspace/ReconstructionData/SDF_Full/Validation/ --sdf_count 256
```

Meshing example (assumes trained model `pointconv_mse` is present in the models path and the PCD Files are unzipped in `PCD_PATH`):
```
python main.py --model_func pointconv --log_path LOGS_PATH --model_path MODELS_PATH --model_name pointconv_mse --mesh --mesh_folder MESH_RESULTS_PATH --pcd_folder PCD_PATH
```

## References:

[1] Wu, Wenxuan, Zhongang Qi, and Li Fuxin. "PointConv: Deep Convolutional Networks on 3D Point Clouds." arXiv preprint arXiv:1811.07246 (2018).

[2] Joon Park, Jeong, et al. "DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.

[3] Mescheder, Lars, et al. "Occupancy Networks: Learning 3D Reconstruction in Function Space." arXiv preprint arXiv:1812.03828 (2018).

[4] Kappler, Daniel, Jeannette Bohg, and Stefan Schaal. "Leveraging big data for grasp planning." 2015 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2015.

## Contact:

Mark Van der Merwe: mark.vandermerwe@utah.edu
