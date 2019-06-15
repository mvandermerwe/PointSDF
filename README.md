# PointSDF

PointSDF novel, point cloud based, implicit surface reconstruction learning architecture based on recent advances in the Computer Vision community [1,2,3]. Our approach uses new advances in point cloud based learning [1] in an architecture designed similar to [2], but with the implicit surface representation introduced in [3]. Our approach allows for point cloud based, reconstructions. This is a part of ongoing research being done at the [LL4MA Lab](https://robot-learning.cs.utah.edu/) at the University of Utah.

## Proposed Architecture:

We seek to represent objects as an implicit surface by learning the Signed Distance Fields (SDF) of objects. The SDF of a point relative to a mesh is the closest distance from that query point in arbitrary space to the _surface_ of the mesh. A point inside the object has a negative SDF and a point outside the object has a positive. We seek to use a single deep neural network architecture to predict SDF values for arbitrary points in space relative to an object. We use point cloud embeddings in order to change the prediction to match the single, partial view of the object available to us. Each object surface is then implicitly defined as the points in space where the SDF is 0.

![alt text](https://github.com/mvandermerwe/PointSDF/raw/master/images/point_sdf_architecture.png)

Our network is designed to take in a point cloud of size Px3 and a set of query points in 3D space around the object of size Tx3. The point cloud is passed through 4 PointConv [1] embedding layers. Each query point is passed through 2 fully connected layers with dropout to get a Tx256 query point embedding. The point cloud embedding and query point embedding are then concatenated and passed through 8 fully connected layers with Batch Normalization applied to each layer. The embedding of point cloud and query points is additionally passed in at the 4th layer. This architecture is inspired by [2,3]. The final Tx1 output represents the SDF predictions for each query point, and is passed through a Tanh activation to get a prediction in [-1,1]. To fascilitate this, we scale all meshes and query points to lie roughly within a 1x1x1 cube. We implement this architecture using Tensorflow.

## Preliminary Qualitative Results:

We have a pretrained version of our architecture ready to run (this is not our final trained model, but shows the potential of our approach). The architecture is trained on 500 meshes from the Grasp Database [4], on 200 synthetically rendered views of each object (we will make our data generation pipeline publically available when we publish our final work). Below we show several views from our validation set of the true mesh, the point cloud input, and the reconstructed mesh (our network can be run at arbitrary resolution; the displayed meshes are generated at 128x128x128 resolution). Note: we used a slight variation on the proposed architecture on this run, where we perform the first 4 layers of the post embedding network on the query points (Tx3), then feed the join embedding from the first half of the network in at the 4th layer.

![alt text](https://github.com/mvandermerwe/PointSDF/raw/master/images/can_012.png)
![alt text](https://github.com/mvandermerwe/PointSDF/raw/master/images/cellphone_039.png)
![alt text](https://github.com/mvandermerwe/PointSDF/raw/master/images/jar_002.png)

## Usage:

### Dependencies:

Our code depends on the following python libraries:
 * Tensorflow (>=1.9.0)
 * numpy
 * mcubes
 * trimesh
 * matplotlib
 
We additionally rely on PointConv for our embedding layers. PointConv should be cloned from [here](https://github.com/DylanWusee/pointconv) and their instructions should be followed to compile their custom TF operations. Additionally, the environment variable `POINTCONV_HOME` should be set to point at the pointconv repository folder.

### Dataset:

We have a simple TFRecord-based dataset that holds point cloud, query point, SDF training examples in Train, Validation, and Test folds. The dataset should be uncompressed somewhere and the Train and Validation folders passed into the `--train_path` and `--validation_path` arguments to our script (see below). The dataset can be downloaded from [here](https://uofu.box.com/s/xz9nromkjick63kb4fhmx3routy2klsp) (~6.4GB).

### Usage:

For full list of options run:
```
python main.py -h
```
Not all options are necessary each run and have reasonable defaults.

If you have downloaded our dataset to some folder `data`, you can run inference to voxelization with our network using the following command (note, we pass a training folder, but don't use it for testing):
```
python main.py --model_func pointconv_bn --model_name pointconv_full --voxelize --train_path data/SDF_Full/Train/ --validation_path data/SDF_Full/Validation/
```
You can additionally pass `--mesh` to turn the voxelization into a mesh. Currently, voxelization and meshing are done at 32x32x32 resolution, for fast demonstration.

## References:

[1] Wu, Wenxuan, Zhongang Qi, and Li Fuxin. "PointConv: Deep Convolutional Networks on 3D Point Clouds." arXiv preprint arXiv:1811.07246 (2018).

[2] Joon Park, Jeong, et al. "DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.

[3] Mescheder, Lars, et al. "Occupancy Networks: Learning 3D Reconstruction in Function Space." arXiv preprint arXiv:1812.03828 (2018).

[4] Kappler, Daniel, Jeannette Bohg, and Stefan Schaal. "Leveraging big data for grasp planning." 2015 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2015.

## Contact:

Mark Van der Merwe: mark.vandermerwe@utah.edu
