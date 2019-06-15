# PointSDF

PointSDF novel, point cloud based, implicit surface reconstruction learning architecture based on recent advances in the Computer Vision community [1,2,3]. Our approach uses new advances in point cloud based learning [1] in an architecture designed similar to [2], but with the implicit surface representation introduced in [3]. Our approach allows for point cloud based, reconstructions. This is a part of ongoing research being done at the [LL4MA Lab](https://robot-learning.cs.utah.edu/) at the University of Utah.

## Architecture:

![alt text](https://github.com/mvandermerwe/PointSDF/raw/master/images/point_sdf_architecture.png)

## Preliminary Qualitative Results:

We have a pretrained version of our architecture ready to run (this is not our final trained model, but shows the potential of our approach). The architecture is trained on 500 meshes from the Grasp Database [4], on 200 synthetically rendered views of each object (we will make our data generation pipeline publically available when we publish our final work). Below we show several views from our validation set of the true mesh, the point cloud input, and the reconstructed mesh (our network can be run at arbitrary resolution; the displayed meshes are generated at 128x128x128 resolution). Note: we used a slight variation on the proposed architecture on this run, where we perform the first 4 layers of the post embedding network on the query points (Tx3), then feed the join embedding from the first half of the network in at the 4th layer.

![alt text](https://github.com/mvandermerwe/PointSDF/raw/master/images/can_012.png)
![alt text](https://github.com/mvandermerwe/PointSDF/raw/master/images/cellphone_039.png)
![alt text](https://github.com/mvandermerwe/PointSDF/raw/master/images/jar_002.png)

## Usage:

### Dependencies:

### Usage:

## References:

[1] Wu, Wenxuan, Zhongang Qi, and Li Fuxin. "PointConv: Deep Convolutional Networks on 3D Point Clouds." arXiv preprint arXiv:1811.07246 (2018).

[2] Joon Park, Jeong, et al. "DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.

[3] Mescheder, Lars, et al. "Occupancy Networks: Learning 3D Reconstruction in Function Space." arXiv preprint arXiv:1812.03828 (2018).

[4] Kappler, Daniel, Jeannette Bohg, and Stefan Schaal. "Leveraging big data for grasp planning." 2015 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2015.

## Contact:

Mark Van der Merwe: mark.vandermerwe@utah.edu
