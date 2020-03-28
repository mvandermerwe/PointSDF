# Create PointConv + DeepSDF Estimator.

from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys
import pdb
#import mcubes
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2

sys.path.append(os.environ['POINTCONV_HOME'])
from PointConv import feature_encoding_layer

#from helper import get_bn_decay

def get_pointconv_model(points, xyz, sdf_label, is_training, bn_decay, batch_size=32, loss_feature='loss'):
    '''
    Given features and label return prediction, loss ops.
    '''

    # Get inputs from our features map.
    l0_xyz = tf.reshape(points, shape=(batch_size, -1, 3))
    l0_points = None
    xyz_in = tf.reshape(xyz, shape=(batch_size, -1, 3))
    sdf_label = tf.reshape(sdf_label, shape=(batch_size, -1, 1)) # This is important.

    with tf.variable_scope('points_embedding'):

        # Embed our input points to some 256 vector.
        l1_pts = tf.layers.Dense(512, activation=tf.nn.relu, use_bias=True)(xyz_in)
        l1_pts = tf.layers.dropout(l1_pts, rate=0.2, training=is_training)

        pts_embedding = tf.layers.Dense(256, activation=tf.nn.relu, use_bias=True)(l1_pts)
        pts_embedding = tf.layers.dropout(pts_embedding, rate=0.2, training=is_training)
    
    with tf.variable_scope('encoder'):

        # Encode w/ PointConv Layers.
        l1_xyz, l1_points = feature_encoding_layer(l0_xyz, l0_xyz, npoint=512, radius=0.1, sigma=0.05, K=32, mlp=[32,32,64], is_training=is_training, bn_decay=bn_decay, weight_decay=None, scope='layer1')
        l2_xyz, l2_points = feature_encoding_layer(l1_xyz, l1_points, npoint=256, radius=0.2, sigma=0.1, K=32, mlp=[64,64,64], is_training=is_training, bn_decay=bn_decay, weight_decay=None, scope='layer2')
        l3_xyz, l3_points = feature_encoding_layer(l2_xyz, l2_points, npoint=64, radius=0.4, sigma=0.2, K=32, mlp=[128,128,256], is_training=is_training, bn_decay=bn_decay, weight_decay=None, scope='layer3')
        l4_xyz, l4_points = feature_encoding_layer(l3_xyz, l3_points, npoint=36, radius=0.8, sigma=0.4, K=32, mlp=[256,256,512], is_training=is_training, bn_decay=bn_decay, weight_decay=None, scope='layer4')

        # Fully connected layers
        embedding = tf.reshape(l4_points, [batch_size, -1])

        # Encode to a 256 large embedding vector.
        cloud_embedding = tf.layers.Dense(256)(embedding)
        cloud_embedding = tf.layers.batch_normalization(cloud_embedding, training=is_training)
        cloud_embedding = tf.nn.relu(cloud_embedding)

    with tf.variable_scope('sdf'):

        # Combine embeddings. First reshape cloud embeddings to concat with each pt embedding.
        cloud_embedding = tf.tile(tf.expand_dims(cloud_embedding,1), [1, tf.shape(pts_embedding)[1], 1])
        embedded_inputs = tf.concat([pts_embedding, cloud_embedding], axis=2)

        # 8 Dense layers w/ ReLU non-linearities to predict SDF.
        l1_sdf = tf.layers.Dense(512, name='sdf_1')(embedded_inputs)
        l1_sdf_1 = tf.layers.batch_normalization(l1_sdf, training=is_training)
        l1_sdf_2 = tf.nn.relu(l1_sdf_1)

        l2_sdf = tf.layers.Dense(512, name='sdf_2')(l1_sdf_2)
        l2_sdf_1 = tf.layers.batch_normalization(l2_sdf, training=is_training)
        l2_sdf_2 = tf.nn.relu(l2_sdf_1)

        l3_sdf = tf.layers.Dense(256, name='sdf_3')(l2_sdf_2)
        l3_sdf_1 = tf.layers.batch_normalization(l3_sdf, training=is_training)
        l3_sdf_2 = tf.nn.relu(l3_sdf_1)

        # Feed our input embedding space back in here.
        l3_sdf_aug = tf.concat([l3_sdf_2, embedded_inputs], axis=2)
        l4_sdf = tf.layers.Dense(512, name='sdf_4')(l3_sdf_aug)
        l4_sdf_1 = tf.layers.batch_normalization(l4_sdf, training=is_training)
        l4_sdf_2 = tf.nn.relu(l4_sdf_1)

        l5_sdf = tf.layers.Dense(512, name='sdf_5')(l4_sdf_2)
        l5_sdf_1 = tf.layers.batch_normalization(l5_sdf, training=is_training)
        l5_sdf_2 = tf.nn.relu(l5_sdf_1)

        l6_sdf = tf.layers.Dense(512, name='sdf_6')(l5_sdf_2)
        l6_sdf_1 = tf.layers.batch_normalization(l6_sdf, training=is_training)
        l6_sdf_2 = tf.nn.relu(l6_sdf_1)

        l7_sdf = tf.layers.Dense(512, name='sdf_7')(l6_sdf_2)
        l7_sdf_1 = tf.layers.batch_normalization(l7_sdf, training=is_training)
        l7_sdf_2 = tf.nn.relu(l7_sdf_1)

        sdf_prediction = tf.layers.Dense(1, activation=tf.nn.tanh, use_bias=True, name='sdf_8')(l7_sdf_2) # Last is tanh

    # Define the loss: clipped surface loss.
    # loss = tf.losses.absolute_difference(
    #     tf.clip_by_value(sdf_label, -0.1, 0.1),
    #     tf.clip_by_value(sdf_prediction, -0.1, 0.1)) 
    loss = tf.losses.mean_squared_error(sdf_label, sdf_prediction)
    tf.summary.scalar(loss_feature, loss)

    # Collect debug print statements as needed.
    debug = tf.no_op()
    
    return sdf_prediction, loss, debug

def get_sdf_model(cloud_embedding, xyz, sdf_label, is_training, bn_decay, batch_size=32, loss_feature='loss'):
    '''
    Given features and label return prediction, loss ops. Make savable version to run in C++. That is, we remove the cloud embedding w/ PointConv since that 
    will be difficult due to PointConv layers.
    '''

    # Get inputs from our features map.
    # l0_xyz = tf.reshape(points, shape=(batch_size, -1, 3))
    # l0_points = None
    cloud_embedding = tf.reshape(cloud_embedding, shape=(batch_size, 256))
    xyz_in = tf.reshape(xyz, shape=(batch_size, -1, 3))
    sdf_label = tf.reshape(sdf_label, shape=(batch_size, -1, 1)) # This is important.

    with tf.variable_scope('points_embedding'):
        # Embed our input points to some 256 vector.

        l1_pts = tf.layers.Dense(512, activation=tf.nn.relu, use_bias=True)(xyz_in)
        l1_pts = tf.layers.dropout(l1_pts, rate=0.2, training=is_training)

        pts_embedding = tf.layers.Dense(256, activation=tf.nn.relu, use_bias=True)(l1_pts)
        pts_embedding = tf.layers.dropout(pts_embedding, rate=0.2, training=is_training)

    with tf.variable_scope('sdf'):

        # Combine embeddings. First reshape cloud embeddings to concat with each pt embedding.
        cloud_embedding = tf.tile(tf.expand_dims(cloud_embedding,1), [1, tf.shape(pts_embedding)[1], 1])
        embedded_inputs = tf.concat([pts_embedding, cloud_embedding], axis=2)

        # 8 Dense layers w/ ReLU non-linearities to predict SDF.
        l1_sdf = tf.layers.Dense(512, name='sdf_1')(embedded_inputs)
        l1_sdf_1 = tf.layers.batch_normalization(l1_sdf, training=is_training)
        l1_sdf_2 = tf.nn.relu(l1_sdf_1)

        l2_sdf = tf.layers.Dense(512, name='sdf_2')(l1_sdf_2)
        l2_sdf_1 = tf.layers.batch_normalization(l2_sdf, training=is_training)
        l2_sdf_2 = tf.nn.relu(l2_sdf_1)

        l3_sdf = tf.layers.Dense(256, name='sdf_3')(l2_sdf_2)
        l3_sdf_1 = tf.layers.batch_normalization(l3_sdf, training=is_training)
        l3_sdf_2 = tf.nn.relu(l3_sdf_1)

        # Feed our input embedding space back in here.
        l3_sdf_aug = tf.concat([l3_sdf_2, embedded_inputs], axis=2)
        l4_sdf = tf.layers.Dense(512, name='sdf_4')(l3_sdf_aug)
        l4_sdf_1 = tf.layers.batch_normalization(l4_sdf, training=is_training)
        l4_sdf_2 = tf.nn.relu(l4_sdf_1)

        l5_sdf = tf.layers.Dense(512, name='sdf_5')(l4_sdf_2)
        l5_sdf_1 = tf.layers.batch_normalization(l5_sdf, training=is_training)
        l5_sdf_2 = tf.nn.relu(l5_sdf_1)

        l6_sdf = tf.layers.Dense(512, name='sdf_6')(l5_sdf_2)
        l6_sdf_1 = tf.layers.batch_normalization(l6_sdf, training=is_training)
        l6_sdf_2 = tf.nn.relu(l6_sdf_1)

        l7_sdf = tf.layers.Dense(512, name='sdf_7')(l6_sdf_2)
        l7_sdf_1 = tf.layers.batch_normalization(l7_sdf, training=is_training)
        l7_sdf_2 = tf.nn.relu(l7_sdf_1)

        sdf_prediction = tf.layers.Dense(1, activation=tf.nn.tanh, use_bias=True, name='sdf_8')(l7_sdf_2) # Last is tanh

    # Define the loss:
    loss = tf.losses.mean_squared_error(sdf_label, sdf_prediction)
    tf.summary.scalar(loss_feature, loss)

    # Collect debug print statements as needed.
    debug = tf.no_op()
    
    return sdf_prediction, loss, debug

def get_embedding_model(points, is_training, bn_decay, batch_size=1):

    # Get inputs from our features map.
    l0_xyz = tf.reshape(points, shape=(batch_size, -1, 3))
    l0_points = None
    
    with tf.variable_scope('encoder'):

        # Encode w/ PointConv Layers.
        l1_xyz, l1_points = feature_encoding_layer(l0_xyz, l0_xyz, npoint=512, radius=0.1, sigma=0.05, K=32, mlp=[32,32,64], is_training=is_training, bn_decay=bn_decay, weight_decay=None, scope='layer1')
        l2_xyz, l2_points = feature_encoding_layer(l1_xyz, l1_points, npoint=256, radius=0.2, sigma=0.1, K=32, mlp=[64,64,64], is_training=is_training, bn_decay=bn_decay, weight_decay=None, scope='layer2')
        l3_xyz, l3_points = feature_encoding_layer(l2_xyz, l2_points, npoint=64, radius=0.4, sigma=0.2, K=32, mlp=[128,128,256], is_training=is_training, bn_decay=bn_decay, weight_decay=None, scope='layer3')
        l4_xyz, l4_points = feature_encoding_layer(l3_xyz, l3_points, npoint=36, radius=0.8, sigma=0.4, K=32, mlp=[256,256,512], is_training=is_training, bn_decay=bn_decay, weight_decay=None, scope='layer4')

        # Fully connected layers
        embedding = tf.reshape(l4_points, [batch_size, -1])

        # Encode to a 256 large embedding vector.
        cloud_embedding = tf.layers.Dense(256)(embedding)
        cloud_embedding = tf.layers.batch_normalization(cloud_embedding, training=is_training)
        cloud_embedding = tf.nn.relu(cloud_embedding)

    return cloud_embedding

def get_sdf_prediction(get_model, model_path):

    # Setup model operations.
    points = tf.placeholder(tf.float32)
    # cloud = tf.placeholder(tf.float32)
    xyz_in = tf.placeholder(tf.float32)    
    sdf_labels = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)

    sdf_prediction, loss, _ = get_model(points, xyz_in, sdf_labels, is_training, None, batch_size=1)

    # Save/Restore model.
    saver = tf.train.Saver()

    sess = tf.Session(config=config)
    saver.restore(sess, os.path.join(model_path, 'model.ckpt'))

    # Get embedding tensor.
    embedding = tf.get_default_graph().get_tensor_by_name("encoder/Relu:0")

    # SDF gradient:
    points_gradient = tf.gradients(sdf_prediction, xyz_in)

    # Setup function that predicts SDF for (x,y,z) given a point cloud.
    def get_sdf(pt_cloud, query_pts):
        prediction = sess.run(sdf_prediction, feed_dict = {
            points: pt_cloud, xyz_in: query_pts, sdf_labels: None, is_training: False,
        })
        return prediction

    def get_embedding(point_cloud):
        cloud_embedding = sess.run(embedding, feed_dict = {
            points: point_cloud, xyz_in: None, sdf_labels: None, is_training: False,
        })
        return cloud_embedding

    def get_sdf_gradient(pt_cloud, query_pts):
        # Does both sdf and gradient.
        prediction, gradient = sess.run([sdf_prediction, points_gradient], feed_dict = {
            points: pt_cloud, xyz_in: query_pts, sdf_labels: None, is_training: False,
        })
        return prediction, gradient

    return get_sdf, get_embedding, get_sdf_gradient
