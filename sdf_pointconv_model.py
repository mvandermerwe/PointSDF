# Create PointNet2 + DeepSDF Estimator.

from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys
import pdb
import mcubes
import os

sys.path.append(os.environ['POINTCONV_HOME'])
from PointConv import feature_encoding_layer

from helper import get_bn_decay

def get_pointconv_deep_model(points, xyz, sdf_label, is_training, bn_decay, batch_size=32, loss_feature='loss', alpha=0.5, loss_function='mse', sdf_count=64):
    '''
    Given features and label return prediction, loss ops.
    '''

    # Get inputs from our features map.
    l0_xyz = tf.reshape(points, shape=(batch_size, -1, 3))
    l0_points = None
    xyz_in = tf.reshape(xyz, shape=(batch_size, sdf_count, 3))
    sdf_label = tf.reshape(sdf_label, shape=(batch_size, sdf_count, 1)) # This is important.

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
        cloud_embedding = tf.tile(tf.expand_dims(cloud_embedding,1), [1, sdf_count, 1])
        embedded_inputs = tf.concat([pts_embedding, cloud_embedding], axis=2)

        # 8 Dense layers w/ ReLU non-linearities to predict SDF.
        l1_sdf = tf.layers.Dense(512, activation=tf.nn.relu, use_bias=True)(embedded_inputs)
        l1_sdf = tf.layers.dropout(l1_sdf, rate=0.2, training=is_training)
        l2_sdf = tf.layers.Dense(512, activation=tf.nn.relu, use_bias=True)(l1_sdf)
        l2_sdf = tf.layers.dropout(l2_sdf, rate=0.2, training=is_training)        
        l3_sdf = tf.layers.Dense(256, activation=tf.nn.relu, use_bias=True)(l2_sdf)
        l3_sdf = tf.layers.dropout(l3_sdf, rate=0.2, training=is_training)        

        # Feed our input embedding space back in here.
        l3_sdf_aug = tf.concat([l3_sdf, embedded_inputs], axis=2)
        l4_sdf = tf.layers.Dense(512, activation=tf.nn.relu, use_bias=True)(l3_sdf_aug)
        l4_sdf = tf.layers.dropout(l4_sdf, rate=0.2, training=is_training)

        l5_sdf = tf.layers.Dense(512, activation=tf.nn.relu, use_bias=True)(l4_sdf)
        l5_sdf = tf.layers.dropout(l5_sdf, rate=0.2, training=is_training)        
        l6_sdf = tf.layers.Dense(512, activation=tf.nn.relu, use_bias=True)(l5_sdf)
        l6_sdf = tf.layers.dropout(l6_sdf, rate=0.2, training=is_training)        
        l7_sdf = tf.layers.Dense(512, activation=tf.nn.relu, use_bias=True)(l6_sdf)
        l7_sdf = tf.layers.dropout(l7_sdf, rate=0.2, training=is_training)

        sdf_prediction = tf.layers.Dense(1, activation=tf.nn.tanh, use_bias=True)(l7_sdf) # Last is tanh

    # Define the loss:
    loss = tf.losses.mean_squared_error(sdf_label, sdf_prediction)
    tf.summary.scalar(loss_feature, loss)

    # Collect debug print statements as needed.
    debug = tf.no_op()
    
    return sdf_prediction, loss, debug

def get_pointconv_deep_bn_model(points, xyz, sdf_label, is_training, bn_decay, batch_size=32, loss_feature='loss', alpha=0.5, loss_function='mse', sdf_count=64):
    '''
    Given features and label return prediction, loss ops.
    '''

    # Get inputs from our features map.
    l0_xyz = tf.reshape(points, shape=(batch_size, -1, 3))
    l0_points = None
    xyz_in = tf.reshape(xyz, shape=(batch_size, sdf_count, 3))
    sdf_label = tf.reshape(sdf_label, shape=(batch_size, sdf_count, 1)) # This is important.

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
        cloud_embedding = tf.tile(tf.expand_dims(cloud_embedding,1), [1, sdf_count, 1])
        embedded_inputs = tf.concat([pts_embedding, cloud_embedding], axis=2)

        # 8 Dense layers w/ ReLU non-linearities to predict SDF.
        l1_sdf = tf.layers.Dense(512)(xyz_in)
        l1_sdf = tf.layers.batch_normalization(l1_sdf, training=is_training)
        l1_sdf = tf.nn.relu(l1_sdf)

        l2_sdf = tf.layers.Dense(512)(l1_sdf)
        l2_sdf = tf.layers.batch_normalization(l2_sdf, training=is_training)
        l2_sdf = tf.nn.relu(l2_sdf)

        l3_sdf = tf.layers.Dense(509)(l2_sdf)
        l3_sdf = tf.layers.batch_normalization(l3_sdf, training=is_training)
        l3_sdf = tf.nn.relu(l3_sdf)

        # Feed our input embedding space back in here.
        l3_sdf_aug = tf.concat([l3_sdf, embedded_inputs], axis=2)
        l4_sdf = tf.layers.Dense(512)(l3_sdf_aug)
        l4_sdf = tf.layers.batch_normalization(l4_sdf, training=is_training)
        l4_sdf = tf.nn.relu(l4_sdf)

        l5_sdf = tf.layers.Dense(512)(l4_sdf)
        l5_sdf = tf.layers.batch_normalization(l5_sdf, training=is_training)
        l5_sdf = tf.nn.relu(l5_sdf)

        l6_sdf = tf.layers.Dense(512)(l5_sdf)
        l6_sdf = tf.layers.batch_normalization(l6_sdf, training=is_training)
        l6_sdf = tf.nn.relu(l6_sdf)

        l7_sdf = tf.layers.Dense(512)(l6_sdf)
        l7_sdf = tf.layers.batch_normalization(l7_sdf, training=is_training)
        l7_sdf = tf.nn.relu(l7_sdf)

        sdf_prediction = tf.layers.Dense(1, activation=tf.nn.tanh, use_bias=True)(l7_sdf) # Last is tanh

    # Define the loss:
    loss = tf.losses.mean_squared_error(sdf_label, sdf_prediction)
    tf.summary.scalar(loss_feature, loss)

    # Collect debug print statements as needed.
    debug = tf.no_op()
    
    return sdf_prediction, loss, debug
