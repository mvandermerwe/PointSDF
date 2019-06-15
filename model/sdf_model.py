# Create SDF Model that memorizes a single shape view using only the FC layers.
# This is to test the representational power of the FC layers of the network.

from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys
import pdb
import mcubes

def get_fc_model(points, xyz, sdf_label, voxel_label, is_training, bn_decay, batch_size=32, alpha=0.5, loss_feature='loss', loss_function='mse'):
    '''
    Just the FC Layers for predicting SDF on points.
    '''

    # Get inputs from our features map.
    xyz_in = tf.reshape(xyz, shape=(batch_size, 3))
    sdf_label = tf.reshape(sdf_label, shape=(batch_size, 1))
    
    with tf.variable_scope('sdf'):
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
        l3_sdf_aug = tf.concat([l3_sdf, xyz_in], axis=1)
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
        # b = tf.print(l7_sdf)

        sdf_prediction = tf.layers.Dense(1, activation=tf.nn.tanh, use_bias=True)(l7_sdf) # Last is tanh

    # Define the loss: MSE.
    a = tf.print(tf.shape(sdf_prediction))
    b = tf.print(tf.shape(sdf_label))
    loss = tf.losses.mean_squared_error(sdf_label, sdf_prediction)
    tf.summary.scalar(loss_feature, loss)
    
    return sdf_prediction, None, loss, tf.group([a,b])

def get_fc_small_model(points, xyz, sdf_label, voxel_label, is_training, bn_decay, batch_size=32, alpha=0.5, loss_feature='loss'):
    '''
    Just the FC Layers for predicting SDF on points.
    '''

    # Get inputs from our features map.
    xyz_in = tf.reshape(xyz, shape=(batch_size, 3))

    with tf.variable_scope('sdf'):

        l1_sdf = tf.layers.Dense(512)(xyz_in)
        l1_sdf = tf.layers.batch_normalization(l1_sdf, training=is_training)
        l1_sdf = tf.nn.relu(l1_sdf)
        
        l2_sdf = tf.layers.Dense(512)(l1_sdf)
        l2_sdf = tf.layers.batch_normalization(l2_sdf, training=is_training)
        l2_sdf = tf.nn.relu(l2_sdf)

        l3_sdf = tf.layers.Dense(256)(l2_sdf)
        l3_sdf = tf.layers.batch_normalization(l3_sdf, training=is_training)
        l3_sdf = tf.nn.relu(l3_sdf)

        sdf_prediction = tf.layers.Dense(1, activation=tf.nn.tanh, use_bias=True)(l3_sdf)

    # Define the loss: MSE.
    loss = tf.losses.mean_squared_error(sdf_label, sdf_prediction)
    tf.summary.scalar(loss_feature, loss)
    
    return sdf_prediction, None, loss

def get_fc_no_bn_small_model(points, xyz, sdf_label, voxel_label, is_training, bn_decay, batch_size=32, alpha=0.5, loss_feature='loss'):
    '''
    Just the FC Layers for predicting SDF on points.
    '''

    # Get inputs from our features map.
    xyz_in = tf.reshape(xyz, shape=(batch_size, 3))

    with tf.variable_scope('sdf'):

        l1_sdf = tf.layers.Dense(512, activation=tf.nn.relu, use_bias=True)(xyz_in)
        l1_sdf = tf.layers.dropout(l1_sdf, rate=0.2, training=is_training)
        
        l2_sdf = tf.layers.Dense(512, activation=tf.nn.relu, use_bias=True)(l1_sdf)
        l2_sdf = tf.layers.dropout(l2_sdf, rate=0.2, training=is_training)

        l3_sdf = tf.layers.Dense(256, activation=tf.nn.relu, use_bias=True)(l2_sdf)
        l3_sdf = tf.layers.dropout(l3_sdf, rate=0.2, training=is_training)

        sdf_prediction = tf.layers.Dense(1, activation=tf.nn.tanh, use_bias=True)(l3_sdf)

    # Define the loss: MSE.
    loss = tf.losses.mean_squared_error(sdf_label, sdf_prediction)
    tf.summary.scalar(loss_feature, loss)
    
    return sdf_prediction, None, loss

def get_fc_no_bn_model(points, xyz, sdf_label, voxel_label, is_training, bn_decay, batch_size=32, alpha=0.5, loss_feature='loss'):
    '''
    Just the FC Layers for predicting SDF on points.
    '''

    # Get inputs from our features map.
    xyz_in = tf.reshape(xyz, shape=(batch_size, 3))

    with tf.variable_scope('sdf'):
        # 8 Dense layers w/ ReLU non-linearities to predict SDF.
        l1_sdf = tf.layers.Dense(512, activation=tf.nn.relu, use_bias=True)(xyz_in)
        l2_sdf = tf.layers.Dense(512, activation=tf.nn.relu, use_bias=True)(l1_sdf)
        l3_sdf = tf.layers.Dense(509, activation=tf.nn.relu, use_bias=True)(l2_sdf)

        # Feed our input embedding space back in here.
        l3_sdf_aug = tf.concat([l3_sdf, xyz_in], axis=1)
        l4_sdf = tf.layers.Dense(512, activation=tf.nn.relu, use_bias=True)(l3_sdf_aug)

        l5_sdf = tf.layers.Dense(512, activation=tf.nn.relu, use_bias=True)(l4_sdf)
        l6_sdf = tf.layers.Dense(512, activation=tf.nn.relu, use_bias=True)(l5_sdf)
        l7_sdf = tf.layers.Dense(512, activation=tf.nn.relu, use_bias=True)(l5_sdf)

        sdf_prediction = tf.layers.Dense(1, activation=tf.nn.tanh, use_bias=True)(l7_sdf) # Last is tanh

    # Define the loss: MSE.
    loss = tf.losses.mean_squared_error(sdf_label, sdf_prediction)
    tf.summary.scalar(loss_feature, loss)
    
    return sdf_prediction, None, loss

def get_fc_no_bn_dropout_model(points, xyz, sdf_label, voxel_label, is_training, bn_decay, loss_function, batch_size=32, alpha=0.5, loss_feature='loss'):
    '''
    Just the FC Layers for predicting SDF on points.
    '''

    # Get inputs from our features map.
    xyz_in = tf.reshape(xyz, shape=(batch_size, 3))
    sdf_label = tf.reshape(sdf_label, shape=(batch_size, 1))

    with tf.variable_scope('sdf'):
        # 8 Dense layers w/ ReLU non-linearities to predict SDF.
        l1_sdf = tf.layers.Dense(512, activation=tf.nn.relu, use_bias=True)(xyz_in)
        l1_sdf = tf.layers.dropout(l1_sdf, rate=0.2, training=is_training)
        l2_sdf = tf.layers.Dense(512, activation=tf.nn.relu, use_bias=True)(l1_sdf)
        l2_sdf = tf.layers.dropout(l2_sdf, rate=0.2, training=is_training)        
        l3_sdf = tf.layers.Dense(509, activation=tf.nn.relu, use_bias=True)(l2_sdf)
        l3_sdf = tf.layers.dropout(l3_sdf, rate=0.2, training=is_training)        

        # Feed our input embedding space back in here.
        l3_sdf_aug = tf.concat([l3_sdf, xyz_in], axis=1)
        l4_sdf = tf.layers.Dense(512, activation=tf.nn.relu, use_bias=True)(l3_sdf_aug)
        l4_sdf = tf.layers.dropout(l4_sdf, rate=0.2, training=is_training)

        l5_sdf = tf.layers.Dense(512, activation=tf.nn.relu, use_bias=True)(l4_sdf)
        l5_sdf = tf.layers.dropout(l5_sdf, rate=0.2, training=is_training)        
        l6_sdf = tf.layers.Dense(512, activation=tf.nn.relu, use_bias=True)(l5_sdf)
        l6_sdf = tf.layers.dropout(l6_sdf, rate=0.2, training=is_training)        
        l7_sdf = tf.layers.Dense(512, activation=tf.nn.relu, use_bias=True)(l5_sdf)
        l7_sdf = tf.layers.dropout(l7_sdf, rate=0.2, training=is_training)

        sdf_prediction = tf.layers.Dense(1, activation=tf.nn.tanh, use_bias=True)(l7_sdf) # Last is tanh

    # b = tf.print(tf.shape(sdf_prediction))
    # b_ = tf.print(tf.shape(sdf_label))
        
    # Define the loss: MSE.
    if loss_function == 'mse':
        loss = tf.losses.mean_squared_error(sdf_label, sdf_prediction)
    elif loss_function == 'norm_mse':
        diff = tf.subtract(sdf_label, sdf_prediction)
        loss = tf.divide(tf.square(diff), tf.square(sdf_label))
        loss = tf.reduce_mean(loss)
    elif loss_function == 'l1':
        loss = tf.losses.absolute_difference(sdf_label, sdf_prediction)
    elif loss_function == 'l1_clip':
        loss = tf.losses.absolute_difference(
            tf.clip_by_value(sdf_label, -0.1, 0.1),
            tf.clip_by_value(sdf_prediction, -0.1, 0.1))
    tf.summary.scalar(loss_feature, loss)
    
    return sdf_prediction, None, loss, None
