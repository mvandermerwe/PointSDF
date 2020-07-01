from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys
import pdb
# import mcubes
# import trimesh
import os
import argparse

from sdf_dataset import get_sdf_dataset, get_point_clouds, get_voxels
from helper import get_num_trainable_variables, shuffle_in_unison, get_bn_decay, get_learning_rate
from visualization import plot_3d_points, plot_voxel, convert_to_sparse_voxel_grid

def run_sdf(get_model, train_path, validation_path, model_path, logs_path, batch_size=32, epoch_start=0, epochs=100, learning_rate=1e-4, optimizer='adam', train=True, warm_start=False, sdf_count=64, voxel=False):

    # Read in training and validation files.
    train_files = [os.path.join(train_path, filename) for filename in os.listdir(train_path) if ".tfrecord" in filename]
    validation_files = [os.path.join(validation_path, filename) for filename in os.listdir(validation_path) if ".tfrecord" in filename]
    
    # Fetch the data.
    train_dataset = get_sdf_dataset(train_files, batch_size=batch_size, sdf_count=sdf_count)
    validation_dataset = get_sdf_dataset(validation_files, batch_size=batch_size, sdf_count=sdf_count)

    # Setup iterators.
    train_iterator = train_dataset.make_initializable_iterator()
    train_next_point_cloud, train_next_xyz, train_next_label = train_iterator.get_next()

    val_iterator = validation_dataset.make_initializable_iterator()
    val_next_point_cloud, val_next_xyz, val_next_label = val_iterator.get_next()

    # Setup optimizer.
    batch = tf.get_variable('batch', [], initializer=tf.constant_initializer(0), trainable=False)
    learn_rate = get_learning_rate(batch, batch_size, learning_rate)
    tf.summary.scalar('learning_rate', learn_rate)
    
    if optimizer == 'adam':
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif optimizer == 'momentum':
        opt = tf.train.MomentumOptimizer(learning_rate=learn_rate, momentum=0.9) # Make this another hyperparam?

    # Setup batch norm decay rate for PointConv/Net layers.
    bn_decay = get_bn_decay(batch, batch_size)

    # Setup model operations.
    points = tf.placeholder(tf.float32, name="point_cloud")
    xyz_in = tf.placeholder(tf.float32, name="query_points")
    sdf_labels = tf.placeholder(tf.float32, name="query_labels")
    is_training = tf.placeholder(tf.bool, name="is_training")
    sdf_prediction, loss, debug = get_model(points, xyz_in, sdf_labels, is_training, bn_decay, batch_size=batch_size)

    # Get update ops for the BN.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    # Setup training operation.
    with tf.control_dependencies(update_ops):
        train_op = opt.minimize(loss, global_step=batch)

    # Setup tensorboard operation.
    merged = tf.summary.merge_all()
    
    print("Variable Counts: ")
    print("Encoder: " + str(get_num_trainable_variables('encoder')))
    print("SDF: " + str(get_num_trainable_variables('sdf')))
    init = tf.global_variables_initializer()
    
    # Save/Restore model.
    saver = tf.train.Saver()

    with tf.Session() as sess:

        # Setup tensorboard.
        f_writer = tf.summary.FileWriter(logs_path, sess.graph)

        # Init variables.
        if not train or warm_start:
            model_file = os.path.join(model_path, 'model.ckpt')
            saver.restore(sess, model_file)
            print("Model restored from: ", model_file)
        else:
            sess.run(init)
        
        validation_loss = float('inf')
        best_loss = float('inf')

        for epoch in range(epoch_start, epoch_start + epochs):
            print("Epoch: ", str(epoch))

            sess.run(train_iterator.initializer)

            # Track loss throughout updates.
            total_loss = 0.0
            examples = 0
            while True:
                try:
                    # Split the given features into batches.
                    point_clouds_, xyzs_, labels_ = sess.run((train_next_point_cloud, train_next_xyz, train_next_label))
                    
                    examples += 1
                    if train:
                        _, summary_, sdf_prediction_, loss_, step, _ = sess.run([train_op, merged, sdf_prediction, loss, batch, debug], feed_dict = {
                            points: point_clouds_, xyz_in: xyzs_, sdf_labels: labels_, is_training: True,
                        })
                        f_writer.add_summary(summary_, step)
                    else:
                        sdf_prediction_, loss_ = sess.run([sdf_prediction, loss], feed_dict = {
                            points: point_clouds_, xyz_in: xyzs_, sdf_labels: labels_, is_training: False,
                        })

                        pts = np.reshape(xyzs_[0], (sdf_count, 3))
                        truth = np.reshape(labels_[0], (sdf_count))
                        pred = np.reshape(sdf_prediction_[0], (sdf_count))

                        plot_3d_points(point_clouds_[0])
                        plot_3d_points(pts)
                        plot_3d_points(pts, signed_distances=truth)
                        plot_3d_points(pts, signed_distances=pred)

                    total_loss += loss_

                except tf.errors.OutOfRangeError:
                    break

            avg_loss = total_loss / float(examples)

            print(avg_loss)

            if train:
                f_writer.add_summary(tf.Summary(
                    value=[
                        tf.Summary.Value(tag='epoch_loss', simple_value=avg_loss)
                    ]), epoch)

            # Validation loop every epoch.
            sess.run(val_iterator.initializer)
            
            total_loss = 0.0
            examples = 0
            while True:
                try:
                    point_clouds_, xyzs_, labels_ = sess.run((val_next_point_cloud, val_next_xyz, val_next_label))
                    
                    examples += 1
                    sdf_prediction_, loss_ = sess.run([sdf_prediction, loss], feed_dict = {
                        points: point_clouds_, xyz_in: xyzs_, sdf_labels: labels_, is_training: False,
                    })

                    total_loss += loss_

                except tf.errors.OutOfRangeError:
                    break

            avg_loss = total_loss / float(examples)

            # Save model if it's the best one we've seen.
            if avg_loss < best_loss and train:
                best_loss = avg_loss
                save_path = saver.save(sess, os.path.join(model_path, 'model.ckpt'))
                print("Model saved to: %s" % save_path)

            if train:
                f_writer.add_summary(tf.Summary(
                    value=[
                        tf.Summary.Value(tag='epoch_validation_loss', simple_value=avg_loss)
                    ]), epoch)
