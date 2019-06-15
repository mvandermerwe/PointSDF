from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys
import pdb
import mcubes
import trimesh
import os
import argparse

from sdf_dataset import get_sdf_dataset, get_point_clouds, get_voxels
from helper import get_num_trainable_variables, shuffle_in_unison, get_bn_decay, get_learning_rate
from visualization import plot_3d_points, plot_voxel, convert_to_sparse_voxel_grid

def run(get_model, train_path, validation_path, pc_h5_file, model_path, logs_path, batch_size=32, epoch_start=0, epochs=100, learning_rate=1e-4, optimizer='adam', train=True, warm_start=False, alpha=0.5, loss_function='mse', sdf_count=64):

    # Read in training and validation files.
    train_files = [os.path.join(train_path, filename) for filename in os.listdir(train_path) if ".tfrecord" in filename]
    #train_files = [os.path.join(train_path, 'train_0.tfrecord')]
    #train_files = [os.path.join(train_path, 'donut_poisson_000_0.tfrecord')]
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
    points = tf.placeholder(tf.float32)
    xyz_in = tf.placeholder(tf.float32)    
    sdf_labels = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
    sdf_prediction, loss, debug = get_model(points, xyz_in, sdf_labels, is_training, bn_decay, batch_size=batch_size, alpha=alpha, loss_function=loss_function, sdf_count=sdf_count)

    # Get update ops for the BN.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    # Setup training operation.
    with tf.control_dependencies(update_ops):
        train_op = opt.minimize(loss, global_step=batch)
    
    #train_op = tf.group([train_op, update_ops])

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

                        plot_3d_points(pts, truth)
                        plot_3d_points(pts, pred)

                    total_loss += loss_

                except tf.errors.OutOfRangeError:
                    break

            avg_loss = total_loss / float(examples)
            
            # if not train: #or epoch % 20 == 0:
            #     pts_ = np.reshape(np.concatenate(pts, axis=1), (-1,3))
            #     sdf_predictions_ = np.reshape(np.concatenate(predictions, axis=1), (-1,))
            #     true = np.reshape(np.concatenate(true, axis=1), (-1,))
            #     plot_3d_points(pts_, true)
            #     plot_3d_points(pts_, sdf_predictions_)

            print(avg_loss)

            # Save model if it's the best one we've seen.
            # if avg_loss < best_loss and train:
            #     best_loss = avg_loss
            #     save_path = saver.save(sess, os.path.join(model_path, 'model.ckpt'))
            #     print("Model saved to: %s" % save_path)

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
                

def extract_voxel(get_model, model_path, loss_function, train_path, validation_path, mesh):

    # Read in training and validation files.
    validation_files = [os.path.join(validation_path, filename) for filename in os.listdir(validation_path) if ".tfrecord" in filename]

    sdf_count_ = 2048
    voxel_resolution = 128
    
    # Fetch the data.
    validation_dataset = get_sdf_dataset(validation_files, batch_size=1, sdf_count=sdf_count_)

    # Setup iterators.
    val_iterator = validation_dataset.make_initializable_iterator()
    val_next_point_cloud, val_next_xyz, val_next_label = val_iterator.get_next()
    
    # Setup model operations.
    points = tf.placeholder(tf.float32)
    xyz_in = tf.placeholder(tf.float32)    
    sdf_labels = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)

    sdf_prediction, loss, _ = get_model(points, xyz_in, sdf_labels, is_training, None, batch_size=1, alpha=0.5, loss_function=loss_function, sdf_count=sdf_count_)

    # Generate points to sample.
    pts = []
    for x in range(voxel_resolution):
        for y in range(voxel_resolution):
            for z in range(voxel_resolution):
                x_ = -0.5 + ((1.0 / float(voxel_resolution-1)) * x)
                y_ = -0.5 + ((1.0 / float(voxel_resolution-1)) * y)
                z_ = -0.5 + ((1.0 / float(voxel_resolution-1)) * z)
                pts.append([x_, y_, z_])
    pts = np.array(pts)
    pt_splits = np.split(pts, pts.shape[0] // sdf_count_)
    
    # Save/Restore model.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, os.path.join(model_path, 'model.ckpt'))

        # Setup function that predicts SDF for (x,y,z) given a point cloud.
        def get_sdf(point_cloud, pts):

            prediction = sess.run(sdf_prediction, feed_dict = {
                points: point_cloud, xyz_in: pts, sdf_labels: None, is_training: False,
            })

            # print(xyz)
            # print(prediction)

            return prediction
        
        sess.run(val_iterator.initializer)
        for i in range(20):
            point_clouds_, xyzs_, labels_ = sess.run((val_next_point_cloud, val_next_xyz, val_next_label))
            
            # Setup a voxelization based on the SDF.
            voxelized = np.zeros((voxel_resolution,voxel_resolution,voxel_resolution), dtype=np.float32)

            filled_pts = []

            # For all points sample SDF given the point cloud and include points inside the object to a point cloud.
            for pts_ in pt_splits:
                sdf_ = get_sdf(point_clouds_, np.reshape(pts_, (1,sdf_count_,3)))

                for pt_, sdf in zip(np.reshape(pts_, (sdf_count_,3)), np.reshape(sdf_, (sdf_count_,))):
                    if sdf <= 0.0 and sdf >= -0.05:
                        #filled_pts.append(pt_)
                        x_ = int(round((pt_[0] + 0.5) * float(voxel_resolution-1)))
                        y_ = int(round((pt_[1] + 0.5) * float(voxel_resolution-1)))
                        z_ = int(round((pt_[2] + 0.5) * float(voxel_resolution-1)))
                        voxelized[x_,y_,z_] = 1.0

            # Plot.
            plot_3d_points(point_clouds_[0])
            #plot_3d_points(np.reshape(filled_pts, (-1,3)))
            if mesh:
                # Mesh w/ mcubes.
                #plot_voxel(convert_to_sparse_voxel_grid(voxelized), voxel_res=(voxel_resolution, voxel_resolution, voxel_resolution))
                vertices, triangles = mcubes.marching_cubes(voxelized, 0)
                mcubes.export_mesh(vertices, triangles, 'test.dae', 'test')

                meshed_object = trimesh.load('test.dae')
                meshed_object.show()
