import numpy as np
import time
import os
import tensorflow as tf


def get_voxel_encoder(partial_view, is_training, batch_size=32):

    # Shape up inputs.
    partial_view = tf.reshape(partial_view, shape=(batch_size,32,32,32,1))

    with tf.variable_scope('encoder'):

        # Conv 1
        encode_c1 = tf.layers.conv3d(partial_view, filters=8, kernel_size=3, padding='valid', use_bias=False)
        encode_b1 = tf.layers.batch_normalization(encode_c1, training=is_training)
        encode_e1 = tf.nn.elu(encode_b1)

        # Conv 2
        encode_c2 = tf.layers.conv3d(encode_e1, filters=16, kernel_size=3, padding='same', strides=(2,2,2), use_bias=False)
        encode_b2 = tf.layers.batch_normalization(encode_c2, training=is_training)
        encode_e2 = tf.nn.elu(encode_b2)

        # Conv 3        
        encode_c3 = tf.layers.conv3d(encode_e2, filters=32, kernel_size=3, padding='valid', use_bias=False)
        encode_b3 = tf.layers.batch_normalization(encode_c3, training=is_training)
        encode_e3 = tf.nn.elu(encode_b3)

        # Conv 4        
        encode_c4 = tf.layers.conv3d(encode_e3, filters=64, kernel_size=3, padding='same', strides=(2,2,2), use_bias=False)
        encode_b4 = tf.layers.batch_normalization(encode_c4, training=is_training)
        encode_e4 = tf.nn.elu(encode_b4)

        # Flatten
        encode_flat = tf.layers.flatten(encode_e4)

        # Fully connected. Match PointSDF embedding size.
        latent_d = tf.layers.dense(encode_flat, 256)
        latent_b = tf.layers.batch_normalization(latent_d, training=is_training)
        latent_e = tf.nn.elu(latent_b, name="embedding")
        
    return latent_e

def get_voxel_decoder(embedding, is_training, batch_size=32):

    # Shape up inputs.
    embedding = tf.reshape(embedding, shape=(batch_size, -1))

    with tf.variable_scope('decoder'):

        # Fully connected.
        latent_d = tf.layers.dense(embedding, 343)
        latent_b = tf.layers.batch_normalization(latent_d, training=is_training)
        latent_e = tf.nn.elu(latent_b)

        # Reshape to 3D.
        decode_reshape = tf.reshape(latent_e, shape=(batch_size, 7, 7, 7, 1))

        # Conv 1: Output is 7x7x7.
        decode_c1 = tf.layers.conv3d(decode_reshape, filters=64, kernel_size=3, padding='same', use_bias=False)
        decode_b1 = tf.layers.batch_normalization(decode_c1, training=is_training)
        decode_e1 = tf.nn.elu(decode_b1)

        # Conv 2: output is 15x15x15.
        decode_c2 = tf.layers.conv3d_transpose(decode_e1, filters=32, kernel_size=3, padding='valid', strides=(2,2,2), use_bias=False)
        decode_b2 = tf.layers.batch_normalization(decode_c2, training=is_training)
        decode_e2 = tf.nn.elu(decode_b2)

        # Conv 3: output is 15x15x15.
        decode_c3 = tf.layers.conv3d(decode_e2, filters=16, kernel_size=3, padding='same', use_bias=False)
        decode_b3 = tf.layers.batch_normalization(decode_c3, training=is_training)
        decode_e3 = tf.nn.elu(decode_b3)

        # Conv 4: output is 31x31x31.
        decode_c4 = tf.layers.conv3d_transpose(decode_e3, filters=8, kernel_size=3, padding='valid', strides=(2,2,2), use_bias=False)
        # Pad up to 32x32x32.
        decode_c4_pad = tf.pad(decode_c4, tf.constant([[0,0],[1,0],[1,0],[1,0],[0,0]]))
        decode_b4 = tf.layers.batch_normalization(decode_c4_pad, training=is_training)
        decode_e4 = tf.nn.elu(decode_b4)

        # Conv 5: final output.
        decode_c5 = tf.layers.conv3d(decode_e4, filters=1, kernel_size=3, padding='same')
        decode_b5 = tf.layers.batch_normalization(decode_c5, training=is_training, name='output_logits')
        decode_e5 = tf.nn.sigmoid(decode_b5, name='output_voxel')

    return decode_b5, decode_e5

def get_voxel_cnn_model(partial_view, full_view, is_training, batch_size=32):

    # Shape up inputs.
    partial_view = tf.reshape(partial_view, shape=(batch_size, 32, 32, 32, 1))
    full_view = tf.reshape(full_view, shape=(batch_size, 32, 32, 32, 1))

    # Build model.
    embedding = get_voxel_encoder(partial_view, is_training, batch_size)
    reconstructed_logits, reconstructed_voxel = get_voxel_decoder(embedding, is_training, batch_size)

    # Setup loss.
    loss = tf.losses.sigmoid_cross_entropy(full_view, reconstructed_logits)
    tf.summary.scalar('loss', loss)

    # Collect debug print statements as needed.
    debug = tf.no_op()

    return reconstructed_voxel, loss, debug

def get_voxel_prediction(get_model, model_path):

    # Setup model operations.
    is_training = tf.placeholder(tf.bool, name="is_training")
    partial_view = tf.placeholder(tf.float32, name="partial_voxel")
    full_view = tf.placeholder(tf.float32, name="full_voxel")
    voxel_prediction, loss, debug = get_model(partial_view, full_view, is_training, batch_size=1)

    # Save/Restore model.
    saver = tf.train.Saver()

    sess = tf.Session()
    saver.restore(sess, os.path.join(model_path, 'model.ckpt'))

    def get_voxel(voxel_input):
        prediction = sess.run(voxel_prediction, feed_dict = {
            partial_view: voxel_input, full_view: None, is_training: False,
        })
        return prediction

    return get_voxel
