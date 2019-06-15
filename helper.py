# Helpers.

import tensorflow as tf
import numpy as np

# BN Decay hyperparameters for PointNet/Conv layers BN.
# Hard-coded, because probably best to leave this as used in PointConv/Net.
_BN_INIT_DECAY = 0.5
_BN_DECAY_DECAY_RATE = 0.5
_BN_DECAY_DECAY_STEP = float(3000)
_BN_DECAY_CLIP = 0.99

def get_bn_decay(batch, batch_size):
    bn_momentum = tf.train.exponential_decay(
        _BN_INIT_DECAY,
        batch * batch_size,
        _BN_DECAY_DECAY_STEP,
        _BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(_BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

# These could be tweaked too?
_DECAY_STEP = float(7200)
_DECAY_RATE = 0.7
_MIN_LEARN_RATE = 1e-7 # ?

def get_learning_rate(batch, batch_size, base_learning_rate):
    learning_rate = tf.train.exponential_decay(
        base_learning_rate,
        batch * batch_size,
        _DECAY_STEP,
        _DECAY_RATE,
        staircase=True)
    learning_rate = tf.maximum(learning_rate, _MIN_LEARN_RATE)
    return learning_rate

def get_num_trainable_variables(scope):
    total_parameters = 0
    for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope):
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters

def shuffle_in_unison(a, b, seed=None):

    np.random.seed(seed)
    
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a,b
