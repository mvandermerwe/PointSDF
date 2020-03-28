import tensorflow as tf

def _bytes_feature(value):
    '''
    Returns a bytes_list from a string / byte.
    '''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(partial_voxel, full_voxel):
    '''
    Serialize given partial, full voxel pair.
    '''

    feature = {
        'partial': _bytes_feature(tf.io.serialize_tensor(partial_voxel).numpy()),
        'full': _bytes_feature(tf.io.serialize_tensor(full_voxel).numpy()),
    }

    # Serialize w/ tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize_example(partial_voxel, full_voxel):
    tf_string = tf.py_func(
        serialize_example,
        (partial_voxel, full_voxel),
        tf.string)
    return tf.reshape(tf_string, ())
