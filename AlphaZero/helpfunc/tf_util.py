"""
For helping to configure tensorflow
"""


def set_session_config(per_process_gpu_memory_fraction=None, allow_growth=False):

    import tensorflow as tf
    import keras.backend as k

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=per_process_gpu_memory_fraction,
            allow_growth=allow_growth,
        )
    )
    sess = tf.Session(config=config)
    k.set_session(sess)
