import tensorflow as tf


def gpu_memory_growth():
    gpu = tf.config.experimental.list_physical_devices('GPU')
    if gpu:
        try:
            for i in gpu:
                tf.config.experimental.set_memory_growth(i, True)
        except RuntimeError as e:
            print(e)

