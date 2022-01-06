#%%
import tensorflow as tf
#%%
def box_loss(pred, true):
    return tf.reduce_sum(tf.reduce_sum((pred[...,:2] - true[...,:2])**2, axis=-1) * true[...,4])

def obj_loss(pred, true):
    return tf.reduce_sum(-(true[...,4] * tf.math.log(pred[...,4] + 1e-10) + (1 - true[...,4]) * tf.math.log(1 - pred[...,4] + 1e-10)) * true[...,4])

def nobj_loss(pred, true):
    return tf.reduce_sum(-(true[...,4] * tf.math.log(pred[...,4] + 1e-10) + (1 - true[...,4]) * tf.math.log(1 - pred[...,4] + 1e-10)) * (1-true[...,4]))

def cls_loss(pred, true):
    return tf.reduce_sum(-tf.reduce_sum(true[...,5:] * tf.math.log(pred[...,5:] + 1e-10) + (1 - true[...,5:]) * tf.math.log(1 - pred[...,5:] + 1e-10), axis=-1) * true[...,4])
